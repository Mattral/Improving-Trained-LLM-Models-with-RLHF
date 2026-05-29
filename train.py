#!/usr/bin/env python3
"""Entrypoint for the RLHF distributed training platform."""

import argparse
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rlhf_platform.alignment.ppo_engine import DistributedPPOEngine
from rlhf_platform.alignment.rollout import AsyncRolloutPipeline, RolloutBuffer, RolloutGenerator
from rlhf_platform.distributed.async_io import DistributedCheckpointManager
from rlhf_platform.distributed.comm_hooks import CommunicationHooks
from rlhf_platform.distributed.topology import load_topology_from_yaml
from rlhf_platform.utils.telemetry import RankAwareTelemetry


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rlhf_train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLHF distributed training entrypoint.")
    parser.add_argument("--config", default="configs/cluster_topology.yaml", help="Path to cluster topology YAML config.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for async checkpoints.")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of PPO update steps to execute.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for rollout generation.")
    parser.add_argument("--max-response-length", type=int, default=64, help="Max generated token length for each rollout.")
    parser.add_argument("--prompts", nargs="*", default=None, help="Optional list of prompts to use for generation.")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU execution rather than GPU/distributed.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def init_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    """Initialize distributed process group with comprehensive validation."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    logger.info(f"Rank {rank}/{world_size}: Distributed environment: WORLD_SIZE={world_size}, RANK={rank}, LOCAL_RANK={local_rank}")

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() and not args.use_cpu else "gloo"
        logger.info(f"Rank {rank}: Initializing process group with backend={backend}")
        try:
            dist.init_process_group(backend=backend, init_method="env://")
            logger.info(f"Rank {rank}: Process group initialized successfully.")
            
            # Verify connectivity
            dist.barrier()
            logger.info(f"Rank {rank}: All ranks synchronized at barrier.")
        except Exception as e:
            logger.error(f"Rank {rank}: Failed to initialize process group: {e}")
            raise
    else:
        logger.info("Rank 0: Running in single-process mode (world_size=1).")

    return world_size, rank, local_rank


def create_query_sampler(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    batch_size: int,
    device: torch.device,
    max_prompt_length: int = 64,
) -> callable:
    if not prompts:
        prompts = [
            "Summarize the following paragraph.",
            "Explain the intent of the user request.",
            "Write a short answer describing why RLHF improves alignment.",
        ]

    def _pad_to_batch(encodings: List[torch.Tensor]) -> torch.Tensor:
        max_length = max(t.shape[0] for t in encodings)
        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        batch = torch.full((len(encodings), max_length), pad_token, dtype=torch.long, device=device)
        for i, tensor in enumerate(encodings):
            batch[i, : tensor.shape[0]] = tensor
        return batch

    def sampler() -> torch.Tensor:
        encodings = []
        for _ in range(batch_size):
            prompt = random.choice(prompts)
            encoding = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_length,
            )
            encodings.append(encoding.input_ids[0])
        return _pad_to_batch(encodings)

    return sampler


def build_models(
    actor_name: str,
    critic_name: str,
    reference_name: str,
    reward_name: str,
    device: torch.device,
    use_fp16: bool = True,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module, AutoTokenizer]:
    dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(actor_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor_model = AutoModelForCausalLM.from_pretrained(actor_name).to(device, dtype=dtype)
    reference_model = AutoModelForCausalLM.from_pretrained(reference_name).to(device, dtype=torch.float32)
    critic_model = AutoModelForSequenceClassification.from_pretrained(
        critic_name,
        num_labels=1,
        problem_type="regression",
    ).to(device, dtype=torch.float32)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_name,
        num_labels=1,
        problem_type="regression",
    ).to(device, dtype=torch.float32)

    reference_model.eval()
    reward_model.eval()
    for model in [reference_model, reward_model]:
        for param in model.parameters():
            param.requires_grad = False

    return actor_model, critic_model, reference_model, reward_model, tokenizer


def wrap_distributed(
    model: torch.nn.Module,
    process_group,
    rank: int,
    local_rank: int,
    use_cuda: bool,
) -> torch.nn.Module:
    if process_group is None:
        return model

    if use_cuda:
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            process_group=process_group,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        ddp_model = DistributedDataParallel(
            model,
            device_ids=None,
            process_group=process_group,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    logger.info(f"Wrapped model {model.__class__.__name__} in DistributedDataParallel on rank {rank}.")
    return ddp_model


def get_state_dict(model: torch.nn.Module) -> dict:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def main() -> None:
    global args
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Adjust logging verbosity
    if args.verbose:
        logging.getLogger("rlhf_train").setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    logger.info("\n" + "=" * 80)
    logger.info("RLHF Distributed Training Entrypoint")
    logger.info("=" * 80)

    # Stage 1: Distributed initialization
    logger.info("\n[Stage 1/5] Initializing distributed process group...")
    world_size, rank, local_rank = init_distributed(args)
    use_cuda = torch.cuda.is_available() and not args.use_cpu
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    logger.info(f"Rank {rank}: Device selection: device={device}, use_cuda={use_cuda}")

    # Stage 2: Topology initialization
    logger.info("\n[Stage 2/5] Loading cluster topology and initializing process groups...")
    try:
        topology = load_topology_from_yaml(args.config, rank=rank, local_rank=local_rank)
        logger.info(f"Rank {rank}: Topology loaded successfully.")
        logger.info(f"Rank {rank}: Actor placement: {topology.actor_placement.device_list if topology.actor_placement else 'None'}")
        logger.info(f"Rank {rank}: Critic placement: {topology.critic_placement.device_list if topology.critic_placement else 'None'}")
        logger.info(f"Rank {rank}: Reference placement: {topology.reference_placement.device_list if topology.reference_placement else 'None'}")
        logger.info(f"Rank {rank}: Reward placement: {topology.reward_placement.device_list if topology.reward_placement else 'None'}")
    except Exception as e:
        logger.error(f"Rank {rank}: Failed to load topology: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise

    if not dist.is_initialized():
        logger.warning(f"Rank {rank}: Distributed not initialized; topology groups will not be created.")

    # Stage 3: Model initialization
    logger.info("\n[Stage 3/5] Loading actor, critic, reference, and reward models...")
    try:
        if not topology.actor_placement or not topology.critic_placement or not topology.reference_placement or not topology.reward_placement:
            raise ValueError("Incomplete topology placements in config.")
        
        actor_model, critic_model, reference_model, reward_model, tokenizer = build_models(
            actor_name=topology.actor_placement.model_name,
            critic_name=topology.critic_placement.model_name,
            reference_name=topology.reference_placement.model_name,
            reward_name=topology.reward_placement.model_name,
            device=device,
            use_fp16=True,
        )
        logger.info(f"Rank {rank}: All models loaded successfully.")
    except Exception as e:
        logger.error(f"Rank {rank}: Failed to load models: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    
    # Synchronize after model loading
    if dist.is_initialized():
        dist.barrier()
        logger.info(f"Rank {rank}: Model loading barrier synchronized.")

    # Stage 4: Distributed model wrapping and communication hook registration
    logger.info("\n[Stage 4/5] Wrapping models in DDP and registering communication hooks...")
    if dist.is_initialized() and topology.is_active_rank():
        logger.info(f"Rank {rank}: Wrapping actor and critic in DDP (active rank).")
        actor_model = wrap_distributed(actor_model, topology.actor_group, rank, local_rank, use_cuda)
        critic_model = wrap_distributed(critic_model, topology.critic_group, rank, local_rank, use_cuda)
        
        try:
            CommunicationHooks.register_overlap_hook(actor_model)
            logger.info(f"Rank {rank}: Registered gradient bucket overlap hook.")
        except Exception as e:
            logger.warning(f"Rank {rank}: Could not register overlap hook: {e}")
        
        try:
            CommunicationHooks.register_gradient_clipping_hook(actor_model)
            logger.info(f"Rank {rank}: Registered gradient clipping hook.")
        except Exception as e:
            logger.warning(f"Rank {rank}: Could not register gradient clipping hook: {e}")
        
        try:
            CommunicationHooks.register_nan_check_hook(actor_model)
            logger.info(f"Rank {rank}: Registered NaN detection hook.")
        except Exception as e:
            logger.warning(f"Rank {rank}: Could not register NaN check hook: {e}")
    else:
        logger.info(f"Rank {rank}: Rank is in inference group; skipping DDP wrapping.")

    # Create optimizer
    actor_params = list(actor_model.parameters())
    critic_params = list(critic_model.parameters())
    optimizer = AdamW(
        actor_params + critic_params,
        lr=1e-5,
        eps=1e-8,
    )
    logger.info(f"Rank {rank}: Optimizer created with {len(actor_params) + len(critic_params)} parameters.")

    # Stage 5: Rollout pipeline and training engine initialization
    logger.info("\n[Stage 5/5] Initializing async rollout pipeline and PPO engine...")
    sampler = create_query_sampler(tokenizer, args.prompts or [], args.batch_size, device)
    buffer = RolloutBuffer(capacity=512, device=device.type)
    generator = RolloutGenerator(
        policy_model=actor_model.module if hasattr(actor_model, "module") else actor_model,
        reference_model=reference_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        max_response_length=args.max_response_length,
    )
    logger.info(f"Rank {rank}: Rollout generator initialized.")
    
    pipeline = AsyncRolloutPipeline(
        generator=generator,
        buffer=buffer,
        query_sampler=sampler,
        batch_size=args.batch_size,
        num_generator_threads=2,
    )
    pipeline.start_generators()
    logger.info(f"Rank {rank}: Async rollout pipeline started with 2 generator threads.")

    ppo_engine = DistributedPPOEngine(
        actor_model=actor_model,
        critic_model=critic_model,
        reference_model=reference_model,
        reward_model=reward_model,
        optimizer=optimizer,
        world_size=world_size,
        rank=rank,
        gradient_accumulation_steps=8,
    )
    logger.info(f"Rank {rank}: PPO engine initialized. Target steps: {args.num_steps}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = DistributedCheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        world_size=world_size,
        rank=rank,
        save_frequency=100,
    )
    logger.info(f"Rank {rank}: Checkpoint manager initialized. Directory: {checkpoint_dir.resolve()}")

    telemetry = RankAwareTelemetry(rank=rank, world_size=world_size, log_file=None)
    logger.info(f"Rank {rank}: Telemetry initialized.")
    
    # Final barrier before training
    if dist.is_initialized():
        dist.barrier()
    logger.info(f"Rank {rank}: Initialization complete. Starting training loop.")
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING LOOP")
    logger.info("=" * 80 + "\n")

    try:
        while ppo_engine.step < args.num_steps:
            # Wait for rollouts to populate buffer
            if buffer.size() < max(1, args.batch_size):
                if args.verbose and ppo_engine.step % 10 == 0:
                    logger.debug(f"Rank {rank}: Waiting for rollouts. Buffer size: {buffer.size()}")
                time.sleep(0.2)
                continue

            # Execute PPO step
            start_ts = time.time()
            try:
                metrics = ppo_engine.ppo_step()
            except Exception as e:
                logger.error(f"Rank {rank}: PPO step failed at iteration {ppo_engine.step}: {e}")
                raise
            
            duration_ms = (time.time() - start_ts) * 1000.0

            # Log metrics
            if metrics:
                telemetry.log_ppo_step(
                    step=ppo_engine.step,
                    actor_loss=metrics.get("actor_loss", 0.0),
                    value_loss=metrics.get("value_loss", 0.0),
                    kl_divergence=metrics.get("kl_divergence", 0.0),
                    policy_entropy=metrics.get("policy_entropy", 0.0),
                    duration_ms=duration_ms,
                )
                if rank == 0 and ppo_engine.step % 50 == 0:
                    logger.info(
                        f"Step {ppo_engine.step:4d} | "
                        f"actor_loss={metrics.get('actor_loss', 0.0):.4f} | "
                        f"value_loss={metrics.get('value_loss', 0.0):.4f} | "
                        f"kl={metrics.get('kl_divergence', 0.0):.4f} | "
                        f"entropy={metrics.get('policy_entropy', 0.0):.4f} | "
                        f"time={duration_ms:.1f}ms"
                    )

            # Periodic checkpointing
            if rank == 0 and ppo_engine.step % 100 == 0 and ppo_engine.step > 0:
                logger.info(f"Rank {rank}: Saving checkpoint at step {ppo_engine.step}.")
                checkpoint_manager.save_checkpoint(
                    step=ppo_engine.step,
                    model_name=topology.actor_placement.model_name,
                    model_state_dict=get_state_dict(actor_model),
                    optimizer_state_dict=optimizer.state_dict(),
                    metrics=metrics,
                )

            # Memory monitoring
            if rank == 0 and ppo_engine.step and ppo_engine.step % 10 == 0:
                allocated = torch.cuda.memory_allocated(device) / 1024.0 / 1024.0 if use_cuda else 0.0
                reserved = torch.cuda.memory_reserved(device) / 1024.0 / 1024.0 if use_cuda else 0.0
                max_allocated = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0 if use_cuda else 0.0
                telemetry.log_memory_usage(
                    allocated_mb=allocated,
                    reserved_mb=reserved,
                    max_allocated_mb=max_allocated,
                )
                if args.verbose:
                    logger.debug(
                        f"Rank {rank}: Memory: allocated={allocated:.1f}MB, "
                        f"reserved={reserved:.1f}MB, max={max_allocated:.1f}MB"
                    )

    except KeyboardInterrupt:
        logger.info(f"Rank {rank}: Interrupted by user; attempting graceful shutdown.")
    except Exception as e:
        logger.error(f"Rank {rank}: Training loop failed: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info(f"Rank {rank}: Cleaning up resources.")
        pipeline.stop_generators()
        checkpoint_manager.flush_and_shutdown()
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
                logger.info(f"Rank {rank}: Process group destroyed.")
            except Exception as e:
                logger.warning(f"Rank {rank}: Error destroying process group: {e}")

    logger.info(f"Rank {rank}: Training complete. Final step: {ppo_engine.step}/{args.num_steps}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
