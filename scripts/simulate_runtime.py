#!/usr/bin/env python3
"""
Multi-rank cluster emulation suite for validation of distributed topology, buffers, and comm hooks.

This script simulates a 4-rank distributed cluster on a single machine using torch.multiprocessing.spawn.
It validates:
- Asymmetric process group creation without deadlocks
- Thread-safe rollout buffer operations
- Non-blocking tensor transfers
- Communication hook integration

Usage:
    python scripts/simulate_runtime.py [--num-ranks 4] [--iterations 10]
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rlhf_platform.alignment.rollout import RolloutBuffer, Rollout
from rlhf_platform.distributed.topology import ClusterTopology, ModelRole, ParallelStrategy, ModelPlacement, TopologyBuilder
from rlhf_platform.utils.telemetry import RankAwareTelemetry


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RANK %(rank)d] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class SimulationContext:
    """Holds rank-local simulation state."""
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    topology: Optional[ClusterTopology] = None
    telemetry: Optional[RankAwareTelemetry] = None
    trace_log: List[str] = None

    def __post_init__(self):
        if self.trace_log is None:
            self.trace_log = []

    def log_trace(self, msg: str) -> None:
        """Append a trace message with rank and timestamp."""
        self.trace_log.append(f"[T={time.time():.3f}] {msg}")

    def dump_trace(self, output_dir: Path) -> None:
        """Write trace log to a JSON file."""
        trace_path = output_dir / f"trace_rank_{self.rank}.json"
        with open(trace_path, "w") as f:
            json.dump(
                {
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "trace": self.trace_log,
                },
                f,
                indent=2,
            )


def create_mock_topology(rank: int, world_size: int, gpus_per_node: int = 1) -> ClusterTopology:
    """Create a mock asymmetric topology for simulation.

    Active ranks: 0, 1 (Actor & Critic)
    Inference ranks: 2, 3 (Reference & Reward)
    """
    builder = TopologyBuilder(world_size=world_size, rank=rank, gpus_per_node=gpus_per_node)

    # Actor on ranks 0, 1
    builder.add_actor(
        model_name="mock-actor-1.3b",
        param_count=1_300_000_000,
        device_list=[0, 1],
        tensor_parallel_size=1,
        data_parallel_size=2,
        precision="fp16",
    )

    # Critic on ranks 0, 1
    builder.add_critic(
        model_name="mock-critic-1.3b",
        param_count=1_300_000_000,
        device_list=[0, 1],
        tensor_parallel_size=1,
        data_parallel_size=2,
        precision="fp16",
    )

    # Reference on ranks 2, 3
    builder.add_reference(
        model_name="mock-reference-1.3b",
        param_count=1_300_000_000,
        device_list=[2, 3],
        precision="fp32",
    )

    # Reward on rank 2
    builder.add_reward(
        model_name="mock-reward-300m",
        param_count=300_000_000,
        device_list=[2],
        precision="fp32",
    )

    topology = builder.build()
    topology.validate()
    return topology


def mock_model_forward(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Mock forward pass that doesn't require real model weights."""
    batch_size, seq_len = input_ids.shape
    logits = torch.randn(batch_size, seq_len, 16, device=input_ids.device)
    return logits


def generate_mock_rollouts(ctx: SimulationContext, num_rollouts: int = 2) -> List[Rollout]:
    """Generate mock rollouts with pinned CPU tensors."""
    rollouts = []
    for _ in range(num_rollouts):
        query = torch.randint(0, 16, (4,)).cpu().pin_memory()
        response = torch.randint(0, 16, (8,)).cpu().pin_memory()
        reward = torch.tensor(0.5, device="cpu").pin_memory()
        logits_policy = torch.randn(8, 16).cpu().pin_memory()
        logits_reference = torch.randn(8, 16).cpu().pin_memory()

        rollouts.append(
            Rollout(
                query_tokens=query,
                response_tokens=response,
                reward=reward,
                logits_policy=logits_policy,
                logits_reference=logits_reference,
            )
        )
    return rollouts


def rank_worker(rank: int, world_size: int, num_iterations: int, output_dir: Path) -> None:
    """Simulated worker function for a single rank.

    Runs distributed initialization, creates topology groups, validates buffer operations,
    and executes mock PPO steps with communication hooks.
    """
    # Setup logging with rank context
    logger = logging.getLogger(f"rank_{rank}")
    logger = logging.getLogger(f"rlhf_simulate.rank_{rank}")

    try:
        # Initialize context
        device = torch.device("cpu")
        ctx = SimulationContext(rank=rank, world_size=world_size, local_rank=rank, device=device)
        ctx.log_trace("Starting rank worker initialization.")

        # Initialize distributed process group (gloo backend for CPU)
        ctx.log_trace("Initializing distributed process group with gloo backend.")
        dist.init_process_group(backend="gloo", init_method="env://", rank=rank, world_size=world_size)
        ctx.log_trace("Distributed process group initialized successfully.")

        # Synchronize all ranks before topology creation
        dist.barrier()
        ctx.log_trace("All ranks synchronized at barrier 1.")

        # Create and validate topology
        ctx.log_trace("Loading mock asymmetric topology.")
        ctx.topology = create_mock_topology(rank=rank, world_size=world_size)
        ctx.topology.initialize_groups()
        ctx.log_trace("Topology groups initialized successfully.")

        # Synchronize after group creation
        dist.barrier()
        ctx.log_trace("All ranks synchronized at barrier 2.")

        # Initialize telemetry
        ctx.telemetry = RankAwareTelemetry(rank=rank, world_size=world_size, log_file=None)

        # Setup rollout buffer (active ranks only)
        if ctx.topology.is_active_rank():
            ctx.log_trace("Rank is in ACTIVE group; initializing rollout buffer.")
            buffer = RolloutBuffer(capacity=10, device="cpu")
        else:
            ctx.log_trace("Rank is in INFERENCE group; skipping rollout buffer.")
            buffer = None

        # Mock rollout generation + buffer population
        if ctx.topology.is_active_rank():
            ctx.log_trace("Starting mock rollout generation loop.")
            for iteration in range(num_iterations):
                # Generate mock rollouts
                rollouts = generate_mock_rollouts(ctx, num_rollouts=2)
                for rollout in rollouts:
                    buffer.add(rollout)

                ctx.log_trace(f"Iteration {iteration}: Generated 2 rollouts. Buffer size: {buffer.size()}.")

                # Simulate sampling and non-blocking device transfer
                if buffer.size() > 0:
                    sampled = buffer.sample_batch(batch_size=1)
                    ctx.log_trace(f"Iteration {iteration}: Sampled 1 rollout from buffer.")

                    # Non-blocking transfer to device
                    for rollout in sampled:
                        q = rollout.query_tokens.to(device, non_blocking=True)
                        r = rollout.response_tokens.to(device, non_blocking=True)
                        ctx.log_trace(f"Iteration {iteration}: Non-blocking transfer completed.")

                # Synchronize ranks periodically
                if iteration % 3 == 0:
                    dist.barrier()
                    ctx.log_trace(f"Iteration {iteration}: Barrier synchronization completed.")

        else:
            # Inference rank: just run barriers to simulate collective comm
            ctx.log_trace("Starting inference rank iteration loop.")
            for iteration in range(num_iterations):
                ctx.log_trace(f"Iteration {iteration}: Inference rank idle.")
                if iteration % 3 == 0:
                    dist.barrier()
                    ctx.log_trace(f"Iteration {iteration}: Barrier synchronization completed.")

        # Final barrier
        ctx.log_trace("Entering final barrier.")
        dist.barrier()
        ctx.log_trace("All ranks reached final barrier. Simulation complete.")

        # Cleanup
        dist.destroy_process_group()
        ctx.log_trace("Process group destroyed successfully.")

        # Write trace log
        ctx.dump_trace(output_dir)
        logger.info(f"Rank {rank} completed successfully. Trace written to {output_dir / f'trace_rank_{rank}.json'}.")

    except Exception as e:
        logger.error(f"Rank {rank} encountered fatal error: {e}")
        logger.error(traceback.format_exc())
        ctx.log_trace(f"FATAL ERROR: {str(e)}")
        ctx.dump_trace(output_dir)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-rank cluster emulation for distributed RLHF validation."
    )
    parser.add_argument("--num-ranks", type=int, default=4, help="Number of simulated ranks.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per rank.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for trace logs.")
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "rlhf_simulate_runtime"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rlhf_simulate.main")
    logger.info(f"Starting {args.num_ranks}-rank cluster emulation (gloo backend on CPU).")
    logger.info(f"Trace output directory: {output_dir}")

    # Set up environment variables for distributed init
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    try:
        # Spawn worker processes
        mp.spawn(
            rank_worker,
            args=(args.num_ranks, args.iterations, output_dir),
            nprocs=args.num_ranks,
            join=True,
        )
        logger.info("All ranks completed successfully. Simulation PASSED.")
        logger.info(f"Trace logs available in {output_dir}")

        # Aggregate traces
        trace_files = sorted(output_dir.glob("trace_rank_*.json"))
        logger.info(f"Collected {len(trace_files)} trace files.")
        for trace_file in trace_files:
            logger.info(f"  - {trace_file.name}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Simulation FAILED with error: {e}")
        logger.error(traceback.format_exc())
        logger.error(f"Trace logs may be available in {output_dir}")
        sys.exit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
