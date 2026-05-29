"""
ppo_engine.py: Cluster-scale distributed PPO step executor.

Orchestrates Actor, Critic, Reference, and Reward models across a multi-node cluster
with minimal communication overhead and maximum compute utilization.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .loss import PPOLossFn, compute_advantages, compute_kl_divergence
from .rollout import RolloutBuffer, RolloutCollator
from rlhf_platform.distributed.comm_hooks import CommunicationHooks

logger = logging.getLogger(__name__)


class DistributedPPOEngine:
    """
    Cluster-scale PPO trainer with asymmetric model parallelism.
    
    Manages:
    - Actor (policy) training with gradient accumulation
    - Critic (value network) training with separate loss tracking
    - Reference (frozen) forward passes for KL computation
    - Reward (frozen) inference for trajectory scoring
    """

    def __init__(
        self,
        actor_model: torch.nn.Module,
        critic_model: torch.nn.Module,
        reference_model: torch.nn.Module,
        reward_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        world_size: int,
        rank: int,
        loss_fn: Optional[PPOLossFn] = None,
        gradient_accumulation_steps: int = 4,
    ):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.world_size = world_size
        self.rank = rank
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.loss_fn = loss_fn or PPOLossFn()
        self.rollout_buffer = RolloutBuffer(capacity=1000)
        self.collator = RolloutCollator()
        
        self.step = 0
        self.accumulated_steps = 0
        
        logger.info(
            f"Rank {rank}: DistributedPPOEngine initialized. "
            f"Gradient accumulation: {gradient_accumulation_steps} steps."
        )

        # Register overlap communication hooks when running distributed
        if dist.is_initialized():
            try:
                CommunicationHooks.register_overlap_hook(self.actor_model)
                CommunicationHooks.register_gradient_clipping_hook(self.actor_model)
            except Exception:
                logger.debug("Failed to register DDP comm hooks; ensure model is wrapped in DDP.")

    def collect_rollouts(
        self,
        batch_size: int,
        num_rollouts: int,
    ) -> None:
        """
        Collect rollouts from the policy and store in buffer.
        
        This would typically be called by a background generation process.
        """
        logger.debug(f"Rank {self.rank}: Collecting {num_rollouts} rollouts (batch_size={batch_size}).")
        # Placeholder: in practice, rollouts are generated asynchronously
        # and pushed into self.rollout_buffer

    def ppo_step(self) -> Dict[str, float]:
        """
        Execute a single PPO optimization step.
        
        Returns:
            Dictionary of loss metrics and statistics.
        """
        if self.rollout_buffer.size() == 0:
            logger.warning(f"Rank {self.rank}: No rollouts in buffer. Skipping PPO step.")
            return {}
        
        # Sample batch from rollout buffer
        batch_size = min(32, self.rollout_buffer.size())
        rollouts = self.rollout_buffer.sample_batch(batch_size)
        batch = self.collator(rollouts)
        
        # Move to device
        device = next(self.actor_model.parameters()).device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        
        self.optimizer.zero_grad()
        
        input_ids = torch.cat([batch["query_tokens"], batch["response_tokens"]], dim=1)
        
        actor_out = self.actor_model(input_ids)
        critic_out = self.critic_model(input_ids)
        
        with torch.no_grad():
            reference_out = self.reference_model(input_ids)
        
        actor_logits = actor_out.logits[:, -batch["response_tokens"].shape[1]:, :]
        critic_values = critic_out.logits.squeeze(-1) if critic_out.logits.dim() > 1 else critic_out.logits
        
        new_log_probs = F.log_softmax(actor_logits, dim=-1)
        old_log_probs = F.log_softmax(batch["logits_policy"], dim=-1)
        
        # Compute advantages and returns
        advantages, returns = compute_advantages(
            batch["rewards"].unsqueeze(1).expand(-1, new_log_probs.shape[1]),
            critic_values,
            gamma=0.99,
            gae_lambda=0.95,
            normalize=True,
        )
        
        # Compute KL divergence
        kl_div = compute_kl_divergence(
            actor_logits,
            batch["logits_reference"],
            reduction="mean",
        )
        
        # Compute PPO loss
        loss, loss_dict = self.loss_fn(
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            logits_policy=actor_logits,
            logits_reference=batch["logits_reference"],
            values=critic_values,
            old_values=critic_values.detach(),  # No clipping for now
            rewards=batch["rewards"].unsqueeze(1).expand(-1, advantages.shape[1]),
        )
        
        # Backward pass with gradient accumulation
        (loss / self.gradient_accumulation_steps).backward()
        
        self.accumulated_steps += 1
        
        # Optimizer step every N accumulation steps
        if self.accumulated_steps >= self.gradient_accumulation_steps:
            # Clip gradients before optimizer step
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.accumulated_steps = 0
            self.step += 1
            
            logger.debug(f"Rank {self.rank}: Optimizer step {self.step}. KL: {kl_div:.4f}")
        
        # Synchronize metrics across ranks
        loss_dict = self._allgather_loss_metrics(loss_dict)
        # Numerical safety: ensure metrics are finite
        for k, v in list(loss_dict.items()):
            if not isinstance(v, float) or (isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf"))):
                loss_dict[k] = 0.0
        
        return loss_dict

    def _allgather_loss_metrics(self, loss_dict: Dict[str, float]) -> Dict[str, float]:
        """Aggregate loss metrics across all ranks."""
        if self.world_size == 1:
            return loss_dict
        
        try:
            # Convert dict values to tensor, allgather, and convert back
            aggregated = {}
            for key, val in loss_dict.items():
                tensor_val = torch.tensor(val, device=next(self.actor_model.parameters()).device)
                dist.all_reduce(tensor_val, op=dist.ReduceOp.MEAN)
                aggregated[key] = tensor_val.item()
            return aggregated
        except Exception as e:
            logger.warning(f"Failed to aggregate metrics: {e}. Using local metrics.")
            return loss_dict

    def get_stats(self) -> Dict[str, float]:
        """Get current training statistics."""
        return {
            "ppo_step": self.step,
            "rollouts_in_buffer": self.rollout_buffer.size(),
            "accumulated_steps": self.accumulated_steps,
        }
