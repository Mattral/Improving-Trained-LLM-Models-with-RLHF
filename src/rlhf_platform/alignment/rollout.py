"""
rollout.py: Parallelized rollout generator with shared-memory ring buffers.

Enables decoupled generation and optimization: a subset of worker nodes
continuously generate rollout sequences and stream them into an async
shared-memory CPU ring buffer, while core training nodes pull from this
buffer without waiting for generation phases to complete.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


QuerySampler = Callable[[], torch.Tensor]


@dataclass
class Rollout:
    """Single PPO rollout trajectory."""
    query_tokens: torch.Tensor
    response_tokens: torch.Tensor
    reward: torch.Tensor
    logits_policy: torch.Tensor
    logits_reference: torch.Tensor


class RolloutBuffer:
    """In-memory ring buffer for collecting rollouts during generation phase."""

    def __init__(self, capacity: int = 1000, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer: list = []
        self.ptr = 0
        self._lock = threading.Lock()

    def add(self, rollout: Rollout) -> None:
        """Add a rollout to the buffer (overwrites oldest if full)."""
        # Convert tensors to pinned CPU memory to enable non-blocking device copies
        def _pin(t: torch.Tensor) -> torch.Tensor:
            if not isinstance(t, torch.Tensor):
                return t
            return t.detach().cpu().pin_memory()

        pinned = Rollout(
            query_tokens=_pin(rollout.query_tokens),
            response_tokens=_pin(rollout.response_tokens),
            reward=_pin(rollout.reward),
            logits_policy=_pin(rollout.logits_policy),
            logits_reference=_pin(rollout.logits_reference),
        )

        with self._lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(pinned)
            else:
                self.buffer[self.ptr] = pinned
                self.ptr = (self.ptr + 1) % self.capacity

    def sample_batch(self, batch_size: int) -> list:
        """Sample a random batch of rollouts."""
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot sample from empty rollout buffer.")
        
        import random
        with self._lock:
            return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def sample_batch_to_device(self, batch_size: int, device: torch.device) -> list:
        """Sample a batch and move tensors to `device` using non-blocking transfers.

        Returns a list of Rollout objects resident on `device`.
        """
        sampled = self.sample_batch(batch_size)

        moved = []
        for r in sampled:
            moved.append(
                Rollout(
                    query_tokens=r.query_tokens.to(device, non_blocking=True),
                    response_tokens=r.response_tokens.to(device, non_blocking=True),
                    reward=r.reward.to(device, non_blocking=True),
                    logits_policy=r.logits_policy.to(device, non_blocking=True),
                    logits_reference=r.logits_reference.to(device, non_blocking=True),
                )
            )
        return moved

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.ptr = 0

    def size(self) -> int:
        """Current number of rollouts in buffer."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) >= self.capacity


class RolloutDataset(Dataset):
    """PyTorch Dataset wrapper for rollout buffer."""

    def __init__(self, buffer: RolloutBuffer):
        self.buffer = buffer

    def __len__(self) -> int:
        return self.buffer.size()

    def __getitem__(self, idx: int) -> Rollout:
        return self.buffer.buffer[idx]


class RolloutCollator:
    """Collate rollouts into batched tensors."""

    def __init__(self, max_response_length: int = 128):
        self.max_response_length = max_response_length

    def __call__(self, rollouts: list) -> dict:
        """Collate a list of Rollout objects."""
        batch_size = len(rollouts)
        
        # Pad response tokens
        response_tokens_list = [r.response_tokens for r in rollouts]
        padded_responses = self._pad_sequence(response_tokens_list, self.max_response_length)
        
        # Pad logits
        logits_policy_list = [r.logits_policy for r in rollouts]
        logits_reference_list = [r.logits_reference for r in rollouts]
        
        padded_logits_policy = self._pad_logits(logits_policy_list, self.max_response_length)
        padded_logits_reference = self._pad_logits(logits_reference_list, self.max_response_length)
        
        # Rewards
        rewards = torch.stack([r.reward for r in rollouts])
        
        query_tokens_list = [r.query_tokens for r in rollouts]
        max_query_length = max(token.shape[0] for token in query_tokens_list)
        padded_queries = self._pad_sequence(query_tokens_list, max_query_length)
        query_lengths = torch.tensor([t.shape[0] for t in query_tokens_list], dtype=torch.long)

        response_lengths = torch.tensor([r.response_tokens.shape[0] for r in rollouts], dtype=torch.long)

        return {
            "query_tokens": padded_queries,
            "response_tokens": padded_responses,
            "logits_policy": padded_logits_policy,
            "logits_reference": padded_logits_reference,
            "rewards": rewards,
            "query_lengths": query_lengths,
            "response_lengths": response_lengths,
        }

    @staticmethod
    def _pad_sequence(sequences: list, max_length: int) -> torch.Tensor:
        """Pad token sequences to max_length."""
        padded = torch.zeros(len(sequences), max_length, dtype=torch.long)
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_length)
            padded[i, :seq_len] = seq[:seq_len]
        return padded

    @staticmethod
    def _pad_logits(logits_list: list, max_length: int) -> torch.Tensor:
        """Pad logits to (batch, max_length, vocab_size)."""
        vocab_size = logits_list[0].shape[-1]
        padded = torch.zeros(len(logits_list), max_length, vocab_size)
        for i, logits in enumerate(logits_list):
            seq_len = min(len(logits), max_length)
            padded[i, :seq_len] = logits[:seq_len]
        return padded


class RolloutGenerator:
    """Generate rollouts from policy model and score with reward model."""

    def __init__(
        self,
        policy_model: torch.nn.Module,
        reference_model: torch.nn.Module,
        reward_model: torch.nn.Module,
        tokenizer,
        max_response_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.temperature = temperature
        self.top_p = top_p

    @torch.no_grad()
    def generate_rollout(self, query_tokens: torch.Tensor) -> Rollout:
        """Generate a single rollout trajectory."""
        batch_size = query_tokens.shape[0]
        device = query_tokens.device
        
        # Generate response tokens autoregressively
        response_tokens_list = []
        logits_policy_list = []
        logits_reference_list = []

        current_tokens = query_tokens.clone()

        for _ in range(self.max_response_length):
            # Policy forward pass
            policy_out = self.policy_model(current_tokens)
            policy_logits = policy_out.logits[:, -1, :]

            # Reference forward pass
            ref_out = self.reference_model(current_tokens)
            ref_logits = ref_out.logits[:, -1, :]

            logits_policy_list.append(policy_logits)
            logits_reference_list.append(ref_logits)

            next_token = self._sample_token(
                policy_logits,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            response_tokens_list.append(next_token)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)

        response_tokens = torch.cat(response_tokens_list, dim=1)
        logits_policy = torch.stack(logits_policy_list, dim=1)
        logits_reference = torch.stack(logits_reference_list, dim=1)

        with torch.no_grad():
            reward_input = torch.cat([query_tokens, response_tokens], dim=1)
            reward_out = self.reward_model(reward_input)
            rewards = self._decode_reward(reward_out)

        rollouts = []
        for index in range(query_tokens.shape[0]):
            rollouts.append(
                Rollout(
                    query_tokens=query_tokens[index].detach().cpu(),
                    response_tokens=response_tokens[index].detach().cpu(),
                    logits_policy=logits_policy[index].detach().cpu(),
                    logits_reference=logits_reference[index].detach().cpu(),
                    reward=rewards[index].detach().cpu(),
                )
            )

        return rollouts

    @staticmethod
    def _decode_reward(reward_out: object) -> torch.Tensor:
        if hasattr(reward_out, "logits"):
            logits = reward_out.logits
            if logits.dim() == 2:
                return logits.squeeze(-1)
            if logits.dim() == 3:
                return logits.mean(dim=-1).mean(dim=-1)

        if hasattr(reward_out, "pooler_output"):
            return reward_out.pooler_output.mean(dim=-1)

        raise RuntimeError("Unsupported reward model output shape")

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Sample next token using temperature + top-p (nucleus) sampling."""
        scaled_logits = logits / temperature
        if top_p < 1.0:
            probs = F.softmax(scaled_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumulative_probs > top_p
            cutoff_mask[..., 0] = False

            remove_mask = torch.zeros_like(logits, dtype=torch.bool)
            remove_mask.scatter_(dim=-1, index=sorted_indices, src=cutoff_mask)
            scaled_logits = scaled_logits.masked_fill(remove_mask, float("-inf"))

        token = torch.multinomial(F.softmax(scaled_logits, dim=-1), num_samples=1)
        return token.squeeze(-1)


class AsyncRolloutPipeline:
    """
    Asynchronous rollout pipeline: decouples generation from optimization.
    
    Generator threads continuously produce rollouts and feed them into
    a shared RolloutBuffer. Optimization threads pull from the buffer
    without blocking on generation latency.
    """

    def __init__(
        self,
        generator: RolloutGenerator,
        buffer: RolloutBuffer,
        query_sampler: Optional[QuerySampler] = None,
        batch_size: int = 1,
        num_generator_threads: int = 2,
    ):
        self.generator = generator
        self.buffer = buffer
        self.query_sampler = query_sampler
        self.batch_size = batch_size
        self.num_generator_threads = num_generator_threads
        self.stop_event = threading.Event()
        self.generator_threads: List[threading.Thread] = []

    def start_generators(self) -> None:
        """Start background generator threads."""
        import threading
        self.generator_threads = []
        for i in range(self.num_generator_threads):
            thread = threading.Thread(target=self._generator_loop, args=(i,), daemon=True)
            thread.start()
            self.generator_threads.append(thread)
        logger.info(f"Started {self.num_generator_threads} generator threads.")

    def _generator_loop(self, thread_id: int) -> None:
        """Run generation loop in background thread."""
        logger.info(f"Generator thread {thread_id} started.")
        while not self.stop_event.is_set():
            try:
                if self.query_sampler is None:
                    logger.warning("No query sampler configured for AsyncRolloutPipeline.")
                    break

                query_batch = self.query_sampler()
                if query_batch is None:
                    logger.debug("Query sampler returned no input; sleeping briefly.")
                    import time
                    time.sleep(0.5)
                    continue

                # Move queries to generator device for fast generation
                gen_device = next(self.generator.policy_model.parameters()).device
                query_batch = query_batch.to(gen_device)

                # Generate rollouts on model device
                rollouts = self.generator.generate_rollout(query_batch)

                # Pin rollouts and enqueue without blocking training threads
                for rollout in rollouts:
                    # Ensure each rollout's tensors are pinned CPU before enqueue
                    pinned = Rollout(
                        query_tokens=rollout.query_tokens.detach().cpu().pin_memory(),
                        response_tokens=rollout.response_tokens.detach().cpu().pin_memory(),
                        reward=rollout.reward.detach().cpu().pin_memory(),
                        logits_policy=rollout.logits_policy.detach().cpu().pin_memory(),
                        logits_reference=rollout.logits_reference.detach().cpu().pin_memory(),
                    )
                    self.buffer.add(pinned)

                if self.buffer.is_full():
                    import time
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Generator thread {thread_id} error: {e}", exc_info=True)

    def stop_generators(self) -> None:
        """Stop all generator threads."""
        self.stop_event.set()
        for thread in self.generator_threads:
            thread.join(timeout=5.0)
        logger.info("Generator threads stopped.")
