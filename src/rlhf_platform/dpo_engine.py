"""Direct Preference Optimization (DPO) for reference-free alignment.

This module provides:
- DPO trainer for direct preference learning on pair data
- DPO loss computation without explicit reward model
- Batch processing and W&B logging
- Comparable training loop to PPO
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlhf_platform.config import TrainingConfig
from rlhf_platform.dataset import PreferencePairDataset, ToyDatasetLoader


logger = logging.getLogger(__name__)


class DPOMetrics:
    """Metrics container for DPO training step.

    Attributes:
        dpo_loss: Core DPO objective loss
        policy_diff_mean: Mean difference in policy log probs
        margin_mean: Mean preference margin (chosen - rejected)
        accuracy: Fraction where chosen > rejected
        explained_variance: How well value network explains returns
    """

    def __init__(
        self,
        dpo_loss: float,
        policy_diff_mean: float,
        margin_mean: float,
        accuracy: float,
        explained_variance: float,
    ):
        self.dpo_loss = dpo_loss
        self.policy_diff_mean = policy_diff_mean
        self.margin_mean = margin_mean
        self.accuracy = accuracy
        self.explained_variance = explained_variance

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "dpo_loss": self.dpo_loss,
            "policy_diff_mean": self.policy_diff_mean,
            "margin_mean": self.margin_mean,
            "accuracy": self.accuracy,
            "explained_variance": self.explained_variance,
        }


class DPOTrainer:
    """Direct Preference Optimization trainer.

    DPO directly optimizes the policy on preference pairs without learning
    a separate reward model. Uses the log-likelihood ratio to compute
    preferences directly from policy predictions.

    Mathematical objective:

        L_DPO = -E[(x,y_c,y_r) in D] [log σ(β(log π(y_c|x)/π_ref(y_c|x) - 
                                                   log π(y_r|x)/π_ref(y_r|x)))]

    Where:
    - π(y|x) = trained policy
    - π_ref(y|x) = reference policy  
    - β = temperature parameter (default: 0.1)
    - σ = sigmoid function

    Example:
        >>> config = TrainingConfig.toy_mode()
        >>> trainer = DPOTrainer(config)
        >>> metrics = trainer.train_step(batch)
        >>> print(f"DPO Loss: {metrics.dpo_loss:.4f}")
    """

    def __init__(self, config: TrainingConfig, use_toy: bool = False):
        """Initialize DPO trainer.

        Args:
            config: TrainingConfig with all training settings
            use_toy: If True, use 1K toy dataset for testing

        Raises:
            ValueError: If model loading fails
        """
        self.config = config
        self.use_toy = use_toy

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.policy_model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load policy model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.model.policy_model_id,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load reference model (frozen, no gradients)
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            config.model.reference_model_id,
            device_map="auto",
            trust_remote_code=True,
        )
        self.reference_model.requires_grad_(False)

        # Setup optimizer (only for policy model)
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=config.optimization.learning_rate,
        )

        # DPO hyperparameters
        self.beta = getattr(config.alignment, "dpo_beta", 0.1)  # Temperature
        self.step_count = 0

        logger.info(
            f"DPO Trainer initialized: policy={config.model.policy_model_id}, "
            f"beta={self.beta}"
        )

    def _tokenize_for_loss(
        self, prompt: str, response: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompt and response for loss computation.

        Args:
            prompt: Prompt text
            response: Response text

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        text = prompt + response
        tokens = self.tokenizer(
            text,
            max_length=self.config.dataset.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return tokens["input_ids"], tokens["attention_mask"]

    def compute_dpo_loss(
        self,
        policy_logits_chosen: torch.Tensor,
        policy_logits_rejected: torch.Tensor,
        ref_logits_chosen: torch.Tensor,
        ref_logits_rejected: torch.Tensor,
    ) -> Tuple[torch.Tensor, DPOMetrics]:
        """Compute DPO loss on preference pairs.

        The DPO objective directly models preferences through the log-likelihood
        ratio, avoiding the need for an explicit reward model.

        Args:
            policy_logits_chosen: Policy log-probs for chosen responses (batch,)
            policy_logits_rejected: Policy log-probs for rejected responses (batch,)
            ref_logits_chosen: Reference log-probs for chosen (batch,)
            ref_logits_rejected: Reference log-probs for rejected (batch,)

        Returns:
            Tuple of (loss, metrics)
        """
        # Compute log-likelihood ratios
        policy_diff = policy_logits_chosen - policy_logits_rejected
        ref_diff = ref_logits_chosen - ref_logits_rejected

        # DPO loss: log σ(β * (policy_diff - ref_diff))
        logits = self.beta * (policy_diff - ref_diff)
        dpo_loss = -F.logsigmoid(logits).mean()

        # Compute metrics
        margin = policy_diff.detach() - ref_diff.detach()
        accuracy = (margin > 0).float().mean().item()
        margin_mean = margin.mean().item()

        # Explained variance (dummy for now)
        explained_variance = 0.0

        metrics = DPOMetrics(
            dpo_loss=dpo_loss.item(),
            policy_diff_mean=policy_diff.detach().mean().item(),
            margin_mean=margin_mean,
            accuracy=accuracy,
            explained_variance=explained_variance,
        )

        return dpo_loss, metrics

    @torch.no_grad()
    def compute_log_probs(
        self,
        model: Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for a sequence.

        Args:
            model: Language model
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Log probabilities (batch,) - mean log prob per sequence
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Compute log probs for non-padding tokens
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Get log prob of actual next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_log_probs = log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            shift_log_probs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding tokens
        shift_mask = attention_mask[..., 1:].contiguous()
        token_log_probs = token_log_probs * shift_mask

        # Average over non-padding tokens
        sequence_log_probs = (
            token_log_probs.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        )

        return sequence_log_probs

    def train_step(self, batch: dict) -> DPOMetrics:
        """Single DPO training step.

        Args:
            batch: Dictionary with keys:
                - prompt: Prompt text
                - chosen: Chosen response
                - rejected: Rejected response

        Returns:
            DPOMetrics with loss and other metrics
        """
        self.policy_model.train()
        self.reference_model.eval()

        # Tokenize preference pair
        prompt = batch["prompt"]
        chosen_input_ids, chosen_mask = self._tokenize_for_loss(
            prompt, batch["chosen"]
        )
        rejected_input_ids, rejected_mask = self._tokenize_for_loss(
            prompt, batch["rejected"]
        )

        # Compute log probs under policy
        policy_chosen_log_probs = self.compute_log_probs(
            self.policy_model, chosen_input_ids, chosen_mask
        )
        policy_rejected_log_probs = self.compute_log_probs(
            self.policy_model, rejected_input_ids, rejected_mask
        )

        # Compute log probs under reference (no grad)
        with torch.no_grad():
            ref_chosen_log_probs = self.compute_log_probs(
                self.reference_model, chosen_input_ids, chosen_mask
            )
            ref_rejected_log_probs = self.compute_log_probs(
                self.reference_model, rejected_input_ids, rejected_mask
            )

        # Compute DPO loss
        loss, metrics = self.compute_dpo_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            ref_chosen_log_probs,
            ref_rejected_log_probs,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.optimization.max_grad_norm or 1.0,
        )
        self.optimizer.step()

        self.step_count += 1

        # Log to W&B if enabled
        if getattr(self.config, "log_to_wandb", True):
            try:
                import wandb

                wandb.log(
                    {
                        "dpo/loss": metrics.dpo_loss,
                        "dpo/policy_diff_mean": metrics.policy_diff_mean,
                        "dpo/margin_mean": metrics.margin_mean,
                        "dpo/accuracy": metrics.accuracy,
                        "dpo/global_step": self.step_count,
                    }
                )
            except ImportError:
                pass

        return metrics

    def save_model(self, output_dir: str) -> None:
        """Save trained policy model.

        Args:
            output_dir: Directory to save model
        """
        self.policy_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")


def log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute log softmax with numerical stability.

    Args:
        logits: Unnormalized scores
        dim: Dimension to normalize over

    Returns:
        Log softmax values
    """
    return F.log_softmax(logits, dim=dim)


if __name__ == "__main__":
    """Quick test of DPO trainer."""
    config = TrainingConfig.toy_mode()
    config.alignment.method = "dpo"

    trainer = DPOTrainer(config, use_toy=True)

    # Load toy dataset
    pairs = ToyDatasetLoader.load()

    # Train for a few steps
    for i, pair in enumerate(pairs[:5]):
        batch = {
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        }
        metrics = trainer.train_step(batch)
        print(
            f"Step {i+1}: DPO Loss={metrics.dpo_loss:.4f}, "
            f"Accuracy={metrics.accuracy:.4f}"
        )

    # Save model
    trainer.save_model("output/dpo_toy_checkpoint")
