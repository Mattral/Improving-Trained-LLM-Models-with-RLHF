"""Reward Model Trainer for preference pair ranking.

This module provides:
- Binary cross-entropy training on preference pairs
- LoRA-based efficient training
- Inference wrapper for computing scalar rewards
- Weights & Biases integration
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn import Module, Linear
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from rlhf_platform.config import ModelConfig, TrainingConfig
from rlhf_platform.dataset import PreferencePairDataset, ToyDatasetLoader


logger = logging.getLogger(__name__)


def load_reward_model_with_lora(config: ModelConfig) -> Module:
    """Load reward model (sequence classifier) with LoRA.

    The reward model takes text as input and outputs a scalar reward score
    using a binary classification head.

    Args:
        config: ModelConfig with reward model ID and LoRA settings

    Returns:
        Model with LoRA applied, ready for preference pair training

    Raises:
        ValueError: If model not found
    """
    # Load sequence classification model (num_labels=1 for scalar output)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_id,
        num_labels=1,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules or ["query", "value"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Reward model trainable params: {trainable_params:,} / "
        f"{total_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    return model


class RewardModelTrainer:
    """Trains reward model using binary cross-entropy on preference pairs.

    The reward model learns to score chosen > rejected responses.

    Example:
        >>> config = TrainingConfig.toy_mode()
        >>> trainer = RewardModelTrainer(config)
        >>> trainer.train("output/reward_model")
        >>> reward = trainer.score("Prompt here", "Response here")
    """

    def __init__(self, config: TrainingConfig, use_toy: bool = False):
        """Initialize reward model trainer.

        Args:
            config: TrainingConfig with training settings
            use_toy: If True, use 1K toy dataset

        Raises:
            ValueError: If model loading fails
        """
        self.config = config
        self.use_toy = use_toy

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.reward_model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with LoRA
        self.model = load_reward_model_with_lora(config.model)

        self.use_wandb = getattr(config, "log_to_wandb", True)

    def _tokenize_preference_pair(self, prompt: str, response: str) -> dict:
        """Tokenize prompt and response together.

        Args:
            prompt: Prompt text
            response: Response text

        Returns:
            Dict with input_ids, attention_mask, token_type_ids
        """
        text = prompt + response
        tokens = self.tokenizer(
            text,
            max_length=self.config.dataset.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            k: v.squeeze(0) for k, v in tokens.items()
        }  # Remove batch dim

    def compute_loss(
        self,
        chosen_scores: torch.Tensor,
        rejected_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute binary cross-entropy loss on preference pairs.

        Loss: BCE(sigmoid(score_chosen - score_rejected), 1)
        This encourages score_chosen > score_rejected.

        Args:
            chosen_scores: Reward scores for chosen responses (batch_size,)
            rejected_scores: Reward scores for rejected responses (batch_size,)

        Returns:
            Tuple of (total_loss, accuracy)
        """
        # Margin-based loss: chosen should be > rejected
        score_diff = chosen_scores - rejected_scores
        loss = F.binary_cross_entropy_with_logits(
            score_diff,
            torch.ones_like(score_diff),
        )

        # Compute accuracy (how often chosen > rejected)
        accuracy = (score_diff > 0).float().mean()

        return loss, accuracy

    def train(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
    ) -> dict:
        """Train reward model on preference pairs.

        Args:
            output_dir: Directory to save checkpoints
            num_train_epochs: Number of training epochs

        Returns:
            Dict with training metrics
        """
        # Load dataset
        if self.use_toy:
            pairs = ToyDatasetLoader.load()
        else:
            dataset = PreferencePairDataset(self.config.dataset)
            dataset.load()
            pairs = list(dataset)

        logger.info(f"Training reward model on {len(pairs)} pairs")

        # Create training loop (simplified - normally use Trainer)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimization.learning_rate,
        )

        total_loss = 0.0
        total_accuracy = 0.0
        step = 0

        for epoch in range(num_train_epochs):
            self.model.train()
            for i, pair in enumerate(pairs):
                # Tokenize
                chosen_tokens = self._tokenize_preference_pair(
                    pair.prompt, pair.chosen
                )
                rejected_tokens = self._tokenize_preference_pair(
                    pair.prompt, pair.rejected
                )

                # Forward pass
                chosen_output = self.model(
                    input_ids=chosen_tokens["input_ids"].unsqueeze(0),
                    attention_mask=chosen_tokens.get(
                        "attention_mask", None
                    ),
                )
                rejected_output = self.model(
                    input_ids=rejected_tokens["input_ids"].unsqueeze(0),
                    attention_mask=rejected_tokens.get(
                        "attention_mask", None
                    ),
                )

                chosen_score = chosen_output.logits.squeeze(-1)
                rejected_score = rejected_output.logits.squeeze(-1)

                # Compute loss
                loss, accuracy = self.compute_loss(
                    chosen_score, rejected_score
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimization.max_grad_norm or 1.0,
                )
                optimizer.step()

                # Log metrics
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                step += 1

                if (i + 1) % 100 == 0:
                    avg_loss = total_loss / step
                    avg_acc = total_accuracy / step
                    logger.info(
                        f"Epoch {epoch + 1}, Step {i + 1}: "
                        f"Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}"
                    )

        avg_loss = total_loss / step
        avg_acc = total_accuracy / step
        logger.info(
            f"Training complete: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}"
        )

        return {"loss": avg_loss, "accuracy": avg_acc}

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        """Score a response given a prompt.

        Args:
            prompt: Prompt text
            response: Response text

        Returns:
            Scalar reward score
        """
        self.model.eval()

        tokens = self._tokenize_preference_pair(prompt, response)
        output = self.model(
            input_ids=tokens["input_ids"].unsqueeze(0),
            attention_mask=tokens.get("attention_mask", None),
        )

        reward = output.logits.squeeze(-1).item()
        return reward

    @torch.no_grad()
    def score_batch(
        self, prompts: list[str], responses: list[str]
    ) -> list[float]:
        """Score multiple responses in batch.

        Args:
            prompts: List of prompt texts
            responses: List of response texts

        Returns:
            List of scalar reward scores

        Raises:
            ValueError: If lengths don't match
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have same length")

        self.model.eval()

        rewards = []
        for prompt, response in zip(prompts, responses):
            tokens = self._tokenize_preference_pair(prompt, response)
            output = self.model(
                input_ids=tokens["input_ids"].unsqueeze(0),
                attention_mask=tokens.get("attention_mask", None),
            )
            reward = output.logits.squeeze(-1).item()
            rewards.append(reward)

        return rewards

    def save_model(self, output_dir: str) -> None:
        """Save reward model.

        Args:
            output_dir: Directory to save model
        """
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Reward model saved to {output_dir}")

    def load_for_inference(self, model_dir: str) -> None:
        """Load saved reward model.

        Args:
            model_dir: Directory with saved model
        """
        from peft import PeftModel

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.reward_model_id,
            num_labels=1,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, model_dir)


if __name__ == "__main__":
    """Quick test of reward model trainer."""
    config = TrainingConfig.toy_mode()
    trainer = RewardModelTrainer(config, use_toy=True)

    # Train for 1 epoch on toy dataset
    result = trainer.train(
        output_dir="output/reward_model_toy",
        num_train_epochs=1,
    )
    print(f"Training result: {result}")

    # Score sample responses
    reward_chosen = trainer.score(
        prompt="What is 2+2?", response="The answer is 4."
    )
    reward_rejected = trainer.score(
        prompt="What is 2+2?", response="The answer is 5."
    )
    print(f"Chosen reward: {reward_chosen:.4f}")
    print(f"Rejected reward: {reward_rejected:.4f}")
    print(f"Difference: {reward_chosen - reward_rejected:.4f}")

    # Save model
    trainer.save_model("output/reward_model_toy_checkpoint")
