"""Supervised Fine-Tuning (SFT) engine with LoRA for efficient training.

This module provides:
- LoRA-based SFT training using Hugging Face Trainer
- Automatic model quantization (4-bit/8-bit)
- Early stopping and learning rate scheduling
- Weights & Biases integration for experiment tracking
"""

import logging
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from rlhf_platform.config import ModelConfig, TrainingConfig
from rlhf_platform.dataset import get_dataloader


logger = logging.getLogger(__name__)


def load_model_with_lora(config: ModelConfig) -> torch.nn.Module:
    """Load base model and apply LoRA.

    Supports optional 4-bit or 8-bit quantization for memory efficiency.

    Args:
        config: ModelConfig with model ID and LoRA settings

    Returns:
        Model with LoRA applied (wrapped in peft.PeftModel)

    Raises:
        ValueError: If model not found or config invalid
    """
    quantization_config = None

    # Setup quantization if requested
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading model with 4-bit quantization")
    elif config.load_in_8bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Loading model with 8-bit quantization")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.policy_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules or ["q_proj", "v_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Print trainable params
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model


class SFTTrainer:
    """Supervised Fine-Tuning trainer using Hugging Face Trainer.

    Handles:
    - LoRA model loading and setup
    - Tokenizer setup with special tokens
    - Training loop with HF Trainer
    - Metrics logging and early stopping

    Example:
        >>> config = TrainingConfig.toy_mode()
        >>> trainer = SFTTrainer(config)
        >>> trainer.train("output/sft_model")
        >>> trainer.save_model("output/sft_model")
    """

    def __init__(self, config: TrainingConfig, use_toy: bool = False):
        """Initialize SFT trainer.

        Args:
            config: TrainingConfig with all training settings
            use_toy: If True, use 1K toy dataset for quick testing

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

        # Load model with LoRA
        self.model = load_model_with_lora(config.model)

        # Setup W&B logging
        self.use_wandb = getattr(config, "log_to_wandb", True)

    def train(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
        eval_steps: Optional[int] = None,
    ) -> dict:
        """Run supervised fine-tuning.

        Args:
            output_dir: Directory to save checkpoints and final model
            num_train_epochs: Number of training epochs
            eval_steps: Evaluation frequency (default: every 500 steps)

        Returns:
            Dict with training metrics (loss, eval metrics, etc.)
        """
        # Get dataloader
        train_dataloader = get_dataloader(
            self.config.dataset,
            self.tokenizer,
            use_toy=self.use_toy,
        )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=self.config.dataset.batch_size,
            per_device_eval_batch_size=self.config.dataset.batch_size,
            learning_rate=self.config.optimization.learning_rate,
            weight_decay=self.config.optimization.weight_decay or 0.01,
            warmup_ratio=self.config.optimization.warmup_ratio or 0.1,
            logging_steps=10,
            eval_steps=eval_steps or 500,
            save_steps=1000,
            save_total_limit=3,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.config.optimization.gradient_accumulation_steps or 1,
            max_grad_norm=self.config.optimization.max_grad_norm or 1.0,
            bf16=self.config.optimization.mixed_precision == "bf16",
            fp16=self.config.optimization.mixed_precision == "fp16",
            report_to=["wandb"] if self.use_wandb else [],
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=None,  # Use validation split if available
            tokenizer=self.tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001,
                ),
            ],
        )

        # Run training
        logger.info("Starting SFT training...")
        result = trainer.train()

        # Log final metrics
        logger.info(f"Training loss: {result.training_loss:.4f}")

        return result

    def save_model(self, output_dir: str) -> None:
        """Save fine-tuned model with LoRA weights.

        Args:
            output_dir: Directory to save model
        """
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def load_for_inference(self, model_dir: str) -> torch.nn.Module:
        """Load saved model for inference.

        Args:
            model_dir: Directory with saved model

        Returns:
            Model ready for inference
        """
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.policy_model_id,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        return model


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback for Hugging Face Trainer.

    Stops training if evaluation loss doesn't improve after N steps.
    """

    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.001,
    ):
        """Initialize early stopping.

        Args:
            early_stopping_patience: Steps without improvement to stop
            early_stopping_threshold: Min improvement to count as progress
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_loss = float("inf")
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check for early stopping after evaluation.

        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl (can be modified to stop training)
            metrics: Dict of evaluation metrics
        """
        if metrics is None:
            return

        current_loss = metrics.get("eval_loss")
        if current_loss is None:
            return

        if (
            current_loss < self.best_loss - self.early_stopping_threshold
        ):
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.early_stopping_patience:
            logger.info(
                f"Early stopping triggered after {self.patience_counter} "
                f"steps without improvement"
            )
            control.should_training_stop = True


if __name__ == "__main__":
    """Quick test of SFT trainer."""
    config = TrainingConfig.toy_mode()
    trainer = SFTTrainer(config, use_toy=True)

    # Train for 2 epochs on toy dataset
    result = trainer.train(
        output_dir="output/sft_toy",
        num_train_epochs=1,
        eval_steps=10,
    )
    print(f"Training complete: {result}")

    # Save model
    trainer.save_model("output/sft_toy_checkpoint")
