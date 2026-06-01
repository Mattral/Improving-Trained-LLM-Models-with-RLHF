"""
Production-grade configuration engine for RLHF/DPO alignment training.

This module provides a strict, type-safe Pydantic v2 configuration layer that decouples
training hyperparameters, model paths, and optimization settings from code. All configurations
inherit from a base class and support both YAML loading and JSON serialization for reproducibility.

Example:
    Load default configuration from YAML:
    >>> config = TrainingConfig.from_yaml("configs/default.yaml")
    >>> config_json = config.model_dump_json(indent=2)

    Use toy configuration for local testing:
    >>> toy_config = TrainingConfig.toy_mode()
    >>> print(toy_config.model_id)  # 'distilgpt2'
"""

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class BaseConfig(BaseModel):
    """Abstract base configuration with common settings.
    
    All configuration subclasses inherit from this to ensure consistent
    model validation, serialization, and reproducibility.
    """

    model_config = {
        "validate_assignment": True,
        "use_enum_values": True,
    }

    def to_json_str(self: "BaseConfig") -> str:
        """Serialize configuration to formatted JSON string.
        
        Returns:
            str: JSON representation of configuration with 2-space indentation.
        """
        return self.model_dump_json(indent=2)

    def to_json_file(self: "BaseConfig", path: str | Path) -> None:
        """Save configuration to JSON file.
        
        Args:
            path: File path where JSON configuration will be saved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json_str())

    @classmethod
    def from_json_str(cls: type["BaseConfig"], json_str: str) -> "BaseConfig":
        """Load configuration from JSON string.
        
        Args:
            json_str: JSON string representation of configuration.
            
        Returns:
            Configuration instance with parsed values.
        """
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_json_file(cls: type["BaseConfig"], path: str | Path) -> "BaseConfig":
        """Load configuration from JSON file.
        
        Args:
            path: Path to JSON configuration file.
            
        Returns:
            Configuration instance with loaded values.
        """
        path = Path(path)
        json_str = path.read_text()
        return cls.from_json_str(json_str)


class ModelConfig(BaseConfig):
    """Model architecture and checkpoint paths.
    
    Supports dynamic model selection from Hugging Face, with optional
    quantization and LoRA parameters.
    
    Attributes:
        policy_model_id: Hugging Face model ID or local path for policy/actor model.
        policy_model_revision: Git revision (branch/tag) of policy model.
        reference_model_id: Hugging Face model ID for reference model (KL constraint).
        reward_model_id: Hugging Face model ID for reward scoring model.
        trust_remote_code: Allow loading of custom code from Hugging Face.
        load_in_8bit: Use 8-bit quantization via bitsandbytes.
        load_in_4bit: Use 4-bit quantization via bitsandbytes.
        use_peft_lora: Enable LoRA fine-tuning via PEFT.
        lora_rank: LoRA rank for low-rank adaptation.
        lora_alpha: LoRA scaling factor (effective learning rate).
        lora_dropout: Dropout probability in LoRA layers.
        attn_implementation: Attention mechanism ('flash_attention_2' or 'eager').
    """

    policy_model_id: str = Field(
        default="gpt2",
        description="Hugging Face model ID or path for policy (actor) model",
    )
    policy_model_revision: str = Field(
        default="main",
        description="Git revision (branch/tag/commit) of policy model",
    )
    reference_model_id: str = Field(
        default="gpt2",
        description="Hugging Face model ID for reference model (frozen, for KL penalty)",
    )
    reward_model_id: str = Field(
        default="distilbert-base-uncased",
        description="Hugging Face model ID for reward scoring model",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow execution of custom code from Hugging Face repositories",
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Use 8-bit quantization via bitsandbytes",
    )
    load_in_4bit: bool = Field(
        default=False,
        description="Use 4-bit quantization via bitsandbytes",
    )
    use_peft_lora: bool = Field(
        default=True,
        description="Enable Parameter-Efficient Fine-Tuning (LoRA)",
    )
    lora_rank: int = Field(
        default=8,
        ge=1,
        description="LoRA rank (r parameter in LoRA decomposition)",
    )
    lora_alpha: int = Field(
        default=16,
        ge=1,
        description="LoRA alpha scaling factor (scales learning rate)",
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Dropout probability in LoRA layers",
    )
    attn_implementation: Literal["flash_attention_2", "eager"] = Field(
        default="eager",
        description="Attention mechanism to use (flash_attention_2 requires compatible GPU)",
    )

    @field_validator("load_in_4bit")
    @classmethod
    def validate_quantization(cls, v: bool, info: Any) -> bool:
        """Ensure only one quantization method is enabled."""
        if v is True and info.data.get("load_in_8bit") is True:
            raise ValueError("Cannot enable both 8-bit and 4-bit quantization")
        return v


class OptimizationConfig(BaseConfig):
    """Optimizer and learning rate settings.
    
    Controls gradient-based optimization, learning rate scheduling,
    and mixed-precision training.
    
    Attributes:
        optimizer: Type of optimizer ('adam', 'adamw', 'sgd').
        learning_rate: Peak learning rate for policy training.
        weight_decay: L2 regularization coefficient.
        gradient_accumulation_steps: Number of batches to accumulate before backward pass.
        max_grad_norm: Gradient clipping threshold (L2 norm).
        warmup_ratio: Fraction of total steps for learning rate warmup.
        lr_scheduler_type: Learning rate schedule ('linear', 'cosine', 'constant').
        num_train_epochs: Number of complete passes through training dataset.
        mixed_precision: Mixed precision training mode ('bf16', 'fp16', or 'no').
        gradient_checkpointing: Enable gradient checkpointing to save memory.
    """

    optimizer: Literal["adam", "adamw", "sgd"] = Field(
        default="adamw",
        description="Optimizer type",
    )
    learning_rate: float = Field(
        default=1e-5,
        gt=0.0,
        description="Peak learning rate for policy updates",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="L2 regularization weight",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Number of batches before backward pass",
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Gradient clipping threshold (L2 norm)",
    )
    warmup_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of total training steps for warmup",
    )
    lr_scheduler_type: Literal["linear", "cosine", "constant"] = Field(
        default="linear",
        description="Learning rate scheduler type",
    )
    num_train_epochs: int = Field(
        default=1,
        ge=1,
        description="Number of training epochs",
    )
    mixed_precision: Literal["bf16", "fp16", "no"] = Field(
        default="no",
        description="Mixed precision mode (requires compatible GPU for bf16)",
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to reduce memory usage",
    )


class AlignmentConfig(BaseConfig):
    """RLHF/DPO alignment hyperparameters.
    
    Controls reward modeling, policy constraints, and advantage computation.
    
    Attributes:
        method: Alignment method ('ppo', 'dpo').
        ppo_epsilon: PPO clipping ratio.
        ppo_num_epochs: Number of epochs to train on each batch of rollouts.
        target_kl: Target KL divergence between policy and reference (soft constraint).
        kl_coefficient: Initial coefficient for KL penalty term.
        gamma: Discount factor for advantage estimation.
        gae_lambda: Smoothing parameter for GAE (lambda in GAE-lambda).
        entropy_coefficient: Weight of entropy regularization in policy loss.
        value_loss_coefficient: Weight of value function loss.
        advantage_normalization: Normalize advantages to zero-mean, unit-variance.
    """

    method: Literal["ppo", "dpo"] = Field(
        default="ppo",
        description="Alignment method (PPO or Direct Preference Optimization)",
    )
    ppo_epsilon: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="PPO clipping ratio (epsilon in clipped surrogate objective)",
    )
    ppo_num_epochs: int = Field(
        default=4,
        ge=1,
        description="Number of epochs to train on each batch of collected rollouts",
    )
    target_kl: float = Field(
        default=0.1,
        gt=0.0,
        description="Target KL divergence (policy vs. reference). Stop early if exceeded.",
    )
    kl_coefficient: float = Field(
        default=0.1,
        ge=0.0,
        description="Initial coefficient for KL penalty term (beta in literature)",
    )
    gamma: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Discount factor for future rewards in advantage estimation",
    )
    gae_lambda: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Smoothing parameter for Generalized Advantage Estimation (lambda)",
    )
    entropy_coefficient: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight of entropy regularization (prevents policy collapse)",
    )
    value_loss_coefficient: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight of value function loss in total objective",
    )
    advantage_normalization: bool = Field(
        default=True,
        description="Normalize advantages to zero-mean, unit-variance",
    )


class DatasetConfig(BaseConfig):
    """Dataset and data loading configuration.
    
    Attributes:
        dataset_name: Hugging Face dataset name or local path.
        dataset_split: Which split to use ('train', 'validation', etc.).
        preference_field: Column name for preferred response.
        rejected_field: Column name for rejected response.
        prompt_field: Column name for input prompt.
        batch_size: Batch size for training.
        eval_batch_size: Batch size for evaluation (can differ from training).
        max_seq_length: Maximum sequence length (longer sequences are truncated).
        num_workers: Number of data loader workers.
        prefetch_factor: Number of batches to prefetch.
        pin_memory: Pin memory for faster GPU transfer.
        preprocessing_num_workers: Number of workers for data preprocessing.
    """

    dataset_name: str = Field(
        default="Anthropic/hh-rlhf",
        description="Hugging Face dataset name or local path",
    )
    dataset_split: str = Field(
        default="train",
        description="Dataset split ('train', 'validation', 'test', etc.)",
    )
    preference_field: str = Field(
        default="chosen",
        description="Column name for preferred/chosen response",
    )
    rejected_field: str = Field(
        default="rejected",
        description="Column name for rejected response",
    )
    prompt_field: str = Field(
        default="prompt",
        description="Column name for input prompt",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for training",
    )
    eval_batch_size: int = Field(
        default=64,
        ge=1,
        description="Batch size for evaluation (typically larger for efficiency)",
    )
    max_seq_length: int = Field(
        default=512,
        ge=1,
        description="Maximum sequence length (pads to this length)",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of data loader workers (0 = main process only)",
    )
    prefetch_factor: int = Field(
        default=2,
        ge=1,
        description="Number of batches to prefetch in data loader",
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin data loader memory for faster CPU-to-GPU transfer",
    )
    preprocessing_num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of workers for dataset preprocessing/tokenization",
    )


class TrainingConfig(BaseConfig):
    """Complete training configuration combining all sub-configs.
    
    This is the top-level configuration object that orchestrates all aspects
    of RLHF/DPO training. It aggregates model, optimization, alignment, and
    dataset configurations into a single, serializable object.
    
    Attributes:
        run_name: Experiment name for logging and checkpointing.
        output_dir: Directory for checkpoints, logs, and outputs.
        seed: Random seed for reproducibility.
        device: Training device ('cuda', 'cpu', 'auto').
        num_gpus: Number of GPUs to use (0 = CPU only).
        use_fsdp: Enable Fully Sharded Data Parallel (distributed training).
        use_deepspeed: Enable DeepSpeed ZeRO optimizer state sharding.
        deepspeed_config: Path to DeepSpeed configuration file (if use_deepspeed=True).
        log_to_wandb: Enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (team/username).
        eval_steps: Evaluate every N training steps (0 = no evaluation).
        save_steps: Save checkpoint every N training steps.
        save_total_limit: Maximum number of recent checkpoints to keep.
        model: Model configuration (paths, quantization, LoRA).
        optimization: Optimizer and learning rate settings.
        alignment: RLHF/DPO hyperparameters.
        dataset: Dataset and data loading configuration.
    """

    run_name: str = Field(
        default="rlhf_alignment_run",
        description="Experiment name for logging and checkpointing",
    )
    output_dir: str = Field(
        default="./outputs",
        description="Directory for saving checkpoints and logs",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility across runs",
    )
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Training device (auto = cuda if available, else cpu)",
    )
    num_gpus: int = Field(
        default=1,
        ge=0,
        description="Number of GPUs to use (0 = CPU-only training)",
    )
    use_fsdp: bool = Field(
        default=False,
        description="Enable Fully Sharded Data Parallel for multi-GPU/multi-node",
    )
    use_deepspeed: bool = Field(
        default=False,
        description="Enable DeepSpeed ZeRO optimizer state sharding",
    )
    deepspeed_config: str | None = Field(
        default=None,
        description="Path to DeepSpeed configuration file (required if use_deepspeed=True)",
    )
    log_to_wandb: bool = Field(
        default=True,
        description="Enable Weights & Biases experiment logging",
    )
    wandb_project: str = Field(
        default="rlhf-platform",
        description="Weights & Biases project name",
    )
    wandb_entity: str | None = Field(
        default=None,
        description="Weights & Biases entity (team or username)",
    )
    eval_steps: int = Field(
        default=100,
        ge=0,
        description="Evaluate model every N steps (0 = no evaluation)",
    )
    save_steps: int = Field(
        default=100,
        ge=1,
        description="Save checkpoint every N steps",
    )
    save_total_limit: int | None = Field(
        default=3,
        ge=1,
        description="Maximum number of recent checkpoints to keep (None = keep all)",
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model architecture and paths configuration",
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimizer and learning rate configuration",
    )
    alignment: AlignmentConfig = Field(
        default_factory=AlignmentConfig,
        description="RLHF/DPO hyperparameters configuration",
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig,
        description="Dataset and data loading configuration",
    )

    @classmethod
    def from_yaml(cls: type["TrainingConfig"], path: str | Path) -> "TrainingConfig":
        """Load configuration from YAML file.
        
        YAML structure should mirror the nested configuration classes.
        
        Args:
            path: Path to YAML configuration file.
            
        Returns:
            TrainingConfig instance with loaded values.
            
        Example:
            >>> config = TrainingConfig.from_yaml("configs/default.yaml")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"YAML file is empty: {path}")

        return cls(**data)

    def to_yaml(self: "TrainingConfig", path: str | Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: File path where YAML configuration will be saved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.model_dump()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def toy_mode(cls: type["TrainingConfig"]) -> "TrainingConfig":
        """Create a toy configuration for local testing on single GPU.
        
        Toy configuration uses small models and datasets to verify pipeline
        correctness in <20 minutes on a free-tier T4 GPU.
        
        Toy settings:
        - Policy: distilgpt2 (82M params)
        - Reference: distilgpt2
        - Reward: prajjwal1/bert-tiny (4.4M params)
        - Dataset: Anthropic/hh-rlhf (1,000 sample slice, local cache)
        - Batch size: 8
        - Grad accumulation: 1
        - Training steps: ~100 per phase
        
        Returns:
            TrainingConfig with all toy overrides applied.
        """
        return cls(
            run_name="rlhf_toy_run",
            output_dir="./outputs/toy_run",
            seed=42,
            device="cuda",
            num_gpus=1,
            use_fsdp=False,
            use_deepspeed=False,
            log_to_wandb=True,
            wandb_project="rlhf-platform-toy",
            eval_steps=10,
            save_steps=20,
            save_total_limit=2,
            model=ModelConfig(
                policy_model_id="distilgpt2",
                policy_model_revision="main",
                reference_model_id="distilgpt2",
                reward_model_id="prajjwal1/bert-tiny",
                trust_remote_code=False,
                load_in_8bit=False,
                load_in_4bit=False,
                use_peft_lora=True,
                lora_rank=4,
                lora_alpha=8,
                lora_dropout=0.05,
                attn_implementation="eager",
            ),
            optimization=OptimizationConfig(
                optimizer="adamw",
                learning_rate=5e-5,
                weight_decay=0.01,
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                warmup_ratio=0.1,
                lr_scheduler_type="linear",
                num_train_epochs=1,
                mixed_precision="no",
                gradient_checkpointing=False,
            ),
            alignment=AlignmentConfig(
                method="ppo",
                ppo_epsilon=0.2,
                ppo_num_epochs=2,
                target_kl=0.1,
                kl_coefficient=0.05,
                gamma=0.99,
                gae_lambda=0.95,
                entropy_coefficient=0.01,
                value_loss_coefficient=0.5,
                advantage_normalization=True,
            ),
            dataset=DatasetConfig(
                dataset_name="Anthropic/hh-rlhf",
                dataset_split="train",
                preference_field="chosen",
                rejected_field="rejected",
                prompt_field="prompt",
                batch_size=8,
                eval_batch_size=16,
                max_seq_length=256,
                num_workers=0,
                prefetch_factor=2,
                pin_memory=False,
                preprocessing_num_workers=0,
            ),
        )

    @classmethod
    def default_config(cls: type["TrainingConfig"]) -> "TrainingConfig":
        """Create a sensible default configuration for production training.
        
        Default configuration uses standard models and hyperparameters suitable
        for training on modern GPUs with decent memory (e.g., A100, H100).
        
        Returns:
            TrainingConfig with production-recommended defaults.
        """
        return cls(
            run_name="rlhf_production_run",
            output_dir="./outputs/production_run",
            seed=42,
            device="cuda",
            num_gpus=8,
            use_fsdp=True,
            use_deepspeed=True,
            deepspeed_config="configs/deepspeed_zero3.yaml",
            log_to_wandb=True,
            wandb_project="rlhf-platform",
            eval_steps=500,
            save_steps=500,
            save_total_limit=5,
            model=ModelConfig(
                policy_model_id="meta-llama/Llama-2-7b",
                policy_model_revision="main",
                reference_model_id="meta-llama/Llama-2-7b",
                reward_model_id="microsoft/deberta-v3-large",
                trust_remote_code=False,
                load_in_8bit=False,
                load_in_4bit=False,
                use_peft_lora=True,
                lora_rank=16,
                lora_alpha=32,
                lora_dropout=0.05,
                attn_implementation="flash_attention_2",
            ),
            optimization=OptimizationConfig(
                optimizer="adamw",
                learning_rate=2e-5,
                weight_decay=0.01,
                gradient_accumulation_steps=4,
                max_grad_norm=1.0,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                num_train_epochs=3,
                mixed_precision="bf16",
                gradient_checkpointing=True,
            ),
            alignment=AlignmentConfig(
                method="ppo",
                ppo_epsilon=0.2,
                ppo_num_epochs=4,
                target_kl=0.1,
                kl_coefficient=0.1,
                gamma=0.99,
                gae_lambda=0.95,
                entropy_coefficient=0.01,
                value_loss_coefficient=0.5,
                advantage_normalization=True,
            ),
            dataset=DatasetConfig(
                dataset_name="Anthropic/hh-rlhf",
                dataset_split="train",
                preference_field="chosen",
                rejected_field="rejected",
                prompt_field="prompt",
                batch_size=32,
                eval_batch_size=64,
                max_seq_length=512,
                num_workers=4,
                prefetch_factor=2,
                pin_memory=True,
                preprocessing_num_workers=4,
            ),
        )
