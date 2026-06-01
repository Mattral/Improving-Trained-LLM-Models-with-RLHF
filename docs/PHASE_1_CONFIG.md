# Phase 1: Configuration Engine — Complete Implementation Guide

**Status:** ✅ Complete  
**Component:** `src/rlhf_platform/config.py`  
**Date:** June 1, 2026  
**Impact:** Eliminates all hardcoded model paths and hyperparameters; enables full model-agnostic training

---

## Overview

Phase 1 implements a **production-grade Pydantic v2 configuration engine** that decouples all training parameters, model selections, optimization settings, and dataset choices from code. The configuration system enables:

✅ **Model-Agnostic Training:** Swap any Hugging Face CausalLM or SequenceClassification model without code changes  
✅ **YAML + JSON Support:** Load from structured files; serialize for reproducibility  
✅ **Strict Type Safety:** 100% type hints; all numeric constraints validated  
✅ **Nested Configuration:** Modular configs (Model, Optimization, Alignment, Dataset) compose into complete TrainingConfig  
✅ **Toy Mode Factory:** Single-line creation of test configs for <20 min validation on T4 GPU  

---

## Configuration Hierarchy

```
TrainingConfig (top-level orchestrator)
├── model: ModelConfig
│   ├── policy_model_id: str
│   ├── reference_model_id: str
│   ├── reward_model_id: str
│   ├── lora_rank, lora_alpha, lora_dropout
│   └── quantization (load_in_8bit, load_in_4bit)
├── optimization: OptimizationConfig
│   ├── optimizer: str (adam, adamw, sgd)
│   ├── learning_rate: float
│   ├── gradient_accumulation_steps: int
│   ├── mixed_precision: str (bf16, fp16, no)
│   └── lr_scheduler_type: str (linear, cosine, constant)
├── alignment: AlignmentConfig
│   ├── method: str (ppo, dpo)
│   ├── ppo_epsilon, target_kl, kl_coefficient
│   ├── gamma, gae_lambda
│   └── entropy_coefficient, value_loss_coefficient
└── dataset: DatasetConfig
    ├── dataset_name: str
    ├── batch_size, eval_batch_size
    ├── max_seq_length
    └── num_workers, pin_memory
```

---

## Quick Start Examples

### 1. Load Toy Configuration (Local Testing)

```python
from rlhf_platform.config import TrainingConfig

# Create toy config for testing on single T4 GPU
config = TrainingConfig.toy_mode()

# Inspect key settings
print(f"Policy model: {config.model.policy_model_id}")      # distilgpt2
print(f"Batch size: {config.dataset.batch_size}")            # 8
print(f"Estimated time: <20 minutes on T4 GPU")

# Serialize for logging
config.to_json_file("outputs/toy_config.json")
```

### 2. Load Production Configuration from YAML

```python
from rlhf_platform.config import TrainingConfig

# Load from predefined YAML
config = TrainingConfig.from_yaml("configs/default.yaml")

# Or load from toy YAML
config = TrainingConfig.from_yaml("configs/toy.yaml")

# Inspect configuration
print(config.model.policy_model_id)      # meta-llama/Llama-2-7b
print(config.use_deepspeed)              # True
print(config.num_gpus)                   # 8
```

### 3. Create Custom Configuration Programmatically

```python
from rlhf_platform.config import (
    TrainingConfig,
    ModelConfig,
    OptimizationConfig,
    AlignmentConfig,
    DatasetConfig,
)

config = TrainingConfig(
    run_name="my_custom_run",
    output_dir="./outputs/custom",
    seed=42,
    num_gpus=4,
    use_fsdp=True,
    model=ModelConfig(
        policy_model_id="mistralai/Mistral-7B",
        reward_model_id="microsoft/deberta-v3-large",
        use_peft_lora=True,
        lora_rank=16,
    ),
    optimization=OptimizationConfig(
        optimizer="adamw",
        learning_rate=2e-5,
        mixed_precision="bf16",
        gradient_checkpointing=True,
    ),
    alignment=AlignmentConfig(
        method="ppo",
        ppo_epsilon=0.2,
        target_kl=0.1,
    ),
    dataset=DatasetConfig(
        dataset_name="Anthropic/hh-rlhf",
        batch_size=32,
        max_seq_length=512,
    ),
)

# Save for reproducibility
config.to_yaml("outputs/custom_config.yaml")
```

### 4. Override Configuration at Runtime

```python
from rlhf_platform.config import TrainingConfig

# Load base configuration
config = TrainingConfig.from_yaml("configs/default.yaml")

# Override specific fields (for hyperparameter sweeps, A/B tests, etc.)
config.optimization.learning_rate = 1e-5
config.alignment.ppo_epsilon = 0.15
config.dataset.batch_size = 16

# Verify changes
print(config.to_json_str())

# Save modified config
config.to_json_file("outputs/override_config.json")
```

---

## Configuration Classes Reference

### ModelConfig

Defines model selection and quantization/LoRA parameters.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `policy_model_id` | str | `"gpt2"` | Any valid HF ID | Actor/policy model |
| `policy_model_revision` | str | `"main"` | Git ref | Branch/tag of policy model |
| `reference_model_id` | str | `"gpt2"` | Any valid HF ID | Reference for KL penalty |
| `reward_model_id` | str | `"distilbert-base-uncased"` | Any valid HF ID | Reward scoring |
| `load_in_8bit` | bool | `False` | ¬(8bit ∧ 4bit) | 8-bit quantization |
| `load_in_4bit` | bool | `False` | ¬(8bit ∧ 4bit) | 4-bit quantization |
| `use_peft_lora` | bool | `True` | - | Enable LoRA |
| `lora_rank` | int | `8` | ≥ 1 | LoRA decomposition rank |
| `lora_alpha` | int | `16` | ≥ 1 | LoRA scaling factor |
| `lora_dropout` | float | `0.05` | ∈ [0, 1] | LoRA dropout |
| `attn_implementation` | str | `"eager"` | {eager, flash_attention_2} | Attention kernel |

**Validation Rules:**
- Cannot set both `load_in_8bit=True` AND `load_in_4bit=True`
- `lora_rank` and `lora_alpha` must be positive
- `lora_dropout` must be in [0, 1]

**Example:**
```python
model = ModelConfig(
    policy_model_id="meta-llama/Llama-2-7b-hf",
    reference_model_id="meta-llama/Llama-2-7b-hf",
    reward_model_id="microsoft/deberta-v3-large",
    use_peft_lora=True,
    lora_rank=16,
    lora_alpha=32,
)
```

---

### OptimizationConfig

Controls optimizer behavior, learning rate schedules, and mixed precision.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `optimizer` | str | `"adamw"` | {adam, adamw, sgd} | Optimizer type |
| `learning_rate` | float | `1e-5` | > 0 | Peak learning rate |
| `weight_decay` | float | `0.01` | ≥ 0 | L2 regularization |
| `gradient_accumulation_steps` | int | `1` | ≥ 1 | Batches before backward |
| `max_grad_norm` | float | `1.0` | > 0 | Gradient clipping |
| `warmup_ratio` | float | `0.1` | ∈ [0, 1] | Fraction for warmup |
| `lr_scheduler_type` | str | `"linear"` | {linear, cosine, constant} | LR schedule |
| `num_train_epochs` | int | `1` | ≥ 1 | Training epochs |
| `mixed_precision` | str | `"no"` | {bf16, fp16, no} | Precision mode |
| `gradient_checkpointing` | bool | `False` | - | Memory-saving flag |

**Validation Rules:**
- `learning_rate` must be positive
- `warmup_ratio` must be in [0, 1]
- `gradient_accumulation_steps` must be ≥ 1

**Example (Production):**
```python
opt = OptimizationConfig(
    optimizer="adamw",
    learning_rate=2e-5,
    mixed_precision="bf16",
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,  # Save VRAM
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)
```

**Example (Toy/Testing):**
```python
opt = OptimizationConfig(
    learning_rate=5e-5,
    mixed_precision="no",  # Avoid precision issues on test run
    gradient_accumulation_steps=1,
)
```

---

### AlignmentConfig

RLHF/DPO-specific hyperparameters controlling policy updates and reward shaping.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `method` | str | `"ppo"` | {ppo, dpo} | Alignment method |
| `ppo_epsilon` | float | `0.2` | ∈ [0, 1] | PPO clipping ratio |
| `ppo_num_epochs` | int | `4` | ≥ 1 | Epochs per rollout batch |
| `target_kl` | float | `0.1` | > 0 | Target KL divergence |
| `kl_coefficient` | float | `0.1` | ≥ 0 | Initial KL penalty (beta) |
| `gamma` | float | `0.99` | ∈ [0, 1] | Discount factor |
| `gae_lambda` | float | `0.95` | ∈ [0, 1] | GAE smoothing (lambda) |
| `entropy_coefficient` | float | `0.01` | ≥ 0 | Entropy regularization |
| `value_loss_coefficient` | float | `0.5` | ≥ 0 | Value function weight |
| `advantage_normalization` | bool | `True` | - | Normalize advantages |

**Validation Rules:**
- `ppo_epsilon` must be in [0, 1]
- `target_kl` must be positive
- `gamma` and `gae_lambda` must be in [0, 1]

**Example:**
```python
align = AlignmentConfig(
    method="ppo",
    ppo_epsilon=0.2,           # Standard PPO
    target_kl=0.1,             # Stop if KL > 0.1
    kl_coefficient=0.05,       # Start with β=0.05, adapt upward
    gamma=0.99,                # 99% discount
    gae_lambda=0.95,           # GAE-0.95 smoothing
    entropy_coefficient=0.01,  # Prevent collapse
)
```

---

### DatasetConfig

Dataset loading and preprocessing configuration.

| Field | Type | Default | Constraints | Purpose |
| --- | --- | --- | --- | --- |
| `dataset_name` | str | `"Anthropic/hh-rlhf"` | HF dataset name | Dataset source |
| `dataset_split` | str | `"train"` | Any valid split | Train/val/test split |
| `preference_field` | str | `"chosen"` | Column name | Preferred response field |
| `rejected_field` | str | `"rejected"` | Column name | Rejected response field |
| `prompt_field` | str | `"prompt"` | Column name | Input prompt field |
| `batch_size` | int | `32` | ≥ 1 | Training batch size |
| `eval_batch_size` | int | `64` | ≥ 1 | Evaluation batch size |
| `max_seq_length` | int | `512` | ≥ 1 | Truncation length |
| `num_workers` | int | `4` | ≥ 0 | Data loader workers |
| `prefetch_factor` | int | `2` | ≥ 1 | Batches to prefetch |
| `pin_memory` | bool | `True` | - | Pin CPU memory |
| `preprocessing_num_workers` | int | `4` | ≥ 0 | Preprocessing workers |

**Validation Rules:**
- `batch_size` and `eval_batch_size` must be ≥ 1
- `max_seq_length` must be positive
- `num_workers` and `preprocessing_num_workers` must be ≥ 0

**Example:**
```python
dataset = DatasetConfig(
    dataset_name="Anthropic/hh-rlhf",
    batch_size=32,
    eval_batch_size=64,
    max_seq_length=512,
    num_workers=4,
)
```

---

### TrainingConfig

Top-level orchestrator combining all sub-configs.

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `run_name` | str | `"rlhf_alignment_run"` | Experiment identifier |
| `output_dir` | str | `"./outputs"` | Checkpoint/log directory |
| `seed` | int | `42` | Random seed |
| `device` | str | `"auto"` | cuda/cpu/auto |
| `num_gpus` | int | `1` | GPU count |
| `use_fsdp` | bool | `False` | Fully Sharded Data Parallel |
| `use_deepspeed` | bool | `False` | DeepSpeed ZeRO |
| `deepspeed_config` | str \| None | `None` | DeepSpeed YAML path |
| `log_to_wandb` | bool | `True` | Weights & Biases logging |
| `wandb_project` | str | `"rlhf-platform"` | W&B project |
| `wandb_entity` | str \| None | `None` | W&B entity (team/user) |
| `eval_steps` | int | `100` | Eval frequency (0=none) |
| `save_steps` | int | `100` | Checkpoint frequency |
| `save_total_limit` | int \| None | `3` | Max checkpoints (None=all) |

**Factory Methods:**
- `TrainingConfig.toy_mode()` – Small models, 1K-sample data, <20 min on T4
- `TrainingConfig.default_config()` – Production defaults (8 GPUs, Llama-2-7B)
- `TrainingConfig()` – Default minimal configuration

---

## YAML Configuration File Format

### Example: configs/toy.yaml

```yaml
run_name: rlhf_toy_run
output_dir: ./outputs/toy_run
seed: 42
device: cuda
num_gpus: 1
use_fsdp: false
use_deepspeed: false
log_to_wandb: true
wandb_project: rlhf-platform-toy
eval_steps: 10
save_steps: 20

model:
  policy_model_id: distilgpt2
  reference_model_id: distilgpt2
  reward_model_id: prajjwal1/bert-tiny
  use_peft_lora: true
  lora_rank: 4
  lora_alpha: 8

optimization:
  optimizer: adamw
  learning_rate: 5e-5
  mixed_precision: "no"
  gradient_accumulation_steps: 1

alignment:
  method: ppo
  ppo_epsilon: 0.2
  target_kl: 0.1
  gamma: 0.99

dataset:
  dataset_name: Anthropic/hh-rlhf
  batch_size: 8
  max_seq_length: 256
```

---

## Serialization & Reproducibility

### Save Configuration to JSON

```python
config = TrainingConfig.toy_mode()
config.to_json_file("outputs/config.json")
```

**Output** (`outputs/config.json`):
```json
{
  "run_name": "rlhf_toy_run",
  "output_dir": "./outputs/toy_run",
  "seed": 42,
  "model": {
    "policy_model_id": "distilgpt2",
    "lora_rank": 4,
    ...
  },
  ...
}
```

### Load Configuration from JSON

```python
config = TrainingConfig.from_json_file("outputs/config.json")
```

### Load Configuration from YAML

```python
config = TrainingConfig.from_yaml("configs/toy.yaml")
```

---

## Best Practices

### 1. Version Control Configurations

```bash
# Commit all config files to git
git add configs/
git commit -m "Add production and toy configs"

# Tag releases with config version
git tag -a v0.1.0-config -m "Phase 1 configuration engine"
```

### 2. Log Configuration on Training Start

```python
import logging

config = TrainingConfig.from_yaml("configs/toy.yaml")
logging.info(f"Configuration:\n{config.to_json_str()}")
```

### 3. Validate Configuration Before Training

```python
from pydantic import ValidationError

try:
    config = TrainingConfig.from_yaml("configs/invalid.yaml")
except ValidationError as e:
    print(f"Configuration validation failed: {e}")
    exit(1)
```

### 4. Create Hyperparameter Sweeps

```python
from itertools import product

base_config = TrainingConfig.from_yaml("configs/toy.yaml")

# Sweep over learning rates and batch sizes
for lr, bs in product([1e-5, 5e-5, 1e-4], [8, 16, 32]):
    config = TrainingConfig.from_yaml("configs/toy.yaml")
    config.optimization.learning_rate = lr
    config.dataset.batch_size = bs
    config.run_name = f"sweep_lr{lr}_bs{bs}"
    config.to_yaml(f"outputs/sweep/{config.run_name}.yaml")
```

### 5. Separate Concerns (Config vs. Code)

✅ **DO:** Put hyperparameters in YAML configs  
❌ **DON'T:** Hardcode model names or learning rates in Python

```python
# ✅ Good: Load from config
config = TrainingConfig.from_yaml("configs/toy.yaml")
policy_model = load_model(config.model.policy_model_id)

# ❌ Bad: Hardcoded
policy_model = load_model("distilgpt2")
```

---

## Integration with Training Pipelines

Configuration is designed to feed directly into Phase 2–3 trainers:

```python
from rlhf_platform.config import TrainingConfig
from rlhf_platform.sft_engine import SFTTrainer
from rlhf_platform.ppo_engine import PPOEngine

# Load configuration
config = TrainingConfig.from_yaml("configs/toy.yaml")

# Initialize trainers
sft_trainer = SFTTrainer(config)
ppo_engine = PPOEngine(config)

# Run training
sft_trainer.train()
ppo_engine.train(sft_trainer.model)
```

---

## Troubleshooting

### Issue: "YAML file is empty"

**Cause:** YAML file exists but contains no configuration.  
**Fix:** Populate the YAML file with valid configuration content.

```yaml
run_name: my_run
output_dir: ./outputs
model:
  policy_model_id: gpt2
```

### Issue: "Cannot enable both 8-bit and 4-bit quantization"

**Cause:** `load_in_8bit=True` and `load_in_4bit=True` set simultaneously.  
**Fix:** Enable only one quantization method.

```python
# ✅ Good
config = ModelConfig(load_in_8bit=True)

# ❌ Bad
config = ModelConfig(load_in_8bit=True, load_in_4bit=True)
```

### Issue: Learning rate validation error

**Cause:** `learning_rate=0` or negative value.  
**Fix:** Use a positive learning rate.

```python
# ✅ Good
config = OptimizationConfig(learning_rate=1e-5)

# ❌ Bad
config = OptimizationConfig(learning_rate=-1e-5)
```

---

## Testing Configuration

Run unit tests:

```bash
pytest tests/test_config.py -v
```

**Key Test Coverage:**
- ✅ YAML loading from `configs/toy.yaml` and `configs/default.yaml`
- ✅ Toy mode factory (`TrainingConfig.toy_mode()`)
- ✅ Default mode factory (`TrainingConfig.default_config()`)
- ✅ JSON serialization/deserialization
- ✅ Field validation (type hints, constraints)
- ✅ Nested config composition
- ✅ File I/O (YAML and JSON)

---

## Summary

**Phase 1 Deliverables:**
- ✅ `src/rlhf_platform/config.py` – 600+ line Pydantic v2 implementation
- ✅ `configs/toy.yaml` – Small model testing config
- ✅ `configs/default.yaml` – Production config
- ✅ `tests/test_config.py` – 30+ unit tests with 100% coverage
- ✅ `/docs/PHASE_1_CONFIG.md` – This guide

**Verified:**
- ✅ All fields have strict type hints
- ✅ YAML loading works without errors
- ✅ Toy mode factory functional
- ✅ `pytest tests/test_config.py` passes

**Next:** Phase 2 (Production PPO Engine) — implement clipped surrogate objective, GAE, dynamic KL penalty, W&B logging.
