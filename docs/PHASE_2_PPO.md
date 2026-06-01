# Phase 2: Production-Grade PPO Engine — Complete Implementation Guide

**Status:** ✅ Complete  
**Component:** `src/rlhf_platform/ppo_engine.py`  
**Date:** June 1, 2026  
**Lines of Code:** 700+  
**Impact:** Provides production-grade PPO algorithm with adaptive KL control, entropy regularization, and comprehensive W&B logging

---

## Overview

Phase 2 implements a **production-grade Proximal Policy Optimization (PPO) trainer** with the following core features:

✅ **Generalized Advantage Estimation (GAE)** – Flexible advantage computation with configurable lambda  
✅ **Clipped Surrogate Objective** – Robust policy gradient with epsilon clipping  
✅ **Dynamic KL Penalty** – Adaptive beta adjustment based on target KL divergence  
✅ **Entropy Regularization** – Prevents policy collapse through exploration bonus  
✅ **Value Function Training** – Critic network with gradient clipping  
✅ **Mixed Precision Support** – BF16/FP16 compatible  
✅ **Weights & Biases Integration** – Real-time metrics logging  

---

## Mathematical Foundations

### Clipped Surrogate Objective

The core PPO loss prevents excessively large policy updates by clipping probability ratios:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ – probability ratio
- $\hat{A}_t$ – estimated advantage (from GAE)
- $\epsilon$ – clip ratio (typically 0.2)

**Intuition:** The minimum operation selects whichever term is smaller, discouraging policy changes that increase the objective beyond the clipped range.

### Generalized Advantage Estimation

GAE smooths the bias-variance tradeoff via:

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$

Where:
- $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ – TD residual
- $\gamma$ – discount factor (typically 0.99)
- $\lambda$ – smoothing parameter (typically 0.95)

**Behavior:**
- $\lambda = 0$ → low variance (TD-only)
- $\lambda = 1$ → high variance (Monte Carlo)
- $\lambda = 0.95$ → balanced tradeoff

### KL Divergence Penalty

For categorical actions:

$$D_{\text{KL}}(P || Q) = \sum_a P(a) \left( \log P(a) - \log Q(a) \right)$$

**Adaptive Control:**
- If $D_{\text{KL}} > 1.5 \times \text{target_kl}$: increase $\beta$ (more penalty)
- If $D_{\text{KL}} < \frac{\text{target_kl}}{1.5}$: decrease $\beta$ (less penalty)
- Otherwise: keep $\beta$ constant

This ensures KL divergence stays near the target without manual tuning.

---

## Core Components

### 1. GeneralizedAdvantageEstimation Class

Computes advantages and returns using GAE algorithm.

```python
from rlhf_platform.ppo_engine import GeneralizedAdvantageEstimation

gae = GeneralizedAdvantageEstimation(gamma=0.99, lam=0.95)

# Compute advantages from rollout trajectory
advantages, returns = gae.compute(
    rewards=torch.randn(batch_size, seq_len),
    values=torch.randn(batch_size, seq_len),
    next_values=torch.randn(batch_size, seq_len),
    dones=torch.zeros(batch_size, seq_len),
)
```

**Key Methods:**
- `__init__(gamma, lam)` – Initialize with discount and smoothing factors
- `compute(rewards, values, next_values, dones)` → (advantages, returns)

**Constraints:**
- `gamma` ∈ [0, 1] – discount factor
- `lambda` ∈ [0, 1] – smoothing parameter

### 2. PPOMetrics Dataclass

Container for all training metrics from a single step.

```python
from rlhf_platform.ppo_engine import PPOMetrics

metrics = PPOMetrics(
    policy_loss=0.5,
    value_loss=0.3,
    entropy=1.2,
    kl_divergence=0.08,
    explained_variance=0.7,
    adaptive_kl_coefficient=0.1,
    reward_mean=0.5,
    advantage_mean=0.0,
    gae_mean=0.0,
)

print(f"Policy loss: {metrics.policy_loss}")
print(f"KL divergence: {metrics.kl_divergence}")
```

**Fields:**
- `policy_loss` – Clipped surrogate objective value
- `value_loss` – Critic MSE loss
- `entropy` – Policy entropy (exploration metric)
- `kl_divergence` – KL(new || old)
- `explained_variance` – Fraction of return variance explained by value function
- `adaptive_kl_coefficient` – Current KL penalty coefficient (beta)
- `reward_mean` – Mean reward in batch
- `advantage_mean` – Mean advantage
- `gae_mean` – Mean GAE advantage

### 3. PPOTrainer Class

Main trainer orchestrating policy and value updates.

#### Initialization

```python
from rlhf_platform.config import TrainingConfig
from rlhf_platform.ppo_engine import PPOTrainer

config = TrainingConfig.toy_mode()

# Assume models loaded
policy_model = load_policy(config.model.policy_model_id)
value_model = load_value_model(config.model.policy_model_id)
reference_model = load_reference(config.model.reference_model_id)

optimizer = torch.optim.AdamW(
    list(policy_model.parameters()) + list(value_model.parameters()),
    lr=config.optimization.learning_rate,
)

trainer = PPOTrainer(
    config=config,
    policy_model=policy_model,
    value_model=value_model,
    reference_model=reference_model,
    optimizer=optimizer,
)
```

#### Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        metrics = trainer.train_step(
            batch={
                "input_ids": input_ids,
                "rewards": rewards,
                "values": values,
                "next_values": next_values,
                "dones": dones,
            },
            old_log_probs=old_log_probs,
        )
        
        # Use metrics for logging, early stopping, etc.
        if metrics.kl_divergence > trainer.target_kl * 2.0:
            logger.warning(f"KL too high: {metrics.kl_divergence:.4f}")
        
        if metrics.explained_variance > 0.8:
            logger.info("Value function converged")
```

#### Key Methods

**Loss Computations:**
- `compute_policy_loss(log_probs_new, log_probs_old, advantages)` → (loss, ratio)
- `compute_value_loss(value_pred, returns)` → scalar loss
- `compute_entropy_bonus(logits)` → entropy scalar
- `compute_kl_divergence(logits_new, logits_old)` → KL divergence

**Metrics:**
- `compute_explained_variance(value_pred, returns)` → float in (-∞, 1]
- `adaptive_kl_penalty(kl_div)` → None (updates self.kl_coefficient)

**Training:**
- `train_step(batch, old_log_probs)` → PPOMetrics
- `log_config_to_wandb()` → None

---

## Configurable Hyperparameters

All hyperparameters come from `TrainingConfig`, ensuring reproducibility:

| Parameter | Config Class | Type | Default (Toy) | Range/Notes |
| --- | --- | --- | --- | --- |
| `ppo_epsilon` | AlignmentConfig | float | 0.2 | ∈ [0, 1], clip ratio |
| `target_kl` | AlignmentConfig | float | 0.1 | > 0, KL constraint |
| `kl_coefficient` | AlignmentConfig | float | 0.05 | ≥ 0, beta (adaptive) |
| `gamma` | AlignmentConfig | float | 0.99 | ∈ [0, 1], discount |
| `gae_lambda` | AlignmentConfig | float | 0.95 | ∈ [0, 1], smoothing |
| `entropy_coefficient` | AlignmentConfig | float | 0.01 | ≥ 0, exploration bonus |
| `value_loss_coefficient` | AlignmentConfig | float | 0.5 | ≥ 0, critic weight |
| `max_grad_norm` | OptimizationConfig | float | 1.0 | > 0, gradient clipping |
| `learning_rate` | OptimizationConfig | float | 5e-5 (toy) | > 0 |

---

## Training Metrics

PPO trainer logs comprehensive metrics for monitoring:

### Per-Step Metrics (logged to W&B)

```
ppo/policy_loss        – Clipped surrogate objective
ppo/value_loss         – Critic MSE loss
ppo/entropy            – Policy entropy (should stay ~positive)
ppo/kl_divergence      – KL(new || old) (should track target)
ppo/explained_variance – How well value function explains returns
ppo/adaptive_kl_coefficient – Current KL penalty coefficient (adapts)
ppo/reward_mean        – Mean reward in batch
ppo/advantage_mean     – Mean advantage estimate
ppo/global_step        – Training step counter
```

### Example W&B Dashboard Interpretation

**Healthy Training:**
- `entropy` stays > 0 (exploring)
- `kl_divergence` fluctuates near `target_kl` (0.1)
- `policy_loss` decreases over time
- `explained_variance` increases (critic improving)
- `adaptive_kl_coefficient` stable (not oscillating)

**Warning Signs:**
- `entropy` → 0 (policy collapse, increase `entropy_coefficient`)
- `kl_divergence` >> `target_kl` (increase `learning_rate` or `kl_coefficient`)
- `policy_loss` diverges (decrease `learning_rate`)
- `explained_variance` < -1 (value function failing)

---

## Numerical Stability Features

Phase 2 includes several safeguards for stable training:

### 1. Advantage Normalization
```python
if config.alignment.advantage_normalization:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```
Prevents extreme values from destabilizing gradients.

### 2. Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(
    policy_model.parameters(),
    config.optimization.max_grad_norm
)
```
Prevents exploding gradients common in RL.

### 3. KL Penalty Bounds
```python
# Coefficient stays in [0.001, 10.0]
self.kl_coefficient = max(0.001, min(10.0, self.kl_coefficient))
```
Prevents penalty from becoming too extreme.

### 4. Entropy Buffer
```python
# Add small epsilon to prevent log(0)
entropy = -(probs * (log_probs + 1e-8)).sum(dim=-1).mean()
```

---

## Integration with Phase 1 Config

Configuration flows automatically from Phase 1 YAML/JSON:

```yaml
# configs/toy.yaml
alignment:
  method: ppo
  ppo_epsilon: 0.2
  target_kl: 0.1
  kl_coefficient: 0.05
  gamma: 0.99
  gae_lambda: 0.95
  entropy_coefficient: 0.01
  value_loss_coefficient: 0.5

optimization:
  learning_rate: 5e-5
  max_grad_norm: 1.0
```

Load and use directly:

```python
config = TrainingConfig.from_yaml("configs/toy.yaml")
trainer = PPOTrainer(config, policy, value, reference, optimizer)
```

---

## Mixed-Precision Training

PPO trainer is compatible with AMP (Automatic Mixed Precision):

```python
# Enable mixed precision in config
config.optimization.mixed_precision = "bf16"

# Wrap training loop with autocast
from torch.cuda.amp import autocast

with autocast(dtype=torch.bfloat16):
    metrics = trainer.train_step(batch, old_log_probs)
```

Benefits:
- **Speed:** 2-3x faster on modern GPUs (H100, A100)
- **Memory:** 50-60% reduction in peak VRAM
- **Stability:** BF16 more stable than FP16 for RL

---

## Testing Coverage

Phase 2 includes 30+ unit tests covering:

### GAE Tests
- [x] Valid initialization and bounds checking
- [x] Correct output shapes
- [x] Edge cases (gamma=0, lambda=0, identical distributions)
- [x] TD consistency checks

### Loss Computation Tests
- [x] Clipped policy loss behavior
- [x] Entropy bounds and properties
- [x] KL divergence non-negativity
- [x] Value function loss convergence

### Metric Tests
- [x] Explained variance computation
- [x] Adaptive KL coefficient bounds
- [x] PPOMetrics dataclass instantiation

### Integration Tests
- [x] Full training step with weight updates
- [x] Metrics validity after step
- [x] W&B logging (when wandb available)

Run tests:
```bash
pytest tests/test_ppo_engine.py -v
```

---

## Weights & Biases Integration

Automatic logging to W&B (if `config.log_to_wandb=True`):

```python
trainer = PPOTrainer(config, policy, value, reference, optimizer)

# Optional: log config once
trainer.log_config_to_wandb()

# Each train_step automatically logs metrics
for batch in dataloader:
    metrics = trainer.train_step(batch, old_log_probs)
    # Metrics automatically logged to wandb
```

Example W&B chart configuration:

```python
# In trainer.__init__:
wandb.define_metric("ppo/global_step")
wandb.define_metric("ppo/*", step_metric="ppo/global_step")
```

This enables:
- Automatic chart generation
- Custom metrics dashboards
- Experiment comparison
- Hyperparameter tracking

---

## Performance Characteristics

Expected behavior on toy config (distilgpt2, T4 GPU):

| Metric | Expected | Note |
| --- | --- | --- |
| **Step Time** | 0.5-1.0 sec | Per training step |
| **VRAM** | 2-3 GB | With LoRA, no gradient checkpointing |
| **Throughput** | ~8-16 tokens/sec | Batch size 8, seq_len 256 |
| **KL Convergence** | 50-100 steps | To target 0.1 |
| **Policy Loss** | 0.5 → 0.1 | Over 500 steps |
| **Value Loss** | 1.0 → 0.1 | Over 500 steps |

With production config (Llama-2-7B, 8x A100):

| Metric | Expected |
| --- | --- |
| **Step Time** | 2-5 sec (per batch) |
| **VRAM per GPU** | 40-60 GB (with ZeRO-3) |
| **Throughput** | 100-200 tokens/sec |
| **Global Batch Size** | 256 tokens/step |

---

## Common Issues & Troubleshooting

### Issue: Policy Loss NaN After Few Steps
**Cause:** Exploding gradients, often from high learning rate or extreme rewards.  
**Fix:** 
- Decrease `learning_rate` (e.g., 1e-5 → 5e-6)
- Enable gradient clipping (increase `max_grad_norm`)
- Normalize rewards: `rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)`

### Issue: KL Divergence Not Decreasing
**Cause:** Target KL too low, or reference model initialization mismatch.  
**Fix:**
- Increase `target_kl` (e.g., 0.1 → 0.2)
- Ensure reference model matches policy initialization
- Check `kl_coefficient` is not stuck at bounds

### Issue: Value Function Never Improves
**Cause:** Value network underfitting, or learning rate too low.  
**Fix:**
- Increase value learning rate (separate optimizer)
- Add more hidden layers to value network
- Increase `value_loss_coefficient` (e.g., 0.5 → 1.0)
- Reduce `entropy_coefficient` (redirects gradient to value)

### Issue: Entropy Drops to Zero
**Cause:** Policy becoming deterministic, losing exploration.  
**Fix:**
- Increase `entropy_coefficient` (e.g., 0.01 → 0.05)
- Decrease `target_kl` (relax policy constraint)
- Initialize reference model with higher temperature

---

## Next Steps (Phase 3)

Phase 2 provides the core PPO algorithm. Phase 3 will:
- Integrate with dataset pipeline
- Create CLI entry points (`train-ppo`)
- Implement toy mode end-to-end pipeline
- Add checkpointing and resumption

---

## Summary

**Phase 2 Deliverables:**
- ✅ `src/rlhf_platform/ppo_engine.py` – 700+ lines, production-grade
- ✅ `GeneralizedAdvantageEstimation` class with full validation
- ✅ `PPOTrainer` with clipped objective, KL control, entropy regularization
- ✅ `PPOMetrics` dataclass for metric tracking
- ✅ `tests/test_ppo_engine.py` – 30+ unit and integration tests
- ✅ `/docs/PHASE_2_PPO.md` – Complete guide with examples

**Verified Features:**
- ✅ Clipped surrogate objective with epsilon clipping
- ✅ Generalized Advantage Estimation (GAE) with lambda smoothing
- ✅ Dynamic KL penalty with adaptive beta adjustment
- ✅ Entropy regularization preventing policy collapse
- ✅ Value function training with gradient clipping
- ✅ Weights & Biases integration for real-time logging
- ✅ Numerical stability safeguards (advantage normalization, grad clipping)
- ✅ Mixed-precision compatible (BF16/FP16)

**Next:** Phase 3 (CLI + Toy Pipeline) — June 6–10, 2026
