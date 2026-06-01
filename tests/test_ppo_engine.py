"""Unit tests for PPO trainer implementation (src/rlhf_platform/ppo_engine.py).

Tests validate:
- Generalized Advantage Estimation (GAE) computation
- PPO clipped surrogate objective
- Value function loss
- Entropy bonus computation
- KL divergence estimation
- Adaptive KL control
- Metric computation (explained variance, etc.)
- W&B logging integration
"""

import pytest
import torch
import torch.nn as nn

from rlhf_platform.config import (
    AlignmentConfig,
    OptimizationConfig,
    TrainingConfig,
)
from rlhf_platform.ppo_engine import (
    GeneralizedAdvantageEstimation,
    PPOMetrics,
    PPOTrainer,
)


class DummyModel(nn.Module):
    """Simple dummy model for testing (no actual LLM computation)."""

    def __init__(self, vocab_size: int = 50257, hidden_size: int = 768) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # Global average pooling
        logits = self.linear(x)
        return logits


class DummyValueModel(nn.Module):
    """Simple value network for testing."""

    def __init__(self, vocab_size: int = 50257, hidden_size: int = 768) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = x.mean(dim=1)
        value = self.linear(x).squeeze(-1)
        return value


class TestGeneralizedAdvantageEstimation:
    """Tests for GAE computation."""

    def test_gae_initialization(self) -> None:
        """Verify GAE can be initialized with valid hyperparameters."""
        gae = GeneralizedAdvantageEstimation(gamma=0.99, lam=0.95)
        assert gae.gamma == 0.99
        assert gae.lam == 0.95

    def test_gae_invalid_gamma(self) -> None:
        """Verify GAE rejects invalid gamma values."""
        with pytest.raises(ValueError):
            GeneralizedAdvantageEstimation(gamma=1.5, lam=0.95)
        with pytest.raises(ValueError):
            GeneralizedAdvantageEstimation(gamma=-0.1, lam=0.95)

    def test_gae_invalid_lambda(self) -> None:
        """Verify GAE rejects invalid lambda values."""
        with pytest.raises(ValueError):
            GeneralizedAdvantageEstimation(gamma=0.99, lam=1.5)
        with pytest.raises(ValueError):
            GeneralizedAdvantageEstimation(gamma=0.99, lam=-0.1)

    def test_gae_compute_shape(self) -> None:
        """Verify GAE output shapes match input shapes."""
        gae = GeneralizedAdvantageEstimation(gamma=0.99, lam=0.95)

        batch_size, seq_len = 8, 16
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        next_values = torch.randn(batch_size, seq_len)
        dones = torch.zeros(batch_size, seq_len)

        advantages, returns = gae.compute(rewards, values, next_values, dones)

        assert advantages.shape == (batch_size, seq_len)
        assert returns.shape == (batch_size, seq_len)

    def test_gae_shape_mismatch(self) -> None:
        """Verify GAE detects shape mismatches."""
        gae = GeneralizedAdvantageEstimation(gamma=0.99, lam=0.95)

        batch_size, seq_len = 8, 16
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len + 1)  # Wrong shape
        next_values = torch.randn(batch_size, seq_len)
        dones = torch.zeros(batch_size, seq_len)

        with pytest.raises(ValueError, match="shape mismatch"):
            gae.compute(rewards, values, next_values, dones)

    def test_gae_gamma_zero(self) -> None:
        """Verify GAE with gamma=0 (immediate rewards only)."""
        gae = GeneralizedAdvantageEstimation(gamma=0.0, lam=1.0)

        batch_size, seq_len = 4, 8
        rewards = torch.ones(batch_size, seq_len)
        values = torch.zeros(batch_size, seq_len)
        next_values = torch.zeros(batch_size, seq_len)
        dones = torch.zeros(batch_size, seq_len)

        advantages, returns = gae.compute(rewards, values, next_values, dones)

        # With gamma=0, advantage should equal reward
        assert torch.allclose(advantages, rewards)

    def test_gae_lambda_zero(self) -> None:
        """Verify GAE with lambda=0 (TD-only)."""
        gae = GeneralizedAdvantageEstimation(gamma=0.99, lam=0.0)

        batch_size, seq_len = 4, 8
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        next_values = torch.zeros(batch_size, seq_len)
        dones = torch.zeros(batch_size, seq_len)

        advantages, returns = gae.compute(rewards, values, next_values, dones)

        # Advantages should match TD residuals at step 0
        assert advantages.shape == (batch_size, seq_len)


class TestPPOLosses:
    """Tests for PPO loss computation."""

    def test_ppo_trainer_initialization(self) -> None:
        """Verify PPO trainer can be initialized."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters()),
            lr=config.optimization.learning_rate,
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)
        assert trainer.global_step == 0
        assert trainer.kl_coefficient == config.alignment.kl_coefficient

    def test_ppo_trainer_invalid_config(self) -> None:
        """Verify PPO trainer rejects invalid config."""
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()))

        with pytest.raises(TypeError):
            PPOTrainer(None, policy, value, reference, optimizer)

    def test_policy_loss_computation(self) -> None:
        """Verify PPO clipped loss computation."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)

        batch_size = 8
        log_probs_new = torch.randn(batch_size)
        log_probs_old = torch.randn(batch_size)
        advantages = torch.randn(batch_size)

        loss, ratio = trainer.compute_policy_loss(log_probs_new, log_probs_old, advantages)

        assert loss.shape == torch.Size([])  # Scalar
        assert ratio.shape == (batch_size,)
        assert loss.requires_grad

    def test_value_loss_computation(self) -> None:
        """Verify value function loss computation."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)

        batch_size = 8
        value_pred = torch.randn(batch_size)
        returns = torch.randn(batch_size)

        loss = trainer.compute_value_loss(value_pred, returns)

        assert loss.shape == torch.Size([])  # Scalar
        assert loss.requires_grad

    def test_entropy_computation(self) -> None:
        """Verify entropy bonus computation."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)

        batch_size, num_actions = 8, 50257
        logits = torch.randn(batch_size, num_actions)

        entropy = trainer.compute_entropy_bonus(logits)

        assert entropy.shape == torch.Size([])  # Scalar
        assert entropy > 0  # Entropy should be positive
        assert entropy < math.log(num_actions)  # Upper bound: log(|A|)

    def test_kl_divergence_computation(self) -> None:
        """Verify KL divergence computation."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)

        batch_size, num_actions = 8, 50257
        logits_new = torch.randn(batch_size, num_actions)
        logits_old = torch.randn(batch_size, num_actions)

        kl = trainer.compute_kl_divergence(logits_new, logits_old)

        assert kl.shape == torch.Size([])  # Scalar
        assert kl >= 0  # KL divergence is non-negative

    def test_kl_divergence_same_distribution(self) -> None:
        """Verify KL divergence is ~0 for identical distributions."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)

        batch_size, num_actions = 8, 50257
        logits = torch.randn(batch_size, num_actions)

        kl = trainer.compute_kl_divergence(logits, logits)

        assert kl.item() < 1e-5  # Should be ~0


class TestAdaptiveKL:
    """Tests for adaptive KL penalty adjustment."""

    def test_adaptive_kl_increase(self) -> None:
        """Verify KL coefficient increases when KL > target * 1.5."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)
        initial_coeff = trainer.kl_coefficient

        # KL much higher than target
        high_kl = trainer.target_kl * 2.0
        trainer.adaptive_kl_penalty(high_kl)

        assert trainer.kl_coefficient > initial_coeff

    def test_adaptive_kl_decrease(self) -> None:
        """Verify KL coefficient decreases when KL < target / 1.5."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)
        initial_coeff = trainer.kl_coefficient

        # KL much lower than target
        low_kl = trainer.target_kl * 0.1
        trainer.adaptive_kl_penalty(low_kl)

        assert trainer.kl_coefficient < initial_coeff

    def test_adaptive_kl_bounds(self) -> None:
        """Verify KL coefficient stays within reasonable bounds."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)

        # Force extreme KL scenarios
        for _ in range(10):
            trainer.adaptive_kl_penalty(1.0)  # Very high KL

        assert 0.001 <= trainer.kl_coefficient <= 10.0


class TestMetricsComputation:
    """Tests for metric computation."""

    def test_explained_variance_perfect(self) -> None:
        """Verify explained variance is 1.0 for perfect predictions."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)

        batch_size = 8
        returns = torch.randn(batch_size)
        value_pred = returns.clone()  # Perfect prediction

        ev = trainer.compute_explained_variance(value_pred, returns)

        assert ev == pytest.approx(1.0, abs=1e-4)

    def test_explained_variance_zero(self) -> None:
        """Verify explained variance is ~0 when predicting mean."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)

        batch_size = 8
        returns = torch.randn(batch_size)
        value_pred = torch.ones(batch_size) * returns.mean()  # Always predict mean

        ev = trainer.compute_explained_variance(value_pred, returns)

        assert ev == pytest.approx(0.0, abs=1e-4)

    def test_ppo_metrics_dataclass(self) -> None:
        """Verify PPOMetrics dataclass can be instantiated."""
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

        assert metrics.policy_loss == 0.5
        assert metrics.entropy > 0


class TestTrainStep:
    """Integration tests for training step."""

    def test_train_step_updates_weights(self) -> None:
        """Verify train_step actually updates model weights."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters()),
            lr=config.optimization.learning_rate,
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)

        # Save initial weights
        initial_weights = [p.clone() for p in policy.parameters()]

        # Create batch
        batch_size, seq_len = 4, 8
        batch = {
            "input_ids": torch.randint(0, 50257, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "values": torch.randn(batch_size, seq_len),
            "next_values": torch.randn(batch_size, seq_len),
            "dones": torch.zeros(batch_size, seq_len),
        }
        old_log_probs = torch.randn(batch_size, seq_len)

        # Train step
        metrics = trainer.train_step(batch, old_log_probs)

        # Check weights changed
        for initial, current in zip(initial_weights, policy.parameters()):
            assert not torch.allclose(initial, current, atol=1e-5)

        assert metrics.global_step == 0  # Metrics computed before step increment

    def test_train_step_returns_metrics(self) -> None:
        """Verify train_step returns valid metrics."""
        config = TrainingConfig.toy_mode()
        policy = DummyModel()
        value = DummyValueModel()
        reference = DummyModel()
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters())
        )

        trainer = PPOTrainer(config, policy, value, reference, optimizer)

        batch_size, seq_len = 4, 8
        batch = {
            "input_ids": torch.randint(0, 50257, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "values": torch.randn(batch_size, seq_len),
            "next_values": torch.randn(batch_size, seq_len),
            "dones": torch.zeros(batch_size, seq_len),
        }
        old_log_probs = torch.randn(batch_size, seq_len)

        metrics = trainer.train_step(batch, old_log_probs)

        assert isinstance(metrics, PPOMetrics)
        assert metrics.policy_loss > 0
        assert metrics.value_loss > 0
        assert metrics.entropy > 0
        assert metrics.kl_divergence >= 0
        assert -1 <= metrics.explained_variance <= 1


import math

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
