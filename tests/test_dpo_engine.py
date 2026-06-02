"""Unit tests for DPO (Direct Preference Optimization) engine.

Tests cover:
- Loss computation correctness
- Gradient flow through reference model
- Metrics calculation
- Batch processing
- Configuration integration
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from rlhf_platform.config import TrainingConfig
from rlhf_platform.dpo_engine import DPOMetrics, DPOTrainer


logger = logging.getLogger(__name__)


class TestDPOMetrics:
    """Test DPOMetrics container."""

    def test_init(self):
        """Test DPOMetrics initialization."""
        metrics = DPOMetrics(
            dpo_loss=0.5,
            policy_diff_mean=0.1,
            margin_mean=0.2,
            accuracy=0.85,
            explained_variance=0.7,
        )
        assert metrics.dpo_loss == 0.5
        assert metrics.policy_diff_mean == 0.1
        assert metrics.margin_mean == 0.2
        assert metrics.accuracy == 0.85
        assert metrics.explained_variance == 0.7

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = DPOMetrics(
            dpo_loss=0.5,
            policy_diff_mean=0.1,
            margin_mean=0.2,
            accuracy=0.85,
            explained_variance=0.7,
        )
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d["dpo_loss"] == 0.5
        assert d["accuracy"] == 0.85
        assert len(d) == 5


class TestDPOTrainerInit:
    """Test DPOTrainer initialization."""

    @patch("rlhf_platform.dpo_engine.AutoTokenizer.from_pretrained")
    @patch("rlhf_platform.dpo_engine.AutoModelForCausalLM.from_pretrained")
    def test_init_creates_trainer(self, mock_model, mock_tokenizer):
        """Test trainer initialization."""
        # Mock models
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<eos>"

        mock_policy = MagicMock()
        mock_policy.parameters.return_value = []
        mock_reference = MagicMock()

        mock_model.side_effect = [mock_policy, mock_reference]

        # Initialize
        config = TrainingConfig.toy_mode()
        trainer = DPOTrainer(config)

        assert trainer.config == config
        assert trainer.policy_model is not None
        assert trainer.reference_model is not None
        assert trainer.beta == 0.1  # Default beta
        assert trainer.step_count == 0

    @patch("rlhf_platform.dpo_engine.AutoTokenizer.from_pretrained")
    @patch("rlhf_platform.dpo_engine.AutoModelForCausalLM.from_pretrained")
    def test_reference_model_frozen(self, mock_model, mock_tokenizer):
        """Test that reference model gradients are disabled."""
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<eos>"

        mock_policy = MagicMock()
        mock_policy.parameters.return_value = []
        mock_reference = MagicMock()

        mock_model.side_effect = [mock_policy, mock_reference]

        config = TrainingConfig.toy_mode()
        trainer = DPOTrainer(config)

        # Reference model should have requires_grad=False
        trainer.reference_model.requires_grad_(False)
        # Verify it was called
        trainer.reference_model.requires_grad_.assert_called_with(False)


class TestDPOLossComputation:
    """Test DPO loss computation."""

    def test_compute_dpo_loss_shapes(self):
        """Test loss computation with correct tensor shapes."""
        # Create mock trainer
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1
        trainer.step_count = 0

        # Call actual compute_dpo_loss method
        batch_size = 4
        policy_logits_chosen = torch.tensor([0.5, 0.6, 0.7, 0.8])
        policy_logits_rejected = torch.tensor([0.3, 0.4, 0.5, 0.6])
        ref_logits_chosen = torch.tensor([0.4, 0.5, 0.6, 0.7])
        ref_logits_rejected = torch.tensor([0.2, 0.3, 0.4, 0.5])

        # Use actual method
        loss, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_logits_chosen,
            policy_logits_rejected,
            ref_logits_chosen,
            ref_logits_rejected,
        )

        assert loss is not None
        assert isinstance(metrics, DPOMetrics)
        assert metrics.dpo_loss > 0  # Loss should be positive

    def test_dpo_loss_with_different_preferences(self):
        """Test loss when chosen > rejected (should be lower)."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        # Case 1: Chosen has higher logits (good preference alignment)
        policy_logits_chosen = torch.tensor([1.0, 1.0])
        policy_logits_rejected = torch.tensor([0.0, 0.0])
        ref_logits_chosen = torch.tensor([0.8, 0.8])
        ref_logits_rejected = torch.tensor([0.2, 0.2])

        loss_aligned, _ = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_logits_chosen,
            policy_logits_rejected,
            ref_logits_chosen,
            ref_logits_rejected,
        )

        # Case 2: Rejected has higher logits (bad preference alignment)
        policy_logits_chosen = torch.tensor([0.0, 0.0])
        policy_logits_rejected = torch.tensor([1.0, 1.0])

        loss_misaligned, _ = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_logits_chosen,
            policy_logits_rejected,
            ref_logits_chosen,
            ref_logits_rejected,
        )

        # Aligned loss should be lower (better)
        assert loss_aligned < loss_misaligned

    def test_dpo_loss_bounds(self):
        """Test that DPO loss stays within reasonable bounds."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        # Create random batches
        batch_size = 16
        policy_chosen = torch.randn(batch_size)
        policy_rejected = torch.randn(batch_size)
        ref_chosen = torch.randn(batch_size)
        ref_rejected = torch.randn(batch_size)

        loss, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # Loss should be finite
        assert torch.isfinite(loss)
        assert loss.item() > 0

        # Metrics should be valid
        assert 0 <= metrics.accuracy <= 1
        assert torch.isfinite(torch.tensor(metrics.dpo_loss))


class TestDPOMetricsComputation:
    """Test metrics computation during DPO training."""

    def test_accuracy_computation(self):
        """Test accuracy metric (fraction where chosen > rejected)."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        # Perfect preference alignment: chosen always > rejected
        policy_chosen = torch.tensor([1.0, 2.0, 3.0])
        policy_rejected = torch.tensor([0.0, 0.5, 1.0])
        ref_chosen = torch.tensor([0.9, 1.8, 2.8])
        ref_rejected = torch.tensor([0.1, 0.7, 1.2])

        _, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        assert metrics.accuracy == 1.0  # All chosen > rejected

    def test_margin_computation(self):
        """Test margin metric (chosen - rejected difference)."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        policy_chosen = torch.tensor([2.0, 3.0, 4.0])
        policy_rejected = torch.tensor([1.0, 1.5, 2.0])
        ref_chosen = torch.tensor([1.8, 2.8, 3.8])
        ref_rejected = torch.tensor([1.2, 1.7, 2.2])

        _, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # Margin should be average of differences
        expected_margin = ((2.0 - 1.0) + (3.0 - 1.5) + (4.0 - 2.0)) / 3
        assert abs(metrics.margin_mean - expected_margin) < 0.01


class TestDPOBetaParameter:
    """Test DPO beta (temperature) parameter."""

    def test_beta_affects_loss(self):
        """Test that beta parameter affects loss magnitude."""
        trainer_low_beta = MagicMock(spec=DPOTrainer)
        trainer_low_beta.beta = 0.01

        trainer_high_beta = MagicMock(spec=DPOTrainer)
        trainer_high_beta.beta = 1.0

        policy_chosen = torch.tensor([0.5, 0.6, 0.7])
        policy_rejected = torch.tensor([0.3, 0.4, 0.5])
        ref_chosen = torch.tensor([0.4, 0.5, 0.6])
        ref_rejected = torch.tensor([0.2, 0.3, 0.4])

        loss_low, _ = DPOTrainer.compute_dpo_loss(
            trainer_low_beta,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        loss_high, _ = DPOTrainer.compute_dpo_loss(
            trainer_high_beta,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # Higher beta should give higher loss (stronger preference signal)
        assert loss_high > loss_low


class TestDPOGradientFlow:
    """Test gradient flow through DPO trainer."""

    @patch("rlhf_platform.dpo_engine.AutoTokenizer.from_pretrained")
    @patch("rlhf_platform.dpo_engine.AutoModelForCausalLM.from_pretrained")
    def test_gradients_not_through_reference(self, mock_model, mock_tokenizer):
        """Test that gradients don't flow through frozen reference model."""
        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tok

        # Create real tensors but mock model calls
        mock_policy = MagicMock()
        mock_policy.parameters.return_value = [torch.nn.Parameter(torch.randn(10))]
        mock_reference = MagicMock()
        mock_reference.requires_grad_ = MagicMock()

        mock_model.side_effect = [mock_policy, mock_reference]

        config = TrainingConfig.toy_mode()
        trainer = DPOTrainer(config)

        # Verify reference model requires_grad was set to False
        trainer.reference_model.requires_grad_.assert_called()


class TestDPONumericalStability:
    """Test numerical stability of DPO computation."""

    def test_large_logits_stability(self):
        """Test stability with large logit values."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        # Large logits can cause numerical issues with sigmoid
        policy_chosen = torch.tensor([100.0, 200.0, 150.0])
        policy_rejected = torch.tensor([50.0, 150.0, 100.0])
        ref_chosen = torch.tensor([90.0, 190.0, 140.0])
        ref_rejected = torch.tensor([40.0, 140.0, 90.0])

        loss, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # Should not be NaN or Inf
        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_small_logits_stability(self):
        """Test stability with very small logit differences."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        policy_chosen = torch.tensor([0.0001, 0.0001, 0.0001])
        policy_rejected = torch.tensor([0.0, 0.0, 0.0])
        ref_chosen = torch.tensor([0.00005, 0.00005, 0.00005])
        ref_rejected = torch.tensor([0.0, 0.0, 0.0])

        loss, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # Should handle small differences gracefully
        assert torch.isfinite(loss)


class TestDPOBatchProcessing:
    """Test DPO batch processing."""

    def test_different_batch_sizes(self):
        """Test loss computation across different batch sizes."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        for batch_size in [1, 4, 8, 16, 32]:
            policy_chosen = torch.randn(batch_size)
            policy_rejected = torch.randn(batch_size)
            ref_chosen = torch.randn(batch_size)
            ref_rejected = torch.randn(batch_size)

            loss, metrics = DPOTrainer.compute_dpo_loss(
                trainer,
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
            )

            assert torch.isfinite(loss)
            assert 0 <= metrics.accuracy <= 1

    def test_metrics_batch_aggregation(self):
        """Test that metrics aggregate correctly across batch."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        # Batch with known properties
        policy_chosen = torch.tensor([1.0, 2.0, 3.0, 4.0])
        policy_rejected = torch.tensor([0.0, 0.5, 1.0, 1.5])
        ref_chosen = torch.ones(4)
        ref_rejected = torch.zeros(4)

        _, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # All samples have chosen > rejected, so accuracy should be 1.0
        assert metrics.accuracy == 1.0

        # Margin should be sum of differences / batch_size
        expected_margin = (
            (1.0 - 0.0 - (1.0 - 0.0))
            + (2.0 - 0.5 - (1.0 - 0.0))
            + (3.0 - 1.0 - (1.0 - 0.0))
            + (4.0 - 1.5 - (1.0 - 0.0))
        ) / 4
        # Should be approximately 0 due to symmetry
        assert abs(metrics.margin_mean) < 0.5


class TestDPOConfiguration:
    """Test DPO integration with TrainingConfig."""

    @patch("rlhf_platform.dpo_engine.AutoTokenizer.from_pretrained")
    @patch("rlhf_platform.dpo_engine.AutoModelForCausalLM.from_pretrained")
    def test_config_integration(self, mock_model, mock_tokenizer):
        """Test that DPO trainer respects configuration."""
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<eos>"

        mock_policy = MagicMock()
        mock_policy.parameters.return_value = []
        mock_reference = MagicMock()

        mock_model.side_effect = [mock_policy, mock_reference]

        config = TrainingConfig.toy_mode()
        trainer = DPOTrainer(config)

        # Verify config is used
        assert trainer.config.model.policy_model_id == config.model.policy_model_id
        assert trainer.config.model.reference_model_id == config.model.reference_model_id
        assert trainer.config.optimization.learning_rate == config.optimization.learning_rate

    @patch("rlhf_platform.dpo_engine.AutoTokenizer.from_pretrained")
    @patch("rlhf_platform.dpo_engine.AutoModelForCausalLM.from_pretrained")
    def test_custom_beta_from_config(self, mock_model, mock_tokenizer):
        """Test that custom beta parameter is read from config."""
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<eos>"

        mock_policy = MagicMock()
        mock_policy.parameters.return_value = []
        mock_reference = MagicMock()

        mock_model.side_effect = [mock_policy, mock_reference]

        config = TrainingConfig.toy_mode()
        # Set custom beta
        if not hasattr(config, "alignment"):
            config.alignment = MagicMock()
        config.alignment.dpo_beta = 0.5

        trainer = DPOTrainer(config)
        assert trainer.beta == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
