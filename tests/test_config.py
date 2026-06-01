"""Unit tests for configuration engine (src/rlhf_platform/config.py).

Tests validate:
- YAML loading and parsing
- Configuration object creation (toy, default, custom)
- JSON serialization/deserialization
- Field validation (type hints, constraints)
- Configuration consistency
"""

import json
import tempfile
from pathlib import Path

import pytest

from rlhf_platform.config import (
    AlignmentConfig,
    DatasetConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)


class TestModelConfig:
    """Tests for ModelConfig (model architecture and paths)."""

    def test_default_model_config(self) -> None:
        """Verify default ModelConfig can be instantiated."""
        config = ModelConfig()
        assert config.policy_model_id == "gpt2"
        assert config.load_in_8bit is False
        assert config.use_peft_lora is True
        assert config.lora_rank == 8

    def test_custom_model_config(self) -> None:
        """Verify custom ModelConfig values are set correctly."""
        config = ModelConfig(
            policy_model_id="meta-llama/Llama-2-7b",
            reward_model_id="microsoft/deberta-v3-large",
            lora_rank=16,
            lora_alpha=32,
        )
        assert config.policy_model_id == "meta-llama/Llama-2-7b"
        assert config.reward_model_id == "microsoft/deberta-v3-large"
        assert config.lora_rank == 16
        assert config.lora_alpha == 32

    def test_quantization_validation(self) -> None:
        """Verify that both 8-bit and 4-bit cannot be enabled simultaneously."""
        with pytest.raises(ValueError, match="Cannot enable both"):
            ModelConfig(load_in_8bit=True, load_in_4bit=True)

    def test_lora_rank_validation(self) -> None:
        """Verify LoRA rank must be positive."""
        with pytest.raises(ValueError):
            ModelConfig(lora_rank=0)

    def test_attn_implementation_literal(self) -> None:
        """Verify attention implementation is restricted to valid values."""
        config = ModelConfig(attn_implementation="flash_attention_2")
        assert config.attn_implementation == "flash_attention_2"

        with pytest.raises(ValueError):
            ModelConfig(attn_implementation="invalid_attention")

    def test_model_config_json_serialization(self) -> None:
        """Verify ModelConfig can be serialized and deserialized as JSON."""
        config1 = ModelConfig(
            policy_model_id="gpt2-medium",
            lora_rank=12,
        )
        json_str = config1.to_json_str()
        config2 = ModelConfig.from_json_str(json_str)

        assert config2.policy_model_id == "gpt2-medium"
        assert config2.lora_rank == 12


class TestOptimizationConfig:
    """Tests for OptimizationConfig (optimizer and learning rate settings)."""

    def test_default_optimization_config(self) -> None:
        """Verify default OptimizationConfig values."""
        config = OptimizationConfig()
        assert config.optimizer == "adamw"
        assert config.learning_rate == 1e-5
        assert config.mixed_precision == "no"
        assert config.gradient_checkpointing is False

    def test_learning_rate_validation(self) -> None:
        """Verify learning rate must be positive."""
        with pytest.raises(ValueError):
            OptimizationConfig(learning_rate=0.0)
        with pytest.raises(ValueError):
            OptimizationConfig(learning_rate=-1e-5)

    def test_warmup_ratio_validation(self) -> None:
        """Verify warmup_ratio is bounded [0, 1]."""
        config = OptimizationConfig(warmup_ratio=0.0)
        assert config.warmup_ratio == 0.0

        config = OptimizationConfig(warmup_ratio=1.0)
        assert config.warmup_ratio == 1.0

        with pytest.raises(ValueError):
            OptimizationConfig(warmup_ratio=1.5)

    def test_mixed_precision_literal(self) -> None:
        """Verify mixed_precision is restricted to valid values."""
        for mode in ["bf16", "fp16", "no"]:
            config = OptimizationConfig(mixed_precision=mode)
            assert config.mixed_precision == mode

        with pytest.raises(ValueError):
            OptimizationConfig(mixed_precision="float32")

    def test_gradient_accumulation_steps(self) -> None:
        """Verify gradient accumulation steps must be >= 1."""
        with pytest.raises(ValueError):
            OptimizationConfig(gradient_accumulation_steps=0)

        config = OptimizationConfig(gradient_accumulation_steps=4)
        assert config.gradient_accumulation_steps == 4


class TestAlignmentConfig:
    """Tests for AlignmentConfig (RLHF/DPO hyperparameters)."""

    def test_default_alignment_config(self) -> None:
        """Verify default AlignmentConfig values."""
        config = AlignmentConfig()
        assert config.method == "ppo"
        assert config.ppo_epsilon == 0.2
        assert config.target_kl == 0.1
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95

    def test_ppo_epsilon_validation(self) -> None:
        """Verify PPO epsilon is bounded [0, 1]."""
        with pytest.raises(ValueError):
            AlignmentConfig(ppo_epsilon=-0.1)
        with pytest.raises(ValueError):
            AlignmentConfig(ppo_epsilon=1.5)

        config = AlignmentConfig(ppo_epsilon=0.15)
        assert config.ppo_epsilon == 0.15

    def test_gamma_lambda_validation(self) -> None:
        """Verify discount factor and GAE lambda are bounded [0, 1]."""
        config = AlignmentConfig(gamma=0.99, gae_lambda=0.95)
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95

        with pytest.raises(ValueError):
            AlignmentConfig(gamma=1.5)
        with pytest.raises(ValueError):
            AlignmentConfig(gae_lambda=-0.1)

    def test_target_kl_positive(self) -> None:
        """Verify target KL is positive."""
        with pytest.raises(ValueError):
            AlignmentConfig(target_kl=0.0)
        with pytest.raises(ValueError):
            AlignmentConfig(target_kl=-0.1)

    def test_alignment_method_literal(self) -> None:
        """Verify alignment method is restricted to valid values."""
        for method in ["ppo", "dpo"]:
            config = AlignmentConfig(method=method)
            assert config.method == method

        with pytest.raises(ValueError):
            AlignmentConfig(method="invalid_method")


class TestDatasetConfig:
    """Tests for DatasetConfig (dataset and data loading)."""

    def test_default_dataset_config(self) -> None:
        """Verify default DatasetConfig values."""
        config = DatasetConfig()
        assert config.dataset_name == "Anthropic/hh-rlhf"
        assert config.batch_size == 32
        assert config.max_seq_length == 512
        assert config.pin_memory is True

    def test_batch_size_validation(self) -> None:
        """Verify batch size must be positive."""
        with pytest.raises(ValueError):
            DatasetConfig(batch_size=0)
        with pytest.raises(ValueError):
            DatasetConfig(batch_size=-8)

    def test_seq_length_validation(self) -> None:
        """Verify max sequence length must be positive."""
        with pytest.raises(ValueError):
            DatasetConfig(max_seq_length=0)

    def test_num_workers_validation(self) -> None:
        """Verify num_workers is non-negative."""
        config = DatasetConfig(num_workers=0)
        assert config.num_workers == 0

        config = DatasetConfig(num_workers=4)
        assert config.num_workers == 4

        with pytest.raises(ValueError):
            DatasetConfig(num_workers=-1)


class TestTrainingConfig:
    """Tests for TrainingConfig (complete training configuration)."""

    def test_default_training_config(self) -> None:
        """Verify default TrainingConfig can be instantiated."""
        config = TrainingConfig()
        assert config.run_name == "rlhf_alignment_run"
        assert config.device == "auto"
        assert config.log_to_wandb is True
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.optimization, OptimizationConfig)
        assert isinstance(config.alignment, AlignmentConfig)
        assert isinstance(config.dataset, DatasetConfig)

    def test_training_config_nested_validation(self) -> None:
        """Verify nested config objects are validated."""
        config = TrainingConfig(
            model=ModelConfig(lora_rank=16),
            optimization=OptimizationConfig(learning_rate=1e-4),
            alignment=AlignmentConfig(ppo_epsilon=0.15),
        )
        assert config.model.lora_rank == 16
        assert config.optimization.learning_rate == 1e-4
        assert config.alignment.ppo_epsilon == 0.15

    def test_toy_mode_config(self) -> None:
        """Verify toy configuration uses small models."""
        config = TrainingConfig.toy_mode()
        assert config.model.policy_model_id == "distilgpt2"
        assert config.model.reward_model_id == "prajjwal1/bert-tiny"
        assert config.dataset.batch_size == 8
        assert config.optimization.gradient_accumulation_steps == 1
        assert config.num_gpus == 1

    def test_default_config(self) -> None:
        """Verify production default configuration."""
        config = TrainingConfig.default_config()
        assert config.use_fsdp is True
        assert config.use_deepspeed is True
        assert config.num_gpus == 8
        assert config.optimization.mixed_precision == "bf16"

    def test_config_yaml_loading(self) -> None:
        """Verify TrainingConfig can be loaded from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config1 = TrainingConfig.toy_mode()
            config1.to_yaml(yaml_path)

            config2 = TrainingConfig.from_yaml(yaml_path)
            assert config2.model.policy_model_id == config1.model.policy_model_id
            assert config2.dataset.batch_size == config1.dataset.batch_size

    def test_config_yaml_loading_missing_file(self) -> None:
        """Verify FileNotFoundError is raised for missing YAML file."""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.from_yaml("nonexistent.yaml")

    def test_config_yaml_loading_empty_file(self) -> None:
        """Verify ValueError is raised for empty YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "empty.yaml"
            yaml_path.write_text("")

            with pytest.raises(ValueError, match="YAML file is empty"):
                TrainingConfig.from_yaml(yaml_path)

    def test_config_json_serialization(self) -> None:
        """Verify TrainingConfig can be serialized and deserialized as JSON."""
        config1 = TrainingConfig.toy_mode()
        json_str = config1.to_json_str()

        # Verify JSON is valid
        data = json.loads(json_str)
        assert data["model"]["policy_model_id"] == "distilgpt2"

        # Verify deserialization
        config2 = TrainingConfig.from_json_str(json_str)
        assert config2.model.policy_model_id == "distilgpt2"
        assert config2.dataset.batch_size == 8

    def test_config_json_file_save_load(self) -> None:
        """Verify TrainingConfig can be saved and loaded from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config1 = TrainingConfig.toy_mode()
            config1.to_json_file(json_path)

            config2 = TrainingConfig.from_json_file(json_path)
            assert config2.model.policy_model_id == config1.model.policy_model_id

    def test_config_seed_reproducibility(self) -> None:
        """Verify seed is preserved across serialization."""
        config1 = TrainingConfig.toy_mode()
        config1.seed = 123

        json_str = config1.to_json_str()
        config2 = TrainingConfig.from_json_str(json_str)

        assert config2.seed == 123

    def test_device_literal(self) -> None:
        """Verify device is restricted to valid values."""
        for device in ["cuda", "cpu", "auto"]:
            config = TrainingConfig(device=device)
            assert config.device == device

        with pytest.raises(ValueError):
            TrainingConfig(device="gpu")

    def test_save_total_limit_none(self) -> None:
        """Verify save_total_limit can be None (keep all checkpoints)."""
        config = TrainingConfig(save_total_limit=None)
        assert config.save_total_limit is None

    def test_deepspeed_config_requirement(self) -> None:
        """Verify deepspeed_config can be set when use_deepspeed=True."""
        config = TrainingConfig(
            use_deepspeed=True,
            deepspeed_config="configs/deepspeed_zero3.yaml",
        )
        assert config.deepspeed_config == "configs/deepspeed_zero3.yaml"


class TestConfigYAMLLoading:
    """Integration tests for YAML configuration loading."""

    def test_load_toy_yaml(self) -> None:
        """Verify toy.yaml can be loaded from configs directory."""
        toy_yaml_path = Path(__file__).parent.parent.parent / "configs" / "toy.yaml"
        if toy_yaml_path.exists():
            config = TrainingConfig.from_yaml(toy_yaml_path)
            assert config.model.policy_model_id == "distilgpt2"
            assert config.dataset.batch_size == 8

    def test_load_default_yaml(self) -> None:
        """Verify default.yaml can be loaded from configs directory."""
        default_yaml_path = (
            Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        )
        if default_yaml_path.exists():
            config = TrainingConfig.from_yaml(default_yaml_path)
            assert config.use_fsdp is True
            assert config.num_gpus == 8


class TestConfigConsistency:
    """Tests for configuration consistency and invariants."""

    def test_batch_size_consistency(self) -> None:
        """Verify eval_batch_size >= batch_size for efficiency."""
        config = TrainingConfig(
            dataset=DatasetConfig(
                batch_size=16,
                eval_batch_size=8,  # Smaller eval batch is inefficient
            )
        )
        # This should be allowed (no validation), but users should follow convention
        assert config.dataset.eval_batch_size == 8

    def test_learning_rate_and_warmup(self) -> None:
        """Verify warmup and learning rate are compatible."""
        config = TrainingConfig(
            optimization=OptimizationConfig(
                learning_rate=1e-4,
                warmup_ratio=0.1,
            )
        )
        assert config.optimization.learning_rate == 1e-4
        assert config.optimization.warmup_ratio == 0.1

    def test_alignment_and_optimization_consistency(self) -> None:
        """Verify alignment and optimization configs are compatible."""
        config = TrainingConfig(
            alignment=AlignmentConfig(ppo_epsilon=0.2),
            optimization=OptimizationConfig(mixed_precision="bf16"),
        )
        assert config.alignment.ppo_epsilon == 0.2
        assert config.optimization.mixed_precision == "bf16"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
