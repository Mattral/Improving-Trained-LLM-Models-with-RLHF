"""Command-line interface (CLI) for RLHF training pipelines.

Provides:
- train-sft: Supervised fine-tuning on instruction-following data
- train-reward: Reward model training on preference pairs
- run-ppo: PPO training for policy alignment
- run-dpo: DPO training as alternative to PPO
- Global --toy flag for small-model testing on T4 (<20 minutes)
- Config validation before execution
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from rlhf_platform.config import TrainingConfig
from rlhf_platform.sft_engine import SFTTrainer
from rlhf_platform.reward_engine import RewardModelTrainer
from rlhf_platform.ppo_engine import PPOTrainer


logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="rlhf-platform",
    help="Production-grade RLHF training platform with PPO and DPO support",
)


def setup_logging(verbose: bool = False) -> None:
    """Setup console logging.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def train_sft(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to training config YAML (default: configs/default.yaml)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for checkpoints (default: output/sft_{timestamp})",
    ),
    num_epochs: int = typer.Option(
        3,
        "--epochs",
        help="Number of training epochs",
    ),
    toy: bool = typer.Option(
        False,
        "--toy",
        help="Use toy dataset (1K samples) for quick testing on T4",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable debug logging",
    ),
) -> None:
    """Train SFT (supervised fine-tuning) model on instruction data.

    This stage fine-tunes the base model on high-quality instruction-following
    examples before RLHF training.

    Example:
        python -m rlhf_platform.cli train-sft --toy --epochs 1
        python -m rlhf_platform.cli train-sft --config configs/default.yaml
    """
    setup_logging(verbose)

    try:
        # Load config
        if config:
            train_config = TrainingConfig.from_yaml(config)
            console.print(f"[green]✓[/green] Loaded config: {config}")
        else:
            if toy:
                train_config = TrainingConfig.toy_mode()
                console.print("[green]✓[/green] Using toy config")
            else:
                train_config = TrainingConfig.from_yaml("configs/default.yaml")
                console.print(
                    "[green]✓[/green] Using default config"
                )

        # Validate config
        if train_config.optimization.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if train_config.dataset.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        console.print(
            f"[dim]Config: {train_config.model.policy_model_id}, "
            f"batch_size={train_config.dataset.batch_size}[/dim]"
        )

        # Setup output directory
        if output_dir is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"output/sft_{timestamp}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Train
        console.print(
            f"\n[bold cyan]Starting SFT training...[/bold cyan]"
        )
        trainer = SFTTrainer(train_config, use_toy=toy)
        result = trainer.train(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
        )

        console.print(
            f"\n[green]✓ SFT training complete[/green]\n"
            f"  Output: {output_path}\n"
            f"  Final Loss: {result.training_loss:.4f}"
        )

        # Save model
        trainer.save_model(str(output_path))
        console.print(f"[green]✓[/green] Model saved to {output_path}")

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]", style="bold")
        raise typer.Exit(1)


@app.command()
def train_reward(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to training config YAML",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for checkpoints",
    ),
    num_epochs: int = typer.Option(
        3,
        "--epochs",
        help="Number of training epochs",
    ),
    toy: bool = typer.Option(
        False,
        "--toy",
        help="Use toy dataset for quick testing",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable debug logging",
    ),
) -> None:
    """Train reward model on preference pairs.

    The reward model learns to score chosen > rejected responses,
    enabling PPO to optimize the policy.

    Example:
        python -m rlhf_platform.cli train-reward --toy --epochs 1
    """
    setup_logging(verbose)

    try:
        # Load config
        if config:
            train_config = TrainingConfig.from_yaml(config)
        else:
            if toy:
                train_config = TrainingConfig.toy_mode()
            else:
                train_config = TrainingConfig.from_yaml("configs/default.yaml")

        console.print(
            f"[dim]Config: {train_config.model.reward_model_id}, "
            f"batch_size={train_config.dataset.batch_size}[/dim]"
        )

        # Setup output directory
        if output_dir is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"output/reward_model_{timestamp}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Train
        console.print(
            f"\n[bold cyan]Starting reward model training...[/bold cyan]"
        )
        trainer = RewardModelTrainer(train_config, use_toy=toy)
        result = trainer.train(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
        )

        console.print(
            f"\n[green]✓ Reward training complete[/green]\n"
            f"  Output: {output_path}\n"
            f"  Final Loss: {result['loss']:.4f}\n"
            f"  Accuracy: {result['accuracy']:.4f}"
        )

        # Save model
        trainer.save_model(str(output_path))
        console.print(f"[green]✓[/green] Model saved to {output_path}")

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]", style="bold")
        raise typer.Exit(1)


@app.command()
def run_ppo(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to training config YAML",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for checkpoints",
    ),
    num_epochs: int = typer.Option(
        3,
        "--epochs",
        help="Number of training epochs",
    ),
    toy: bool = typer.Option(
        False,
        "--toy",
        help="Use toy dataset for quick testing",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable debug logging",
    ),
) -> None:
    """Run PPO (Proximal Policy Optimization) training.

    Fine-tunes the policy to maximize rewards while staying close
    to the reference model (KL constraint).

    Example:
        python -m rlhf_platform.cli run-ppo --toy --epochs 1
    """
    setup_logging(verbose)

    try:
        # Load config
        if config:
            train_config = TrainingConfig.from_yaml(config)
        else:
            if toy:
                train_config = TrainingConfig.toy_mode()
            else:
                train_config = TrainingConfig.from_yaml("configs/default.yaml")

        # Validate alignment config
        if train_config.alignment.method != "ppo":
            raise ValueError(
                f"Alignment method must be 'ppo', got "
                f"'{train_config.alignment.method}'"
            )

        console.print(
            f"[dim]Config: {train_config.model.policy_model_id}, "
            f"ppo_epsilon={train_config.alignment.ppo_epsilon}, "
            f"target_kl={train_config.alignment.target_kl}[/dim]"
        )

        # Setup output directory
        if output_dir is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"output/ppo_{timestamp}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(
            f"\n[bold cyan]Starting PPO training...[/bold cyan]"
        )
        console.print(
            f"[dim]This requires pre-trained SFT and reward models.[/dim]"
        )

        # NOTE: Full PPO training requires:
        # 1. Pre-trained SFT model
        # 2. Reward model
        # 3. Rollout trajectories
        # 4. Policy + value networks
        # For now, just validate config and show setup instructions

        console.print(
            f"\n[yellow]⚠ PPO training requires:[/yellow]\n"
            f"  1. Pre-trained SFT model from 'train-sft'\n"
            f"  2. Trained reward model from 'train-reward'\n"
            f"  3. Inference pipeline for trajectory generation\n"
            f"\n[dim]Full integration coming in Phase 3.5[/dim]"
        )

        # Save config for reference
        train_config.to_yaml(str(output_path / "ppo_config.yaml"))
        console.print(f"[green]✓[/green] Config saved to {output_path}")

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]", style="bold")
        raise typer.Exit(1)


@app.command()
def run_dpo(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to training config YAML",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for checkpoints",
    ),
    toy: bool = typer.Option(
        False,
        "--toy",
        help="Use toy dataset for quick testing",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable debug logging",
    ),
) -> None:
    """Run DPO (Direct Preference Optimization) training.

    Alternative to PPO that directly optimizes on preference pairs
    without learning a separate reward model.

    Example:
        python -m rlhf_platform.cli run-dpo --toy
    """
    setup_logging(verbose)

    console.print(
        f"[yellow]ℹ DPO implementation coming in Phase 4[/yellow]"
    )
    console.print(
        f"[dim]Use 'run-ppo' for preference optimization in Phase 3[/dim]"
    )

    raise typer.Exit(0)


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        help="Show version",
    ),
) -> None:
    """RLHF Platform - Production-grade alignment training.

    For detailed help on any command:
        python -m rlhf_platform.cli COMMAND --help
    """
    if version:
        console.print("rlhf-platform version 0.1.0")
        raise typer.Exit()


if __name__ == "__main__":
    app()
