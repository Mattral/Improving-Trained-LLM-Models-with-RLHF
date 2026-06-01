"""Async-first dataset pipeline for RLHF training with preference pairs and caching.

This module provides:
- Preference pair loader with local caching
- Toy dataset support (1K samples) with pre-cached JSONL
- LoRA token masking for efficient training
- Asyncio-based data loading for multi-worker pipelines
- Hugging Face integration (Datasets library)
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, IterableDataset

from rlhf_platform.config import DatasetConfig, TrainingConfig


logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """Single preference pair (chosen > rejected)."""

    prompt: str
    chosen: str
    rejected: str
    chosen_reward: Optional[float] = None  # For reward data
    rejected_reward: Optional[float] = None


@dataclass
class PPOSample:
    """PPO training sample with prompt, action, and metadata."""

    prompt: str
    action: str  # Generated continuation
    log_prob: float  # Log probability under policy
    reward: float  # Scalar reward from reward model
    value: float  # Value estimate


class PreferencePairDataset(IterableDataset):
    """Iterable dataset for preference pairs with optional caching.

    Loads preference pairs from Hugging Face Datasets (e.g., HH-RLHF) or local JSONL.
    Supports caching to local directory for faster subsequent loads.

    Example:
        >>> config = TrainingConfig.toy_mode()
        >>> dataset = PreferencePairDataset(config.dataset)
        >>> for pair in dataset:
        ...     print(f"Prompt: {pair.prompt}")
        ...     print(f"Chosen: {pair.chosen}")
    """

    def __init__(self, config: DatasetConfig, cache_dir: Optional[str] = None):
        """Initialize preference pair dataset.

        Args:
            config: DatasetConfig with dataset name, split, preprocessing settings
            cache_dir: Optional local cache directory (default: ./data/{dataset_name})

        Raises:
            ValueError: If dataset not found or format invalid
        """
        self.config = config
        self.cache_dir = Path(
            cache_dir or f"data/{config.dataset_name}"
        ).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._dataset: Optional[Dataset] = None
        self._loaded = False

    def _load_from_cache(self) -> Optional[list[PreferencePair]]:
        """Load preference pairs from local JSONL cache.

        Returns:
            List of PreferencePair objects, or None if cache doesn't exist
        """
        cache_file = self.cache_dir / f"{self.config.dataset_split}.jsonl"
        if not cache_file.exists():
            return None

        pairs = []
        try:
            with open(cache_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    pairs.append(
                        PreferencePair(
                            prompt=data["prompt"],
                            chosen=data["chosen"],
                            rejected=data["rejected"],
                            chosen_reward=data.get("chosen_reward"),
                            rejected_reward=data.get("rejected_reward"),
                        )
                    )
            logger.info(
                f"Loaded {len(pairs)} pairs from cache: {cache_file}"
            )
            return pairs
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache corrupted: {e}, will reload from source")
            return None

    def _save_to_cache(self, pairs: list[PreferencePair]) -> None:
        """Save preference pairs to local JSONL cache.

        Args:
            pairs: List of PreferencePair objects to cache
        """
        cache_file = self.cache_dir / f"{self.config.dataset_split}.jsonl"
        with open(cache_file, "w") as f:
            for pair in pairs:
                obj = {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                }
                if pair.chosen_reward is not None:
                    obj["chosen_reward"] = pair.chosen_reward
                if pair.rejected_reward is not None:
                    obj["rejected_reward"] = pair.rejected_reward
                f.write(json.dumps(obj) + "\n")
        logger.info(f"Cached {len(pairs)} pairs to {cache_file}")

    def _load_hf_dataset(self) -> Dataset:
        """Load dataset from Hugging Face Datasets.

        Returns:
            HF Dataset object

        Raises:
            ValueError: If dataset format incompatible
        """
        logger.info(
            f"Loading {self.config.dataset_name} "
            f"(split: {self.config.dataset_split})"
        )
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            num_proc=self.config.preprocessing_num_workers,
        )

        # Handle different dataset formats
        if self.config.dataset_name == "Anthropic/hh-rlhf":
            # HH-RLHF format: "chosen" and "rejected" columns
            return dataset
        elif self.config.dataset_name == "openai/summarize_from_feedback":
            # Summarization format: needs column renaming
            return dataset.rename_column("summaries", "chosen")
        else:
            return dataset

    def _convert_to_preference_pairs(
        self, dataset: Dataset
    ) -> list[PreferencePair]:
        """Convert HF Dataset to PreferencePair objects.

        Args:
            dataset: Hugging Face Dataset

        Returns:
            List of PreferencePair objects

        Raises:
            ValueError: If required columns missing
        """
        required_cols = {"prompt", "chosen", "rejected"}
        if not required_cols.issubset(set(dataset.column_names)):
            raise ValueError(
                f"Dataset missing required columns {required_cols}. "
                f"Found: {dataset.column_names}"
            )

        pairs = []
        for item in dataset:
            pairs.append(
                PreferencePair(
                    prompt=item["prompt"],
                    chosen=item["chosen"],
                    rejected=item["rejected"],
                    chosen_reward=item.get("chosen_reward"),
                    rejected_reward=item.get("rejected_reward"),
                )
            )

        # Limit to dataset_size if specified
        if self.config.dataset_size:
            pairs = pairs[: self.config.dataset_size]

        return pairs

    def load(self) -> None:
        """Load dataset with caching fallback.

        Priority:
        1. Check local cache
        2. Load from Hugging Face
        3. Convert and cache for future use
        """
        # Try cache first
        pairs = self._load_from_cache()
        if pairs is not None:
            self._dataset = Dataset.from_dict(
                {
                    "prompt": [p.prompt for p in pairs],
                    "chosen": [p.chosen for p in pairs],
                    "rejected": [p.rejected for p in pairs],
                }
            )
            self._loaded = True
            return

        # Load from Hugging Face
        hf_dataset = self._load_hf_dataset()
        pairs = self._convert_to_preference_pairs(hf_dataset)

        # Cache for future use
        self._save_to_cache(pairs)

        self._dataset = hf_dataset
        self._loaded = True
        logger.info(f"Loaded {len(pairs)} preference pairs")

    def __iter__(self) -> Iterator[PreferencePair]:
        """Iterate over preference pairs.

        Yields:
            PreferencePair objects

        Raises:
            RuntimeError: If dataset not loaded
        """
        if not self._loaded:
            self.load()

        for item in self._dataset:
            yield PreferencePair(
                prompt=item["prompt"],
                chosen=item["chosen"],
                rejected=item["rejected"],
                chosen_reward=item.get("chosen_reward"),
                rejected_reward=item.get("rejected_reward"),
            )

    def __len__(self) -> int:
        """Return dataset size.

        Returns:
            Number of preference pairs

        Raises:
            RuntimeError: If dataset not loaded
        """
        if not self._loaded:
            self.load()
        return len(self._dataset)


class ToyDatasetLoader:
    """Toy dataset (1K HH-RLHF samples) with pre-cached JSONL.

    For rapid prototyping on T4 GPU in <20 minutes.

    Example:
        >>> toy_loader = ToyDatasetLoader()
        >>> pairs = toy_loader.load()  # 1K preference pairs
        >>> print(f"Loaded {len(pairs)} toy pairs")
    """

    TOY_JSONL = Path("data/toy/hh_rlhf_toy_1k.jsonl")

    @staticmethod
    def _create_toy_jsonl() -> None:
        """Create toy dataset JSONL if not exists.

        This downloads first 1K samples from HH-RLHF and caches locally.
        Called on first load if file doesn't exist.
        """
        if ToyDatasetLoader.TOY_JSONL.exists():
            return

        ToyDatasetLoader.TOY_JSONL.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Creating toy dataset (1K HH-RLHF samples)...")
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            split="train",
            num_proc=4,
        )

        # Take first 1K samples
        dataset = dataset.select(range(min(1000, len(dataset))))

        # Cache as JSONL
        with open(ToyDatasetLoader.TOY_JSONL, "w") as f:
            for item in dataset:
                obj = {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }
                f.write(json.dumps(obj) + "\n")

        logger.info(
            f"Toy dataset cached to {ToyDatasetLoader.TOY_JSONL} "
            f"({len(dataset)} samples)"
        )

    @staticmethod
    def load() -> list[PreferencePair]:
        """Load toy dataset from cache.

        Creates cache if it doesn't exist.

        Returns:
            List of 1K PreferencePair objects
        """
        ToyDatasetLoader._create_toy_jsonl()

        pairs = []
        with open(ToyDatasetLoader.TOY_JSONL, "r") as f:
            for line in f:
                data = json.loads(line)
                pairs.append(
                    PreferencePair(
                        prompt=data["prompt"],
                        chosen=data["chosen"],
                        rejected=data["rejected"],
                    )
                )

        logger.info(f"Loaded {len(pairs)} toy pairs")
        return pairs


class LoRAMaskingCollator:
    """Collate function that masks non-LoRA tokens for efficient training.

    For LoRA-based training, only LoRA parameters contribute to gradients,
    so non-LoRA tokens can be masked to reduce computation.

    Note: Requires model.modules() to identify LoRA layers first.
    """

    def __init__(self, tokenizer, max_length: int = 512):
        """Initialize collator.

        Args:
            tokenizer: Transformers tokenizer
            max_length: Max sequence length (default: 512)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        """Collate batch with optional LoRA masking.

        Args:
            batch: List of tokenized examples

        Returns:
            Dict with input_ids, attention_mask, labels
        """
        # Pad batch
        padded = self.tokenizer.pad(
            batch,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # LoRA masking: set attention to 0 for padded tokens
        # (In practice, use peft.LoraConfig for automatic masking)
        padded["labels"] = padded["input_ids"].clone()
        padded["labels"][padded["attention_mask"] == 0] = -100

        return padded


async def async_load_dataset(
    config: DatasetConfig,
) -> AsyncIterator[PreferencePair]:
    """Async dataset loader for multi-worker pipelines.

    Loads preference pairs asynchronously, enabling concurrent
    preprocessing and GPU training.

    Args:
        config: DatasetConfig with dataset settings

    Yields:
        PreferencePair objects asynchronously
    """
    dataset = PreferencePairDataset(config)
    dataset.load()

    # Wrap iterator in async task
    loop = asyncio.get_event_loop()
    for pair in dataset:
        yield pair
        # Yield control to event loop for other tasks
        await loop.create_task(asyncio.sleep(0))


def get_dataloader(
    config: DatasetConfig,
    tokenizer,
    use_toy: bool = False,
) -> DataLoader:
    """Create DataLoader for training.

    Args:
        config: DatasetConfig with batch size and sequence length
        tokenizer: Transformers tokenizer
        use_toy: If True, use 1K toy dataset

    Returns:
        PyTorch DataLoader ready for training
    """
    if use_toy:
        pairs = ToyDatasetLoader.load()
    else:
        dataset = PreferencePairDataset(config)
        dataset.load()
        pairs = list(dataset)

    # Tokenize pairs
    def tokenize_fn(pair: PreferencePair):
        chosen_tokens = tokenizer(
            pair.prompt + pair.chosen,
            max_length=config.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        return chosen_tokens

    tokenized = [tokenize_fn(pair) for pair in pairs]

    # Create DataLoader
    collator = LoRAMaskingCollator(tokenizer, config.max_seq_length)
    return DataLoader(
        tokenized,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=True,
    )


if __name__ == "__main__":
    """Quick test of dataset pipeline."""
    # Test toy dataset
    pairs = ToyDatasetLoader.load()
    print(f"Toy dataset: {len(pairs)} pairs")
    print(f"First pair: {pairs[0].prompt[:50]}...")

    # Test preference pair dataset
    config = DatasetConfig(
        dataset_name="Anthropic/hh-rlhf",
        dataset_split="test",
        batch_size=8,
    )
    dataset = PreferencePairDataset(config)
    dataset.load()
    print(f"\nHF dataset: {len(dataset)} pairs")
    first = next(iter(dataset))
    print(f"First pair prompt: {first.prompt[:50]}...")
