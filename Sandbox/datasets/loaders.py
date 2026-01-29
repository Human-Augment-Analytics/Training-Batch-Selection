"""
Dataset loaders for RL alignment experiments.

Includes:
- Preference datasets (for DPO/RLHF)
- SFT datasets (for supervised fine-tuning baseline)
- Custom batch samplers with selection support
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Optional, Tuple, Any
import random


class PreferenceDataset(Dataset):
    """
    Dataset for preference-based training (DPO, RLHF).

    Each sample contains:
    - prompt: The input prompt
    - chosen: The preferred response
    - rejected: The non-preferred response

    Common datasets: Anthropic-HH, SHP, UltraFeedback
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            data: List of dicts with 'prompt', 'chosen', 'rejected' keys
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Tokenize prompt + chosen
        chosen_text = prompt + chosen
        chosen_enc = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize prompt + rejected
        rejected_text = prompt + rejected
        rejected_enc = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize prompt only (for masking)
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "prompt_length": prompt_len,
            "index": idx,
        }


class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning.

    Each sample contains:
    - prompt: Input prompt
    - response: Target response
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        text = item["prompt"] + item["response"]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Create labels (same as input_ids for causal LM)
        labels = enc["input_ids"].clone().squeeze(0)

        # Mask prompt tokens in labels (only compute loss on response)
        prompt_enc = self.tokenizer(item["prompt"], return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels[:prompt_len] = -100  # Ignore in loss

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
            "index": idx,
        }


class RLBatchSampler(Sampler):
    """
    Batch sampler that supports selective sampling based on scores.

    Maintains per-sample scores and selects high-scoring samples
    with higher probability.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        selection_ratio: float = 1.0,
        temperature: float = 1.0,
        shuffle: bool = True,
    ):
        """
        Args:
            dataset_size: Total number of samples
            batch_size: Samples per batch
            selection_ratio: Fraction of dataset to consider each epoch
            temperature: Sampling temperature (higher = more uniform)
            shuffle: Whether to shuffle
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.selection_ratio = selection_ratio
        self.temperature = temperature
        self.shuffle = shuffle

        # Per-sample scores (default uniform)
        self.scores = torch.ones(dataset_size)

    def update_scores(self, indices: torch.Tensor, scores: torch.Tensor) -> None:
        """Update scores for specific samples."""
        self.scores[indices] = scores

    def __iter__(self):
        # Compute sampling probabilities from scores
        if self.temperature > 0:
            probs = torch.softmax(self.scores / self.temperature, dim=0)
        else:
            probs = torch.ones(self.dataset_size) / self.dataset_size

        # Number of samples to draw
        n_samples = int(self.dataset_size * self.selection_ratio)
        n_samples = max(self.batch_size, n_samples)

        # Sample without replacement
        indices = torch.multinomial(probs, n_samples, replacement=False)

        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size].tolist()

    def __len__(self) -> int:
        n_samples = int(self.dataset_size * self.selection_ratio)
        return (n_samples + self.batch_size - 1) // self.batch_size


def load_preference_dataset(
    dataset_name: str = "anthropic_hh",
    tokenizer=None,
    max_samples: Optional[int] = None,
    max_length: int = 512,
) -> PreferenceDataset:
    """
    Load a preference dataset.

    Supported datasets:
    - anthropic_hh: Anthropic Helpful-Harmless
    - shp: Stanford Human Preferences
    - synthetic: Synthetic data for testing
    """
    if dataset_name == "synthetic":
        # Generate synthetic preference data for testing
        data = _generate_synthetic_preferences(max_samples or 1000)
    elif dataset_name == "anthropic_hh":
        data = _load_anthropic_hh(max_samples)
    elif dataset_name == "shp":
        data = _load_shp(max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return PreferenceDataset(data, tokenizer, max_length)


def load_sft_dataset(
    dataset_name: str = "alpaca",
    tokenizer=None,
    max_samples: Optional[int] = None,
    max_length: int = 512,
) -> SFTDataset:
    """Load an SFT dataset."""
    if dataset_name == "synthetic":
        data = _generate_synthetic_sft(max_samples or 1000)
    elif dataset_name == "alpaca":
        data = _load_alpaca(max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return SFTDataset(data, tokenizer, max_length)


def _generate_synthetic_preferences(n_samples: int) -> List[Dict[str, str]]:
    """Generate synthetic preference data for testing."""
    data = []
    prompts = [
        "Explain how to make a sandwich.",
        "What is the capital of France?",
        "How do I learn Python?",
        "Tell me about machine learning.",
        "What's the weather like today?",
    ]

    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]
        data.append({
            "prompt": f"Human: {prompt}\n\nAssistant: ",
            "chosen": f"Here's a helpful response to your question about {prompt.lower()[:20]}... This is response {i}.",
            "rejected": f"I don't want to answer that. Response {i}.",
        })

    return data


def _generate_synthetic_sft(n_samples: int) -> List[Dict[str, str]]:
    """Generate synthetic SFT data for testing."""
    data = []
    for i in range(n_samples):
        data.append({
            "prompt": f"Question {i}: What is {i} + {i}?\n\nAnswer: ",
            "response": f"The answer is {2*i}.",
        })
    return data


def _load_anthropic_hh(max_samples: Optional[int]) -> List[Dict[str, str]]:
    """Load Anthropic HH dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Anthropic/hh-rlhf", split="train")

        data = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            # Parse the conversation format
            chosen = item["chosen"]
            rejected = item["rejected"]

            # Extract prompt (common prefix)
            # HH format: "Human: ... Assistant: ..."
            prompt_end = chosen.rfind("Assistant:")
            if prompt_end == -1:
                continue

            prompt = chosen[:prompt_end + len("Assistant: ")]
            chosen_response = chosen[prompt_end + len("Assistant: "):]
            rejected_response = rejected[prompt_end + len("Assistant: "):]

            data.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
            })

        return data
    except Exception as e:
        print(f"Could not load Anthropic HH: {e}")
        return _generate_synthetic_preferences(max_samples or 1000)


def _load_shp(max_samples: Optional[int]) -> List[Dict[str, str]]:
    """Load Stanford Human Preferences dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/SHP", split="train")

        data = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            # SHP has history, human_ref_A, human_ref_B, labels
            prompt = item.get("history", "")
            if item.get("labels", 0) == 1:
                chosen = item.get("human_ref_A", "")
                rejected = item.get("human_ref_B", "")
            else:
                chosen = item.get("human_ref_B", "")
                rejected = item.get("human_ref_A", "")

            data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

        return data
    except Exception as e:
        print(f"Could not load SHP: {e}")
        return _generate_synthetic_preferences(max_samples or 1000)


def _load_alpaca(max_samples: Optional[int]) -> List[Dict[str, str]]:
    """Load Alpaca dataset for SFT."""
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")

        data = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            if input_text:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\n\nResponse: "
            else:
                prompt = f"Instruction: {instruction}\n\nResponse: "

            data.append({
                "prompt": prompt,
                "response": output,
            })

        return data
    except Exception as e:
        print(f"Could not load Alpaca: {e}")
        return _generate_synthetic_sft(max_samples or 1000)
