"""
Language model utilities for RL alignment experiments.

Uses small, well-documented models:
- GPT-2 (small/medium) for policy
- DistilGPT-2 for faster experiments
- Separate reward model or preference model
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from copy import deepcopy


def load_causal_lm(
    model_name: str = "gpt2",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, "PreTrainedTokenizer"]:
    """
    Load a causal language model for policy training.

    Args:
        model_name: HuggingFace model name (gpt2, distilgpt2, etc.)
        device: Device to load model on
        dtype: Model dtype (float32 or float16)

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 doesn't have pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    return model, tokenizer


def load_reward_model(
    model_name: str = "gpt2",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Load or create a reward model.

    For simplicity, we use a language model with a value head.
    In practice, this would be trained on human preferences.
    """
    from transformers import AutoModelForSequenceClassification, AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1  # Single reward value

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    ).to(device)

    return model


def create_reference_model(
    policy_model: nn.Module,
    freeze: bool = True,
) -> nn.Module:
    """
    Create a reference model (frozen copy of policy).

    Used in PPO for KL penalty and in DPO for reference probabilities.
    """
    ref_model = deepcopy(policy_model)

    if freeze:
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

    return ref_model


class ValueHead(nn.Module):
    """Value head for PPO critic."""

    def __init__(self, hidden_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.out = nn.Linear(hidden_size, 1, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use last token's hidden state
        x = hidden_states[:, -1, :]
        x = torch.tanh(self.dense(x))
        return self.out(x).squeeze(-1)


class PolicyWithValueHead(nn.Module):
    """
    Language model with value head for PPO.

    Combines policy (logits) and value (scalar) outputs.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.base_model = base_model

        # Get hidden size from model config
        if hidden_size is None:
            if hasattr(base_model.config, 'hidden_size'):
                hidden_size = base_model.config.hidden_size
            elif hasattr(base_model.config, 'n_embd'):
                hidden_size = base_model.config.n_embd
            else:
                hidden_size = 768  # Default

        self.value_head = ValueHead(hidden_size, dtype=next(base_model.parameters()).dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        values = self.value_head(hidden_states)

        return logits, values

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)
