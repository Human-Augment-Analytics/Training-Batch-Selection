from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    seq_length: int
    dropout: float = 0.1
    device: str = "cpu"


@dataclass
class TrainConfig:
    batch_size: int = 4
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-2


@dataclass
class DataConfig:
    tokenized_root: str
    seq_length: int
    eos_token: int | None = None
    max_samples: int = 10_000_000
    tokenize_stride: int = 1


