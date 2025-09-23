import torch
import torch.nn as nn
from trainer.model.nlp.layers.decoder import TransformerDecoder


class TinyLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        seq_length: int,
        dropout: float = 0.1,
        device: str = 'cpu',
    ) -> None:
        super().__init__()
        self.model = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_length=seq_length,
            dropout=dropout,
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
