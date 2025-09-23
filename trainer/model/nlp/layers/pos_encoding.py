##############################
# Positional Encoding #
##############################

import torch
import torch.nn as nn
import math

#########################################
# Rotary Positional Embedding Functions #
#########################################
def get_rotary_embedding(seq_len, d_head, device):
    """
    Compute rotary positional embeddings (cosine and sine) for a given sequence length and head dimension.
    Returns cos, sin each of shape (seq_len, d_head).
    """
    assert d_head % 2 == 0, "d_head must be even for rotary embeddings."
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    positions = torch.arange(seq_len, device=device).float()
    sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)  # (seq_len, d_head/2)
    sin = torch.sin(sinusoid_inp)  # (seq_len, d_head/2)
    cos = torch.cos(sinusoid_inp)  # (seq_len, d_head/2)
    sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, d_head)
    cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, d_head)
    return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary positional embedding to tensor x.
    x: Tensor of shape (batch, num_heads, seq_len, d_head)
    cos, sin: Tensors of shape (seq_len, d_head)
    Returns: Tensor of the same shape as x.
    """
    x1, x2 = x.split(x.size(-1)//2, dim=-1)
    x_rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos.unsqueeze(0).unsqueeze(0) + x_rotated * sin.unsqueeze(0).unsqueeze(0)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Implements the sinusoidal positional encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Compute the div term
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
