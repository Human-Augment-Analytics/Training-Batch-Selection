#####################################
# Complete Transformer Decoder #
#####################################

import torch.nn as nn
from trainer.model.nlp.layers.repeating.transformer_block import TransformerBlock
from trainer.model.nlp.layers.pos_encoding import PositionalEncoding

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, seq_length, dropout=0.1):
        """
        vocab_size: number of tokens in the vocabulary
        d_model: embedding dimension
        num_layers: number of transformer blocks to stack
        num_heads: number of attention heads
        seq_length: maximum sequence length (for positional encoding)
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len=seq_length)
        # Create a stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=4 * d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Final linear layer to map hidden states to vocabulary logits
        self.linear = nn.Linear(d_model, vocab_size)
        # Note: Softmax will be applied with the loss function (e.g., CrossEntropyLoss)

    def forward(self, x):
        """
        x: Tensor of token IDs of shape (batch_size, seq_length)
        Returns:
          logits: Tensor of shape (batch_size, seq_length, vocab_size)
        """
        x = self.token_embedding(x)  # (batch_size, seq_length, d_model)
        x = self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.linear(x)
        return logits
