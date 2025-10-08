# Models

This package provides a minimal decoder-only Transformer for language model pretraining.

### Modules
- `trainer/model/nlp/model.py`: exports `TinyLLM` wrapper around the decoder.
- `trainer/model/nlp/layers/decoder.py`: `TransformerDecoder` assembling blocks and output head.
- `trainer/model/nlp/layers/repeating/transformer_block.py`: single decoder block.
- `trainer/model/nlp/layers/repeating/attention/masked_mha.py`: causal MHA with rotary embeddings.
- `trainer/model/nlp/layers/repeating/mlp/feed_forward.py`: MLP with ReLU and dropout.
- `trainer/model/nlp/layers/pos_encoding.py`: sinusoidal pos enc + rotary helpers.

### Usage
```python
from trainer.model.nlp.model import TinyLLM

model = TinyLLM(
    vocab_size=128256,
    d_model=1024,
    num_layers=12,
    num_heads=16,
    seq_length=2048,
    dropout=0.1,
    device='cuda'  # or 'cpu'
)
logits = model(input_ids)  # (batch, seq, vocab)
```

For end-to-end training, see `trainer/pipelines/pretrain.py`.