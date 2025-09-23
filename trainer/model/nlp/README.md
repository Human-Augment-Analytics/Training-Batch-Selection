# NLP Model

Minimal decoder-only Transformer for pretraining tasks.

- Entry: `trainer/model/nlp/model.py` (`TinyLLM`)
- Decoder: `trainer/model/nlp/layers/decoder.py`

Import example:
```python
from trainer.model.nlp.model import TinyLLM
```

Run pretraining via the pipeline in `trainer/pipelines/pretrain.py`.

# Language Models