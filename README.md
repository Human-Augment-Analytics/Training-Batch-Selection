# Training-Batch-Selection
Boilerplate for preparing data and pretraining a minimal decoder-only Transformer.

---

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Preparation
Use the loader to download, split, and tokenize datasets from Hugging Face.

```bash
# Example: Falcon-RefinedWeb (content column)
python -m trainer.data.loader --max_rows 600000 --split_examples 10000 \
  --dataset_name tiiuae/falcon-refinedweb --dataset_col content
```

Tokenized files are stored under `trainer/data/tokenized/<dataset_name_sanitized>/`.

---

## Pretraining
```bash
python run_pretraining.py
```
Model weights are saved to `tinyllm.pt`.

---

## Package layout
- `trainer/data/loader.py`: download/split/tokenize utilities
- `trainer/model/nlp/`: model code
- `trainer/pipelines/pretrain.py`: training loop
- `trainer/pipelines/config.py`: config dataclasses
