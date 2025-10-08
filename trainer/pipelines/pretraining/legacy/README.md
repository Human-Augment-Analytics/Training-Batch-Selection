# Pipelines

Pretraining entrypoint: `trainer/pipelines/pretrain.py`.

### Quick start
```bash
python -m trainer.pipelines.pretrain \
  --tokenized_root trainer/data/tokenized/tiiuae_falcon-refinedweb \
  --seq_length 1024 \
  --batch_size 8 \
  --epochs 1 \
  --lr 3e-4 \
  --vocab_size 128256 \
  --d_model 1024 \
  --num_layers 12 \
  --num_heads 16
```

To prepare data, see `trainer/data/loader.py`.

### One-file runner (easy to edit)
Prefer editing and running the top-level script:

```bash
python run_pretraining.py
```

In `run_pretraining.py`, edit the top section to set:
- DATA_CFG: tokenized_root, seq_length, stride, batch_size
- MODEL_CFG: vocab_size, d_model, num_layers, num_heads, seq_length, dropout
- TRAIN_CFG: epochs, lr, weight_decay, warmup_steps, grad_clip_norm, save_every
- OUTPUT_DIR: output directory for checkpoints and plots