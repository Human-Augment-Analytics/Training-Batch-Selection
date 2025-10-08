#!/usr/bin/env python
"""
Edit this file to configure and launch pretraining.

Usage:
  python run_pretraining.py

What to edit:
- MODEL_CFG: depth/width, vocab size, sequence length, dropout
- TRAIN_CFG: epochs, batch size, learning rate, warmup steps, grad clip, weight decay
- DATA_CFG: tokenized data root, sequence length, stride, eos token
- OUTPUT_DIR: where to save checkpoints, hparams, metrics and lr curve
"""

import os
import json
from dataclasses import asdict
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from trainer.model.nlp import TinyLLM
from trainer.pipelines.pretraining.legacy.pretrain import TokenizedDataset
from trainer.pipelines.config import ModelConfig


# =====================
# EDIT THESE SECTIONS
# =====================

# Output directory for this run
OUTPUT_DIR = os.path.join('outputs', datetime.now().strftime('%Y%m%d_%H%M%S'))

# Data settings
DATA_CFG = {
    'tokenized_root': 'trainer/data/tokenized/tiiuae_falcon-refinedweb',
    'seq_length': 1024,
    'eos_token': None,
    'max_samples': 10_000_000,
    'tokenize_stride': 1,
    'batch_size': 8,
}

# Model settings
MODEL_CFG = ModelConfig(
    vocab_size=128256,
    d_model=1024,
    num_layers=12,
    num_heads=16,
    seq_length=1024,
    dropout=0.1,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# Training settings
TRAIN_CFG = {
    'epochs': 2,
    'lr': 3e-4,
    'weight_decay': 1e-2,
    'warmup_steps': 1000,
    'grad_clip_norm': 1.0,
    'save_every': 1,
}

# =====================
# End of editable block
# =====================


def build_scheduler_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(output_dir: str, epoch: int, model, optimizer, scheduler, metrics: dict):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
    }, ckpt_path)
    return ckpt_path


def save_hparams(output_dir: str, model_cfg: ModelConfig):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'hparams.json'), 'w') as f:
        json.dump({
            'data': DATA_CFG,
            'train': TRAIN_CFG,
            'model': asdict(model_cfg),
        }, f, indent=2)


def plot_lr_curve(lrs, output_dir: str, until_step: int):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(lrs)), lrs, label='learning_rate')
    plt.xlabel('Step')
    plt.ylabel('LR')
    plt.title(f'LR Curve (0..{until_step})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_curve.jpg'), dpi=200)
    plt.close()


def main():
    device = torch.device(MODEL_CFG.device)
    print('Using device:', device)

    # Data
    dataset = TokenizedDataset(
        DATA_CFG['tokenized_root'],
        DATA_CFG['seq_length'],
        eos_token=DATA_CFG['eos_token'],
        max_samples=DATA_CFG['max_samples'],
        stride=DATA_CFG['tokenize_stride'],
    )
    dataloader = DataLoader(dataset, batch_size=DATA_CFG['batch_size'], shuffle=True, num_workers=0)

    # Model
    model = TinyLLM(**asdict(MODEL_CFG))
    if torch.cuda.device_count() > 1:
        print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG['lr'], weight_decay=TRAIN_CFG['weight_decay'])
    total_steps = len(dataloader) * TRAIN_CFG['epochs']
    scheduler = build_scheduler_with_warmup(optimizer, total_steps=total_steps, warmup_steps=TRAIN_CFG['warmup_steps'])
    criterion = nn.CrossEntropyLoss()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_hparams(OUTPUT_DIR, MODEL_CFG)

    global_step = 0
    lr_values = []
    history = []

    for epoch in range(1, TRAIN_CFG['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            if TRAIN_CFG['grad_clip_norm'] and TRAIN_CFG['grad_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG['grad_clip_norm'])
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1
            lr_values.append(optimizer.param_groups[0]['lr'])

        avg_loss = total_loss / max(1, len(dataloader))
        metrics = {'epoch': epoch, 'avg_loss': avg_loss, 'global_step': global_step}
        history.append(metrics)
        print(f"Epoch {epoch}/{TRAIN_CFG['epochs']} - loss={avg_loss:.4f}")

        if epoch % TRAIN_CFG['save_every'] == 0:
            ckpt_path = save_checkpoint(OUTPUT_DIR, epoch, model, optimizer, scheduler, metrics)
            print('Saved checkpoint to', ckpt_path)
            plot_lr_curve(lr_values, OUTPUT_DIR, until_step=global_step)

    torch.save(model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(), os.path.join(OUTPUT_DIR, 'tinyllm_final.pt'))
    with open(os.path.join(OUTPUT_DIR, 'metrics_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print('Training complete. Artifacts saved under', OUTPUT_DIR)


if __name__ == '__main__':
    main()


