#!/usr/bin/env python
import os
import json
import argparse
from dataclasses import asdict
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from trainer.pipelines.pretraining.legacy.pretrain import TokenizedDataset, train
from trainer.model.nlp import TinyLLM
from trainer.pipelines.config import ModelConfig


def build_scheduler_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    # Linear warmup to max lr, then cosine annealing to 0
    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay after warmup
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


def save_hparams(output_dir: str, args: argparse.Namespace, model_cfg: ModelConfig):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'hparams.json'), 'w') as f:
        json.dump({
            'args': vars(args),
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
    parser = argparse.ArgumentParser(description='Helper launcher for LLM pretraining')
    # Data
    parser.add_argument('--tokenized_root', type=str, required=True)
    parser.add_argument('--seq_length', type=int, default=1024)
    parser.add_argument('--eos_token', type=int, default=None)
    parser.add_argument('--max_samples', type=int, default=10_000_000)
    parser.add_argument('--tokenize_stride', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    # Model
    parser.add_argument('--vocab_size', type=int, default=128256)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    # Train
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    # IO
    parser.add_argument('--output_dir', type=str, default=os.path.join('outputs', datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Data
    dataset = TokenizedDataset(
        args.tokenized_root,
        args.seq_length,
        eos_token=args.eos_token,
        max_samples=args.max_samples,
        stride=args.tokenize_stride,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Model
    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_length=args.seq_length,
        dropout=args.dropout,
        device=str(device),
    )
    model = TinyLLM(**asdict(model_cfg))

    if torch.cuda.device_count() > 1:
        print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(dataloader) * args.epochs
    scheduler = build_scheduler_with_warmup(optimizer, total_steps=total_steps, warmup_steps=args.warmup_steps)
    criterion = nn.CrossEntropyLoss()

    # Logging setup
    os.makedirs(args.output_dir, exist_ok=True)
    save_hparams(args.output_dir, args, model_cfg)

    # Train loop with gradient clipping and LR tracking
    global_step = 0
    lr_values = []
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1
            lr_values.append(optimizer.param_groups[0]['lr'])

        avg_loss = total_loss / max(1, len(dataloader))
        metrics = {'epoch': epoch, 'avg_loss': avg_loss, 'global_step': global_step}
        history.append(metrics)
        print(f"Epoch {epoch}/{args.epochs} - loss={avg_loss:.4f}")

        if epoch % args.save_every == 0:
            ckpt_path = save_checkpoint(args.output_dir, epoch, model, optimizer, scheduler, metrics)
            print('Saved checkpoint to', ckpt_path)
            # Save LR curve image up to this epoch
            plot_lr_curve(lr_values, args.output_dir, until_step=global_step)

    # Save final artifacts
    torch.save(model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(), os.path.join(args.output_dir, 'tinyllm_final.pt'))
    with open(os.path.join(args.output_dir, 'metrics_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print('Training complete. Artifacts saved under', args.output_dir)


if __name__ == '__main__':
    main()


