#!/usr/bin/env python
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from tqdm import tqdm


# Import our custom modules
from trainer.dataloader.text_dataloader import TokenizedDataset
from trainer.model.nlp import TinyLLM
from trainer.pipelines.config import ModelConfig, TrainConfig, DataConfig


#########################
# Training Loop Helpers #
#########################
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

####################
# Main Entry Point #
####################
def main():
    parser = argparse.ArgumentParser(description="Pretraining a Transformer-based LLM")
    # Data settings
    parser.add_argument('--tokenized_root', type=str, default=os.path.join(os.getcwd(), "trainer", "data", "tokenized", "tiiuae_falcon-refinedweb"), help='Root directory containing tokenized data')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--eos_token', type=int, default=None, help='EOS token id')
    parser.add_argument('--max_samples', type=int, default=10_000_000, help='Dataset cap')
    parser.add_argument('--tokenize_stride', type=int, default=1, help='Stride when slicing tokens')
    # Train settings
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2, help='Epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for LR')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Clip global grad norm')
    # Model settings
    parser.add_argument('--vocab_size', type=int, default=128256, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=2048, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0
    pin_memory = False
    print(f"Using device: {device}")

    # Create dataset and dataloader.
    print('====== Loading tokenized data from: ', args.tokenized_root)
    dataset = TokenizedDataset(
        args.tokenized_root,
        args.seq_length,
        eos_token=args.eos_token,
        max_samples=args.max_samples,
        stride=args.tokenize_stride,
    )
    print("Dataset length:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    import time

    start = time.time()
    sample = dataset[0]
    end = time.time()
    print("Time to retrieve one sample:", end - start)

    # Instantiate the model.
    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_length=args.seq_length,
        dropout=args.dropout,
        device=str(device),
    )
    model = TinyLLM(**model_cfg.__dict__)

    # Optional: DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(dataloader) * args.epochs
    def lr_lambda(current_step: int):
        if args.warmup_steps > 0 and current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}")

    torch.save(model.state_dict(), "tinyllm.pt")
    print("Training complete and model saved as tinyllm.pt")

if __name__ == "__main__":
    main()
