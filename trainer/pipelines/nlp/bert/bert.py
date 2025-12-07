import os
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

from trainer.constants_nlp import (
    NLP_OUTPUT_DIR, MODEL_NAME, NUM_LABELS, MAX_LENGTH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, LOSS_THRESHOLD,
    USE_SUBSET, TRAIN_SUBSET_SIZE, TEST_SUBSET_SIZE
)
from trainer.dataloader.nlp_dataloader import IMDBDataset
from trainer.batching.nlp_batching import random_batch, loss_based_batch


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============ Training Function ============

def train_model(model, train_dataset, test_dataset, epochs, batch_size,
                batch_sampler_fn, strategy_name, seed=None):
    """Train BERT model with specified batch sampling strategy."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Calculate total training steps for scheduler
    n_samples = len(train_dataset)
    steps_per_epoch = n_samples // batch_size
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Track per-sample loss
    per_sample_loss = np.zeros(len(train_dataset))

    # Metrics storage
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    samples_per_epoch = []

    for epoch in range(epochs):
        model.train()
        correct, n_samples_epoch, running_loss = 0, 0, 0

        # Get batch iterator
        if 'loss_history' in batch_sampler_fn.__code__.co_varnames:
            batch_iter = batch_sampler_fn(train_dataset, batch_size,
                                         loss_history=per_sample_loss,
                                         threshold=LOSS_THRESHOLD)
        else:
            batch_iter = batch_sampler_fn(train_dataset, batch_size)

        # Training loop
        for batch_indices in batch_iter:
            # Prepare batch
            batch = [train_dataset[i] for i in batch_indices]
            input_ids = torch.stack([item['input_ids'] for item in batch]).to(DEVICE)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(DEVICE)
            labels = torch.stack([item['label'] for item in batch]).to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Compute loss per sample
            logits = outputs.logits
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            losses_per_sample = loss_fn(logits, labels)
            loss = losses_per_sample.mean()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update metrics
            running_loss += loss.item() * len(batch_indices)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            n_samples_epoch += len(batch_indices)

            # Update per-sample loss history
            if 'loss_history' in batch_sampler_fn.__code__.co_varnames:
                for k, idx in enumerate(batch_indices):
                    per_sample_loss[idx] = losses_per_sample[k].item()

        # Epoch metrics
        train_acc = correct / n_samples_epoch if n_samples_epoch > 0 else 0
        train_loss = running_loss / n_samples_epoch if n_samples_epoch > 0 else 0
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        samples_per_epoch.append(n_samples_epoch)

        # Evaluation
        test_acc, test_loss = evaluate(model, test_dataset, batch_size)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        print(f"[{strategy_name}] Epoch {epoch+1}/{epochs}: "
              f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
              f"train_loss={train_loss:.4f}, samples={n_samples_epoch}")

    return {
        'train_acc': train_accs,
        'test_acc': test_accs,
        'train_loss': train_losses,
        'test_loss': test_losses,
        'samples_per_epoch': samples_per_epoch
    }


# ============ Evaluation Function ============

def evaluate(model, dataset, batch_size):
    """Evaluate model on dataset."""
    model.eval()
    correct, total, total_loss = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()

    indices = list(range(len(dataset)))

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = [dataset[i] for i in batch_indices]

            input_ids = torch.stack([item['input_ids'] for item in batch]).to(DEVICE)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(DEVICE)
            labels = torch.stack([item['label'] for item in batch]).to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            logits = outputs.logits
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * len(batch_indices)
            correct += (predictions == labels).sum().item()
            total += len(batch_indices)

    return correct / total, total_loss / total


# ============ MAIN ============

if __name__ == '__main__':
    # Create output directory
    os.makedirs(NLP_OUTPUT_DIR, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_dataset = IMDBDataset(
        split='train',
        max_length=MAX_LENGTH,
        subset_size=TRAIN_SUBSET_SIZE if USE_SUBSET else None
    )
    test_dataset = IMDBDataset(
        split='test',
        max_length=MAX_LENGTH,
        subset_size=TEST_SUBSET_SIZE if USE_SUBSET else None
    )
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    # ============ Strategy 1: Normal (Baseline) ============
    print("\n" + "="*60)
    print("STRATEGY 1: NORMAL (BASELINE)")
    print("="*60)

    model_normal = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    start_time = time.time()
    results_normal = train_model(
        model_normal, train_dataset, test_dataset,
        EPOCHS, BATCH_SIZE,
        random_batch.batch_sampler, "Normal",
        seed=42
    )
    time_normal = time.time() - start_time
    print(f"\nNormal strategy completed in {time_normal:.2f}s")
    print(f"Final test accuracy: {results_normal['test_acc'][-1]:.4f}")

    # ============ Strategy 2: Loss-Based ============
    print("\n" + "="*60)
    print("STRATEGY 2: LOSS-BASED BATCH SELECTION")
    print("="*60)

    model_loss_based = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    start_time = time.time()
    results_loss_based = train_model(
        model_loss_based, train_dataset, test_dataset,
        EPOCHS, BATCH_SIZE,
        loss_based_batch.batch_sampler, "Loss-Based",
        seed=42
    )
    time_loss_based = time.time() - start_time
    print(f"\nLoss-based strategy completed in {time_loss_based:.2f}s")
    print(f"Final test accuracy: {results_loss_based['test_acc'][-1]:.4f}")

    # ============ Compare Results (4-panel plot) ============
    print("\nCreating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_range = np.arange(1, EPOCHS + 1)

    # Test Accuracy
    axes[0, 0].plot(epochs_range, results_normal['test_acc'], 'o-', label='Normal', linewidth=2)
    axes[0, 0].plot(epochs_range, results_loss_based['test_acc'], 's-', label='Loss-Based', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Test Accuracy Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Train Loss
    axes[0, 1].plot(epochs_range, results_normal['train_loss'], 'o-', label='Normal', linewidth=2)
    axes[0, 1].plot(epochs_range, results_loss_based['train_loss'], 's-', label='Loss-Based', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Train Loss')
    axes[0, 1].set_title('Train Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Train Accuracy
    axes[1, 0].plot(epochs_range, results_normal['train_acc'], 'o-', label='Normal', linewidth=2)
    axes[1, 0].plot(epochs_range, results_loss_based['train_acc'], 's-', label='Loss-Based', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train Accuracy')
    axes[1, 0].set_title('Train Accuracy Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Samples per Epoch
    axes[1, 1].plot(epochs_range, results_normal['samples_per_epoch'], 'o-', label='Normal', linewidth=2)
    axes[1, 1].plot(epochs_range, results_loss_based['samples_per_epoch'], 's-', label='Loss-Based', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Samples per Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(NLP_OUTPUT_DIR, 'comparison_4panel.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nNormal Strategy:")
    print(f"  Final Test Accuracy: {results_normal['test_acc'][-1]:.4f}")
    print(f"  Training Time: {time_normal:.2f}s")
    print(f"  Samples per epoch: {results_normal['samples_per_epoch']}")

    print(f"\nLoss-Based Strategy:")
    print(f"  Final Test Accuracy: {results_loss_based['test_acc'][-1]:.4f}")
    print(f"  Training Time: {time_loss_based:.2f}s")
    print(f"  Samples per epoch: {results_loss_based['samples_per_epoch']}")

    print(f"\nKey Observations:")
    print(f"  - Loss-based filtering reduces samples in later epochs")
    print(f"  - Potential speedup: {(time_normal/time_loss_based):.2f}x")
    print(f"  - Accuracy difference: {(results_loss_based['test_acc'][-1] - results_normal['test_acc'][-1]):.4f}")

    # ============ Training Time & Compute Usage Analysis (8-panel plot) ============
    print("\nCreating efficiency analysis...")

    # Calculate compute metrics
    total_samples_normal = sum(results_normal['samples_per_epoch'])
    total_samples_loss_based = sum(results_loss_based['samples_per_epoch'])
    time_saved = time_normal - time_loss_based
    time_saved_percent = (time_saved / time_normal) * 100
    compute_saved = total_samples_normal - total_samples_loss_based
    compute_saved_percent = (compute_saved / total_samples_normal) * 100

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Total Training Time Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    strategies = ['Normal', 'Loss-Based']
    times = [time_normal, time_loss_based]
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(strategies, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Training Time', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}s',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 2. Time Savings
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.6, f'{time_saved:.1f}s',
             ha='center', va='center', fontsize=48, fontweight='bold', color='#27ae60')
    ax2.text(0.5, 0.35, f'({time_saved_percent:.1f}% faster)',
             ha='center', va='center', fontsize=20, color='#27ae60')
    ax2.text(0.5, 0.15, 'Time Saved',
             ha='center', va='center', fontsize=16, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Time Savings', fontsize=14, fontweight='bold', pad=20)

    # 3. Speedup Factor
    ax3 = fig.add_subplot(gs[0, 2])
    speedup = time_normal / time_loss_based if time_loss_based > 0 else 1.0
    ax3.text(0.5, 0.6, f'{speedup:.2f}x',
             ha='center', va='center', fontsize=48, fontweight='bold', color='#8e44ad')
    ax3.text(0.5, 0.35, 'Speedup Factor',
             ha='center', va='center', fontsize=16, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Training Speedup', fontsize=14, fontweight='bold', pad=20)

    # 4. Total Compute Usage
    ax4 = fig.add_subplot(gs[1, 0])
    total_samples = [total_samples_normal, total_samples_loss_based]
    bars = ax4.bar(strategies, total_samples, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Total Samples Processed', fontsize=12, fontweight='bold')
    ax4.set_title('Total Compute Usage', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 5. Compute Savings
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.text(0.5, 0.6, f'{int(compute_saved):,}',
             ha='center', va='center', fontsize=40, fontweight='bold', color='#27ae60')
    ax5.text(0.5, 0.35, f'({compute_saved_percent:.1f}% reduction)',
             ha='center', va='center', fontsize=20, color='#27ae60')
    ax5.text(0.5, 0.15, 'Samples Saved',
             ha='center', va='center', fontsize=16, fontweight='bold')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('Compute Savings', fontsize=14, fontweight='bold', pad=20)

    # 6. Cumulative Samples Over Epochs
    ax6 = fig.add_subplot(gs[1, 2])
    cumsum_normal = np.cumsum(results_normal['samples_per_epoch'])
    cumsum_loss_based = np.cumsum(results_loss_based['samples_per_epoch'])
    ax6.plot(epochs_range, cumsum_normal, 'o-', label='Normal',
             linewidth=3, markersize=8, color='#3498db')
    ax6.plot(epochs_range, cumsum_loss_based, 's-', label='Loss-Based',
             linewidth=3, markersize=8, color='#e74c3c')
    ax6.fill_between(epochs_range, cumsum_normal, cumsum_loss_based,
                      alpha=0.3, color='#27ae60', label='Compute Saved')
    ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Cumulative Samples', fontsize=12, fontweight='bold')
    ax6.set_title('Cumulative Compute Usage', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # 7. Samples per Epoch Breakdown
    ax7 = fig.add_subplot(gs[2, :2])
    x = np.arange(EPOCHS)
    width = 0.35
    bars1 = ax7.bar(x - width/2, results_normal['samples_per_epoch'], width,
                    label='Normal', color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax7.bar(x + width/2, results_loss_based['samples_per_epoch'], width,
                    label='Loss-Based', color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax7.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Samples Processed', fontsize=12, fontweight='bold')
    ax7.set_title('Samples Processed per Epoch', fontsize=14, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels([f'Epoch {i+1}' for i in range(EPOCHS)])
    ax7.legend(fontsize=11)
    ax7.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 8. Efficiency Metrics Summary Table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('tight')
    ax8.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Time Saved', f'{time_saved:.1f}s'],
        ['Speedup', f'{speedup:.2f}x'],
        ['Samples Saved', f'{int(compute_saved):,}'],
        ['Compute Reduction', f'{compute_saved_percent:.1f}%'],
        ['Final Accuracy Diff', f'{(results_loss_based["test_acc"][-1] - results_normal["test_acc"][-1]):.4f}']
    ]
    table = ax8.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    for i in range(2):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
    ax8.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('Training Time & Compute Usage Analysis: Normal vs Loss-Based Batch Selection',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(NLP_OUTPUT_DIR, 'training_efficiency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print detailed summary
    print("\n" + "="*70)
    print("TRAINING EFFICIENCY ANALYSIS")
    print("="*70)
    print(f"\n{'METRIC':<35} {'NORMAL':<15} {'LOSS-BASED':<15} {'IMPROVEMENT':<15}")
    print("-"*70)
    print(f"{'Total Training Time (s)':<35} {time_normal:<15.2f} {time_loss_based:<15.2f} {time_saved_percent:>14.1f}%")
    print(f"{'Total Samples Processed':<35} {total_samples_normal:<15,} {total_samples_loss_based:<15,} {compute_saved_percent:>14.1f}%")
    print(f"{'Final Test Accuracy':<35} {results_normal['test_acc'][-1]:<15.4f} {results_loss_based['test_acc'][-1]:<15.4f} {(results_loss_based['test_acc'][-1] - results_normal['test_acc'][-1]):>+15.4f}")
    print(f"{'Speedup Factor':<35} {'1.00x':<15} {f'{speedup:.2f}x':<15} {'':<15}")

    print("\n" + "="*70)
    print("EPOCH-BY-EPOCH BREAKDOWN")
    print("="*70)
    print(f"\n{'EPOCH':<10} {'NORMAL':<20} {'LOSS-BASED':<20} {'SAMPLES SAVED':<20}")
    print("-"*70)
    for i in range(EPOCHS):
        samples_saved_epoch = results_normal['samples_per_epoch'][i] - results_loss_based['samples_per_epoch'][i]
        print(f"{i+1:<10} {results_normal['samples_per_epoch'][i]:<20,} "
              f"{results_loss_based['samples_per_epoch'][i]:<20,} "
              f"{samples_saved_epoch:<20,} ({samples_saved_epoch/results_normal['samples_per_epoch'][i]*100:.1f}%)")
    print("\n" + "="*70)

    print(f"\nResults saved to: {NLP_OUTPUT_DIR}")
    print("  - comparison_4panel.png")
    print("  - training_efficiency_analysis.png")
