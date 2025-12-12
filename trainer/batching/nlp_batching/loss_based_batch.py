import numpy as np

def batch_sampler(dataset, batch_size, loss_history=None, threshold=0.5, **kwargs):
    """
    Loss-based batch sampling for NLP tasks.

    First epoch: uses all examples.
    Subsequent epochs: only examples with loss > threshold.
    Always ensures at least batch_size examples are selected to prevent empty epochs.

    Args:
        dataset: The dataset to sample from
        batch_size: Number of samples per batch
        loss_history: Array of per-sample losses from previous epoch
        threshold: Loss threshold for filtering (default: 0.5)
        **kwargs: Additional arguments

    Yields:
        Lists of indices for each batch
    """
    if loss_history is None or loss_history.sum() == 0:
        # First epoch: use all examples
        indices = np.arange(len(dataset))
    else:
        # Filter to examples with loss above threshold
        indices = np.where(loss_history > threshold)[0]

        # IMPORTANT: Always ensure minimum number of samples to prevent empty epochs
        min_samples = batch_size
        if len(indices) < min_samples:
            # Not enough samples - pick top-k by loss instead
            top_k = min(min_samples, len(dataset))
            indices = np.argsort(loss_history)[-top_k:]  # Top-k highest loss samples
            print(f"  Filtering: Only {len(np.where(loss_history > threshold)[0])} examples above threshold, using top-{top_k} by loss instead")
        else:
            print(f"  Filtering: {len(indices)}/{len(dataset)} examples above threshold")

    np.random.shuffle(indices)

    # Yield batches, including any remaining samples in a final smaller batch
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield batch_indices.tolist()
