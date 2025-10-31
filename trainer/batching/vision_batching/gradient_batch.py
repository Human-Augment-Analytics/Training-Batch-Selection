import numpy as np
import torch
from trainer.constants import EXPLORE_FRAC, TOP_K_FRAC

def compute_gradient_norms(model, dataset, indices, loss_fn, device='cpu'):
    """
    Compute gradient norms for a set of samples.

    Args:
        model: The neural network model
        dataset: The training dataset
        indices: Indices of samples to compute gradients for
        loss_fn: Loss function (should use reduction='none')
        device: Device to run computation on

    Returns:
        Array of gradient norms for each sample
    """
    model.eval()
    gradient_norms = []

    for idx in indices:
        # Get single sample
        x, y = dataset[idx]
        x = x.view(1, -1).to(device)
        y = torch.tensor([y]).to(device)

        # Zero gradients
        model.zero_grad()

        # Forward pass
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        # Backward pass
        loss.backward()

        # Compute gradient norm (L2 norm of all gradients)
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = np.sqrt(grad_norm)
        gradient_norms.append(grad_norm)

    model.train()
    return np.array(gradient_norms)


def get_grand_batch(gradient_norms, batch_size, explore_frac=EXPLORE_FRAC, top_k_frac=TOP_K_FRAC):
    """
    Select batch using GraND (Gradient Normed Distance) strategy.

    Combines exploration (random sampling) with exploitation (high gradient norm samples).

    Args:
        gradient_norms: Array of gradient norms for all samples
        batch_size: Size of batch to select
        explore_frac: Fraction of batch to fill with random samples
        top_k_frac: Fraction of dataset to consider as high-gradient candidates

    Returns:
        Array of selected sample indices
    """
    n_explore = int(batch_size * explore_frac)
    n_exploit = batch_size - n_explore
    n_total = len(gradient_norms)

    # Random exploration
    rand_idxs = np.random.choice(n_total, n_explore, replace=False)

    # Exploitation: select from top-k highest gradient norms
    k = int(top_k_frac * n_total)
    exploit_candidates = np.argsort(-gradient_norms)[:k]  # Sort descending, take top k
    exploit_idxs = np.random.choice(
        exploit_candidates,
        min(n_exploit, len(exploit_candidates)),
        replace=False
    )

    # Combine and shuffle
    batch_idxs = np.concatenate([rand_idxs, exploit_idxs])
    np.random.shuffle(batch_idxs)
    return batch_idxs


def batch_sampler(dataset, batch_size, model=None, loss_fn=None, device='cpu',
                  gradient_norms=None):
    """
    GraND batch sampler that yields batches based on gradient norms.

    This function is called by the training loop. It computes gradient norms
    for all samples (expensive operation done once per epoch) and then yields
    batches selected based on those gradient norms.

    Args:
        dataset: Training dataset
        batch_size: Batch size
        model: Neural network model (required for gradient computation)
        loss_fn: Loss function (required for gradient computation)
        device: Device to run on
        gradient_norms: Pre-computed gradient norms (optional, computed if not provided)

    Yields:
        Batch indices
    """
    n = len(dataset)
    n_batches = n // batch_size

    # Compute gradient norms if not provided
    if gradient_norms is None and model is not None and loss_fn is not None:
        print("Computing gradient norms for GraND batch selection...")
        all_indices = np.arange(n)
        gradient_norms = compute_gradient_norms(model, dataset, all_indices, loss_fn, device)
    elif gradient_norms is None:
        # Fallback to random batching if model/loss_fn not provided
        print("Warning: model or loss_fn not provided, falling back to random batching")
        indices = np.arange(n)
        np.random.shuffle(indices)
        for i in range(n_batches):
            yield indices[i*batch_size:(i+1)*batch_size]
        return

    # Yield batches based on gradient norms
    for _ in range(n_batches):
        yield get_grand_batch(gradient_norms, batch_size)
