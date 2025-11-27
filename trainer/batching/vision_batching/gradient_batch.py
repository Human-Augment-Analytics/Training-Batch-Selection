import numpy as np
import torch
import torch.nn.functional as F
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


def compute_gradient_vectors(model, dataset, indices, loss_fn, device='cpu'):
    """
    Compute full gradient vectors for a set of samples (used by GREAT).

    Args:
        model: The neural network model
        dataset: The training dataset
        indices: Indices of samples to compute gradients for
        loss_fn: Loss function (should use reduction='none')
        device: Device to run computation on

    Returns:
        Tensor of gradient vectors (n_samples x gradient_dim)
    """
    model.eval()
    gradient_vectors = []

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

        # Collect all gradients into a single vector
        grad_vector = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.data.flatten())

        grad_vector = torch.cat(grad_vector)
        gradient_vectors.append(grad_vector)

    model.train()
    return torch.stack(gradient_vectors)


def get_great_batch(gradient_vectors, batch_size):
    """
    Select batch using GREAT (GREedy Approximation Taylor Selection).

    GREAT uses a greedy approximation of the Taylor series expansion to select
    samples that maximize the expected loss reduction. The key idea is:
    1. Use first-order Taylor approximation: loss change ≈ gradient^T * update
    2. Greedily select samples whose gradients provide maximum orthogonal
       contribution to the already selected set
    3. This reduces redundancy and maximizes information gain

    The algorithm:
    - Start with the sample with highest gradient norm
    - Iteratively add samples whose gradient has maximum orthogonal component
      to the subspace spanned by already selected gradients

    Args:
        gradient_vectors: Tensor of gradient vectors (n_samples x gradient_dim)
        batch_size: Size of batch to select

    Returns:
        Array of selected sample indices
    """
    n_samples = len(gradient_vectors)
    batch_size = min(batch_size, n_samples)
    device = gradient_vectors.device

    # Compute gradient norms
    grad_norms = torch.norm(gradient_vectors, dim=1)

    # Start with sample having highest gradient norm
    selected_indices = [torch.argmax(grad_norms).item()]
    remaining_mask = torch.ones(n_samples, dtype=torch.bool, device=device)
    remaining_mask[selected_indices[0]] = False

    # Keep track of the orthonormal basis of selected gradients
    # Start with first gradient normalized
    first_grad = gradient_vectors[selected_indices[0]].clone()
    basis = [first_grad / torch.norm(first_grad)]

    # Greedily select remaining samples
    for _ in range(batch_size - 1):
        if remaining_mask.sum() == 0:
            break

        remaining_grads = gradient_vectors[remaining_mask]  # (n_remaining, dim)

        # For each remaining gradient, compute its projection onto the
        # subspace spanned by selected gradients, then get orthogonal component
        max_orthogonal_norm = -1
        best_idx = None
        best_orthogonal_grad = None

        for i, grad in enumerate(remaining_grads):
            # Project onto existing basis and subtract to get orthogonal component
            orthogonal_grad = grad.clone()
            for basis_vec in basis:
                projection = torch.dot(grad, basis_vec)
                orthogonal_grad -= projection * basis_vec

            # Compute norm of orthogonal component
            orthogonal_norm = torch.norm(orthogonal_grad).item()

            if orthogonal_norm > max_orthogonal_norm:
                max_orthogonal_norm = orthogonal_norm
                # Get actual index in original array
                remaining_indices = torch.where(remaining_mask)[0]
                best_idx = remaining_indices[i].item()
                best_orthogonal_grad = orthogonal_grad

        if best_idx is None:
            break

        # Add to selected set
        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False

        # Add normalized orthogonal component to basis
        if max_orthogonal_norm > 1e-6:  # Avoid numerical issues
            basis.append(best_orthogonal_grad / max_orthogonal_norm)

    return np.array(selected_indices)


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


def great_batch_sampler(dataset, batch_size, model=None, loss_fn=None, device='cpu',
                        gradient_vectors=None):
    """
    GREAT batch sampler that yields batches based on greedy Taylor approximation.

    GREAT (GREedy Approximation Taylor Selection) selects batches by greedily
    maximizing the expected loss reduction using first-order Taylor approximation.
    It selects samples whose gradients are maximally informative and non-redundant.

    Args:
        dataset: Training dataset
        batch_size: Batch size
        model: Neural network model (required for gradient computation)
        loss_fn: Loss function (required for gradient computation)
        device: Device to run on
        gradient_vectors: Pre-computed gradient vectors (optional)

    Yields:
        Batch indices
    """
    n = len(dataset)
    n_batches = n // batch_size

    # Compute gradient vectors if not provided
    if gradient_vectors is None and model is not None and loss_fn is not None:
        print("Computing gradient vectors for GREAT batch selection...")
        all_indices = np.arange(n)
        gradient_vectors = compute_gradient_vectors(model, dataset, all_indices, loss_fn, device)
    elif gradient_vectors is None:
        # Fallback to random batching if model/loss_fn not provided
        print("Warning: model or loss_fn not provided, falling back to random batching")
        indices = np.arange(n)
        np.random.shuffle(indices)
        for i in range(n_batches):
            yield indices[i*batch_size:(i+1)*batch_size]
        return

    # Yield batches based on greedy Taylor approximation
    used_indices = set()
    for _ in range(n_batches):
        # Get available indices (not yet used)
        available_indices = [i for i in range(n) if i not in used_indices]

        if len(available_indices) < batch_size:
            # Reset if we've used most samples
            used_indices = set()
            available_indices = list(range(n))

        available_grads = gradient_vectors[available_indices]

        # Select batch using GREAT algorithm
        batch_indices_in_available = get_great_batch(available_grads, batch_size)
        batch_indices = np.array(available_indices)[batch_indices_in_available]

        # Mark as used
        used_indices.update(batch_indices)

        yield batch_indices
