import numpy as np
import torch

CANDIDATE_POOL_DEFAULT = 500
VAL_SAMPLE_SIZE = 2  # Match GREATS paper val_batchsize=2


def compute_gradient_vectors(model, dataset, indices, loss_fn, device='cpu'):
    """
    Compute full gradient vectors for a set of samples.
    """
    model.eval()
    gradient_vectors = []

    for idx in indices:
        x, y = dataset[idx]
        x = x.view(1, -1).to(device)
        y = torch.tensor([y]).to(device)

        model.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        grad_vector = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.data.flatten())

        grad_vector = torch.cat(grad_vector)
        gradient_vectors.append(grad_vector)

    model.train()
    return torch.stack(gradient_vectors)


def greedy_selection(scores, interaction_matrix, K):
    """
    Select K data points based on the highest scores, dynamically updating scores
    by subtracting interactions with previously selected data points.

    Ported from GREATS (NeurIPS 2024):
    GREATS/less/train/utils_ghost_dot_prod.py

    Parameters:
    - scores: A numpy array of initial TracIN scores for each data point.
    - interaction_matrix: A numpy matrix of pairwise gradient similarity between data points.
    - K: The number of data points to select.

    Returns:
    - selected_indices: Indices of the selected data points.
    """
    scores = scores.copy()
    selected_indices = []

    for _ in range(K):
        idx_max = np.argmax(scores)
        selected_indices.append(idx_max)

        # Reduce scores of similar samples to promote diversity
        scores -= interaction_matrix[idx_max, :]

        # Prevent re-selection
        scores[idx_max] = -np.inf

    return selected_indices


def get_great_batch(train_grads, val_grads, pool_idxs, batch_size, lr=1.0):
    """
    Select batch using GREATS algorithm (NeurIPS 2024).

    Uses TracIN scores (gradient dot product with validation set) for importance
    and a similarity matrix (pairwise gradient dot products) for redundancy-aware
    greedy selection.

    Parameters:
    - train_grads: Gradient vectors for training candidate samples [N, D]
    - val_grads: Gradient vectors for validation samples [V, D]
    - pool_idxs: Original dataset indices for the candidate pool
    - batch_size: Number of samples to select
    - lr: Current learning rate for scaling (matching GREATS paper)
    """
    n_samples = len(train_grads)
    batch_size = min(batch_size, n_samples)

    # TracIN scores: dot product of each train gradient with mean validation gradient
    mean_val_grad = val_grads.mean(dim=0)
    tracin_scores = (train_grads @ mean_val_grad).cpu().numpy()

    # Similarity matrix: pairwise dot products within training gradients
    similarity_matrix = (train_grads @ train_grads.T).cpu().numpy()

    # Scale by learning rate (matching GREATS paper: tracin * lr, similarity * lr²)
    selected = greedy_selection(
        tracin_scores * lr,
        similarity_matrix * (lr ** 2),
        batch_size
    )

    return pool_idxs[np.array(selected)]


def batch_sampler(dataset, batch_size, model=None, loss_fn=None, device='cpu',
                  candidate_pool=CANDIDATE_POOL_DEFAULT, val_dataset=None,
                  lr=1e-3, **kwargs):
    """
    GREATS-style batch sampler using TracIN scoring with redundancy-aware
    greedy selection. Architecture-agnostic (uses full backprop gradients).

    Based on: "GREATS: Online Selection of High-Quality Data for LLM Training
    in Every Iteration" (NeurIPS 2024).
    """
    N = len(dataset)
    n_batches = N // batch_size

    if model is None or loss_fn is None:
        print("Warning: model or loss_fn not provided, falling back to random batching")
        indices = np.arange(N)
        np.random.shuffle(indices)
        for i in range(n_batches):
            yield indices[i * batch_size:(i + 1) * batch_size]
        return

    for _ in range(n_batches):
        # Select candidate pool from training set
        if candidate_pool < N:
            pool_idxs = np.random.choice(N, candidate_pool, replace=False)
        else:
            pool_idxs = np.arange(N)

        # Compute gradient vectors for candidate pool
        train_grads = compute_gradient_vectors(model, dataset, pool_idxs, loss_fn, device)

        if val_dataset is not None:
            # Sample a small validation subset (matching GREATS val_batchsize)
            val_idxs = np.random.choice(len(val_dataset), VAL_SAMPLE_SIZE, replace=False)
            val_grads = compute_gradient_vectors(model, val_dataset, val_idxs, loss_fn, device)

            yield get_great_batch(train_grads, val_grads, pool_idxs, batch_size, lr=lr)
        else:
            # Fallback: no validation set — use gradient norms as scores
            # (equivalent to GREATS GradNorm variant)
            grad_norms = torch.norm(train_grads, dim=1).cpu().numpy()
            similarity_matrix = (train_grads @ train_grads.T).cpu().numpy()

            selected = greedy_selection(
                grad_norms,
                similarity_matrix * 0,  # No interaction for GradNorm fallback
                batch_size
            )

            yield pool_idxs[np.array(selected)]
