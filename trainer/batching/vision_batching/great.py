import numpy as np
import torch

CANDIDATE_POOL_DEFAULT = 5000


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


def get_great_batch(gradient_vectors, pool_idxs, batch_size):
    """
    Select batch using GREAT (GREedy Approximation Taylor Selection).

    GREAT uses a greedy approximation of the Taylor series expansion to select
    samples that maximize the expected loss reduction by selecting gradients
    with maximum orthogonal contribution to the already selected set.
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
    first_grad = gradient_vectors[selected_indices[0]].clone()
    basis = [first_grad / torch.norm(first_grad)]

    # Greedily select remaining samples
    for _ in range(batch_size - 1):
        if remaining_mask.sum() == 0:
            break

        remaining_grads = gradient_vectors[remaining_mask]

        max_orthogonal_norm = -1
        best_idx = None
        best_orthogonal_grad = None

        for i, grad in enumerate(remaining_grads):
            # Project onto existing basis and subtract to get orthogonal component
            orthogonal_grad = grad.clone()
            for basis_vec in basis:
                projection = torch.dot(grad, basis_vec)
                orthogonal_grad -= projection * basis_vec

            orthogonal_norm = torch.norm(orthogonal_grad).item()

            if orthogonal_norm > max_orthogonal_norm:
                max_orthogonal_norm = orthogonal_norm
                remaining_indices = torch.where(remaining_mask)[0]
                best_idx = remaining_indices[i].item()
                best_orthogonal_grad = orthogonal_grad

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False

        if max_orthogonal_norm > 1e-6:
            basis.append(best_orthogonal_grad / max_orthogonal_norm)

    # Convert pool indices to original dataset indices
    return pool_idxs[np.array(selected_indices)]


def batch_sampler(dataset, batch_size, model=None, loss_fn=None, device='cpu',
                  candidate_pool=CANDIDATE_POOL_DEFAULT, **kwargs):
    """
    GREAT batch sampler that yields batches based on greedy Taylor approximation.
    Uses a candidate pool approach similar to MILO and CORESET for efficiency.
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
        # Select candidate pool
        if candidate_pool < N:
            pool_idxs = np.random.choice(N, candidate_pool, replace=False)
        else:
            pool_idxs = np.arange(N)

        # Compute gradient vectors for candidate pool
        gradient_vectors = compute_gradient_vectors(model, dataset, pool_idxs, loss_fn, device)

        # Select batch using GREAT algorithm
        yield get_great_batch(gradient_vectors, pool_idxs, batch_size)
