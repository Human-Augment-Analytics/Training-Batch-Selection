import numpy as np
import torch
from trainer.constants import EXPLORE_FRAC, TOP_K_FRAC

CANDIDATE_POOL_DEFAULT = 5000


def compute_gradient_norms(model, dataset, indices, loss_fn, device='cpu'):
    """
    Compute gradient norms for a set of samples.
    """
    model.eval()
    gradient_norms = []

    for idx in indices:
        x, y = dataset[idx]
#        x = x.view(1, -1).to(device)  #LMT for flattened
        x = x.unsqueeze(0).to(device)

        y = torch.tensor([y]).to(device)

        model.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = np.sqrt(grad_norm)
        gradient_norms.append(grad_norm)

    model.train()
    return np.array(gradient_norms)


def get_grand_batch(gradient_norms, pool_idxs, batch_size, explore_frac=EXPLORE_FRAC, top_k_frac=TOP_K_FRAC):
    """
    Select batch using GraND (Gradient Normed Distance) strategy.
    Combines exploration (random sampling) with exploitation (high gradient norm samples).
    """
    n_explore = int(batch_size * explore_frac)
    n_exploit = batch_size - n_explore
    n_total = len(gradient_norms)

    # Random exploration
    rand_idxs = np.random.choice(n_total, n_explore, replace=False)

    # Exploitation: select from top-k highest gradient norms
    k = int(top_k_frac * n_total)
    exploit_candidates = np.argsort(-gradient_norms)[:k]
    exploit_idxs = np.random.choice(
        exploit_candidates,
        min(n_exploit, len(exploit_candidates)),
        replace=False
    )

    # Combine and shuffle
    batch_idxs = np.concatenate([rand_idxs, exploit_idxs])
    np.random.shuffle(batch_idxs)

    # Convert pool indices to original dataset indices
    return pool_idxs[batch_idxs]


def batch_sampler(dataset, batch_size, model=None, loss_fn=None, device='cpu',
                  candidate_pool=CANDIDATE_POOL_DEFAULT, **kwargs):
    """
    GraND batch sampler that yields batches based on gradient norms.
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

        # Compute gradient norms for candidate pool
        gradient_norms = compute_gradient_norms(model, dataset, pool_idxs, loss_fn, device)

        # Select batch using GraND algorithm
        yield get_grand_batch(gradient_norms, pool_idxs, batch_size)
