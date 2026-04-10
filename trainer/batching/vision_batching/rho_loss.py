import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainer.constants import EXPLORE_FRAC, TOP_K_FRAC
from trainer.pipelines.vision.utils import shape_batch_for_model

CANDIDATE_POOL_DEFAULT = 5000
REF_EPOCHS = 2

# Cache: (dataset_id, model_class_name) -> frozen reference model
# Keyed by dataset object identity so a new dataset always gets a fresh ref model.
_ref_model_cache: dict = {}


def train_reference_model(model, dataset, loss_fn, device, ref_epochs=REF_EPOCHS):
    """
    Train a reference model (same architecture) for a few epochs on the dataset.
    The reference model is used to estimate irreducible loss per sample.

    Creates a fresh copy of the model's architecture with re-initialized weights,
    trains it briefly, then freezes it.
    """
    # Deep copy to get a fresh model with same architecture but new random weights
    ref_model = copy.deepcopy(model)
    # Re-initialize weights to get a fresh starting point
    for module in ref_model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    ref_model = ref_model.to(device)
    ref_model.train()
    optimizer = torch.optim.Adam(ref_model.parameters())
    reduction_loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(ref_epochs):
        for x, y in loader:
            x = shape_batch_for_model(ref_model, x)
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = ref_model(x)
            loss = reduction_loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

    # Freeze reference model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


def compute_per_sample_losses(model, dataset, indices, loss_fn, device='cpu'):
    """
    Compute per-sample losses for a set of samples using forward pass only (no grad).
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for idx in indices:
            x, y = dataset[idx]
            x = x.unsqueeze(0).to(device)
            x = shape_batch_for_model(model, x)
            y = torch.tensor([y]).to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            losses.append(loss.item())

    model.train()
    return np.array(losses)


def get_rho_batch(rho_scores, pool_idxs, batch_size,
                  explore_frac=EXPLORE_FRAC, top_k_frac=TOP_K_FRAC):
    """
    Select batch using RHO-LOSS strategy.
    Combines exploration (random sampling) with exploitation (high RHO-loss samples).

    RHO loss = current model loss - reference model loss.
    High RHO loss means the sample is learnable but not yet learned.
    """
    n_explore = int(batch_size * explore_frac)
    n_exploit = batch_size - n_explore
    n_total = len(rho_scores)

    # Random exploration
    rand_idxs = np.random.choice(n_total, n_explore, replace=False)

    # Exploitation: select from top-k highest RHO-loss samples
    k = int(top_k_frac * n_total)
    exploit_candidates = np.argsort(-rho_scores)[:k]
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
    RHO-LOSS batch sampler that yields batches based on irreducible loss.

    Selects samples where the gap between the current model's loss and a
    reference model's loss is largest — these are "learnable but not yet
    learned" samples, making them the most informative for training.

    Based on: "Prioritized Training on Points that are Learnable, Worth
    Learning, and Not Yet Learnt" (Mindermann et al., 2022).
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

    # Cache key: dataset object identity + model class, so each (dataset, model arch)
    # pair gets its own reference model, but reuses it across epochs within a run.
    cache_key = (id(dataset), type(model).__name__)
    if cache_key not in _ref_model_cache:
        print("[RHO-LOSS] Training reference model...")
        _ref_model_cache[cache_key] = train_reference_model(model, dataset, loss_fn, device)
        print("[RHO-LOSS] Reference model ready.")
    ref_model = _ref_model_cache[cache_key]

    for _ in range(n_batches):
        # Select candidate pool
        if candidate_pool < N:
            pool_idxs = np.random.choice(N, candidate_pool, replace=False)
        else:
            pool_idxs = np.arange(N)

        # Compute losses for current model and reference model
        current_losses = compute_per_sample_losses(model, dataset, pool_idxs, loss_fn, device)
        ref_losses = compute_per_sample_losses(ref_model, dataset, pool_idxs, loss_fn, device)

        # RHO loss = current loss - reference loss
        rho_scores = current_losses - ref_losses

        # Select batch using explore/exploit on RHO scores
        yield get_rho_batch(rho_scores, pool_idxs, batch_size)
