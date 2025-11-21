import numpy as np
from trainer.constants import EXPLORE_FRAC, TOP_K_FRAC

# Smart batching: prioritize hard examples based on loss
# Idea: mix exploration (random samples) with exploitation (high-loss samples)

def get_smart_batch(loss_history, batch_size, explore_frac=EXPLORE_FRAC, top_k_frac=TOP_K_FRAC):
    # split batch into exploration and exploitation
    n_explore = int(batch_size * explore_frac)
    n_exploit = batch_size - n_explore
    n_total = len(loss_history)

    # random samples for exploration
    rand_idxs = np.random.choice(n_total, n_explore, replace=False)

    # select from top-k highest loss samples for exploitation
    k = int(top_k_frac * n_total)
    exploit_candidates = np.argsort(-loss_history)[:k]  # negative for descending sort
    exploit_idxs = np.random.choice(exploit_candidates, min(n_exploit, len(exploit_candidates)), replace=False)

    # combine and shuffle
    batch_idxs = np.concatenate([rand_idxs, exploit_idxs])
    np.random.shuffle(batch_idxs)

    return batch_idxs

def batch_sampler(dataset, batch_size, loss_history=None, **kwargs):
    n = len(dataset)
    n_batches = n // batch_size

    for _ in range(n_batches):
        yield get_smart_batch(loss_history, batch_size)