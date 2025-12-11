import numpy as np

CANDIDATE_POOL = 5000   # reduce from 60000 → 5000

def get_coreset_batch(dataset, batch_size, candidate_pool=CANDIDATE_POOL):
    N = len(dataset)

    # Pick a random subset (5k instead of 60k)
    if candidate_pool < N:
        pool_idxs = np.random.choice(N, candidate_pool, replace=False)
    else:
        pool_idxs = np.arange(N)

    # Preload only these 5k images
    pool_imgs = np.stack([dataset[i][0].numpy().flatten() for i in pool_idxs])

    # K-center selection inside this reduced pool
    first = np.random.randint(0, len(pool_idxs))
    centers = [first]

    min_dists = np.linalg.norm(pool_imgs - pool_imgs[first], axis=1)

    for _ in range(batch_size - 1):
        next_center = np.argmax(min_dists)
        centers.append(next_center)

        new_dists = np.linalg.norm(pool_imgs - pool_imgs[next_center], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    # Convert pool indices → original dataset indices
    return pool_idxs[centers]


def batch_sampler(dataset, batch_size, **kwargs):
    N = len(dataset)
    n_batches = N // batch_size
    for _ in range(n_batches):
        yield get_coreset_batch(dataset, batch_size)
