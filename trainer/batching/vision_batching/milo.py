import numpy as np
from sklearn.decomposition import PCA

CANDIDATE_POOL_DEFAULT = 5000
PCA_DIMS = None

def _prepare_pool(dataset, pool_idxs, pca=None):
    imgs = np.stack([dataset[i][0].numpy().reshape(-1) for i in pool_idxs])
    if pca is not None:
        imgs = pca.transform(imgs)
    return imgs

def get_milo_batch(dataset, batch_size, candidate_pool=CANDIDATE_POOL_DEFAULT, pca=None):
    N = len(dataset)
    if candidate_pool < N:
        pool_idxs = np.random.choice(N, candidate_pool, replace=False)
    else:
        pool_idxs = np.arange(N)

    pool_imgs = _prepare_pool(dataset, pool_idxs, pca)

    start = np.random.randint(0, len(pool_idxs))
    batch = [start]

    for _ in range(batch_size - 1):
        dists = np.min(np.linalg.norm(pool_imgs[:, None, :] - pool_imgs[batch][None, :, :], axis=2), axis=1)
        dists[batch] = -np.inf
        next_idx = np.argmax(dists)
        batch.append(next_idx)

    return pool_idxs[np.array(batch)]

def batch_sampler(dataset, batch_size, candidate_pool=CANDIDATE_POOL_DEFAULT, pca_dims=PCA_DIMS, **kwargs):
    N = len(dataset)
    n_batches = N // batch_size

    pca = None
    if pca_dims is not None:
        X = np.stack([dataset[i][0].numpy().reshape(-1) for i in range(len(dataset))])
        pca = PCA(n_components=pca_dims).fit(X)

    for _ in range(n_batches):
        yield get_milo_batch(dataset, batch_size, candidate_pool=candidate_pool, pca=pca)
