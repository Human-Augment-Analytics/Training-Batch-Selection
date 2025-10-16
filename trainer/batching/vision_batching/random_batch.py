import numpy as np

def random_batch_indices(total, batch_size):
    return np.random.choice(total, batch_size, replace=False)

def batch_sampler(dataset, batch_size, **kwargs):
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start:start+batch_size]