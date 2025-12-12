import numpy as np

def batch_sampler(dataset, batch_size, **kwargs):
    """Standard random batch sampling for NLP tasks."""
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start:start + batch_size].tolist()
