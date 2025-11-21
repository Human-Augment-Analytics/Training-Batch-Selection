import numpy as np

# Random batch sampler - standard approach
# Shuffles data at the beginning of each epoch

def random_batch_indices(total, batch_size):
    # helper function to get random batch (not used currently)
    return np.random.choice(total, batch_size, replace=False)

def batch_sampler(dataset, batch_size, **kwargs):
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)  # shuffle once per epoch

    # yield batches sequentially from shuffled indices
    for start in range(0, n, batch_size):
        yield indices[start:start+batch_size]