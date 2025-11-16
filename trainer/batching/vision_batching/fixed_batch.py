# Fixed batch sampler - no shuffling
# Always returns data in the same order
# useful as a baseline

def batch_sampler(dataset, batch_size, **kwargs):
    n = len(dataset)
    indices = range(n)  # no shuffling

    for start in range(0, n, batch_size):
        yield indices[start:start+batch_size]