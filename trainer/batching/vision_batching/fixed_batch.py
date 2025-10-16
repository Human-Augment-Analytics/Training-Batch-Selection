def batch_sampler(dataset, batch_size, **kwargs):
    n = len(dataset)
    indices = range(n)
    for start in range(0, n, batch_size):
        yield indices[start:start+batch_size]