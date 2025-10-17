DATASET_SPECS = {
     "mnist": {
         "builder": "build_mnist",
         "input_dim": 28 * 28,
         "num_classes": 10,
         "subdir": "vision/MNIST",
    },
     "mnist_csv": {
         "builder": "build_mnist_csv",
         "input_dim": 28 * 28,
         "num_classes": 10,
         "subdir": "vision/MNIST/csv",
    },
    "qmnist": {
        "builder": "build_qmnist",
        "input_dim": 28 * 28,
        "num_classes": 10,
        "subdir": "vision/qmnist",
    },
    "cifar10_flat": {
        "builder": "build_cifar10_flat", 
        "input_dim": 3 * 32 * 32,
        "num_classes": 10,
        "subdir": "vision/cifar10",
    },
    # add more …
}


