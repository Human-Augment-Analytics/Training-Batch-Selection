DATASET_SPECS = {
     "mnist": {
         "builder": "build_mnist",
         "input_dim": 28 * 28,
         "num_classes": 10,
         "subdir": "vision",  # torchvision MNIST creates MNIST/ subdirectory automatically
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
    "qmnist_csv": {
        "builder": "build_qmnist_csv",
        "input_dim": 28 * 28,
        "num_classes": 10,
        "subdir": "vision/QMNIST/csv",
    },
    "cifar10_flat": {
        "builder": "build_cifar10_flat",
        "input_dim": 3 * 32 * 32,
        "num_classes": 10,
        "subdir": "vision/cifar10",
    },
    "cifar10_csv": {
        "builder": "build_cifar10_csv",
        "input_dim": 3 * 32 * 32,
        "num_classes": 10,
        "subdir": "vision/cifar10/csv",
    },
    "cifar100_csv": {
        "builder": "build_cifar100_csv",
        "input_dim": 3 * 32 * 32,
        "num_classes": 100,
        "subdir": "vision/cifar100/csv",
    },
    # add more …
}


