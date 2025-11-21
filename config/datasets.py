"""
Dataset specifications for all supported datasets.
"""

DATASET_SPECS = {
    # ========== Vision Datasets ==========
    "mnist": {
        "builder": "build_mnist",
        "input_dim": 28 * 28,
        "num_classes": 10,
        "subdir": "vision",
        "task": "vision",
    },
    "mnist_csv": {
        "builder": "build_mnist_csv",
        "input_dim": 28 * 28,
        "num_classes": 10,
        "subdir": "vision/MNIST/csv",
        "task": "vision",
    },
    "qmnist": {
        "builder": "build_qmnist",
        "input_dim": 28 * 28,
        "num_classes": 10,
        "subdir": "vision/qmnist",
        "task": "vision",
    },
    "qmnist_csv": {
        "builder": "build_qmnist_csv",
        "input_dim": 28 * 28,
        "num_classes": 10,
        "subdir": "vision/QMNIST/csv",
        "task": "vision",
    },
    "cifar10_flat": {
        "builder": "build_cifar10_flat",
        "input_dim": 3 * 32 * 32,
        "num_classes": 10,
        "subdir": "vision/cifar10",
        "task": "vision",
    },
    "cifar10_csv": {
        "builder": "build_cifar10_csv",
        "input_dim": 3 * 32 * 32,
        "num_classes": 10,
        "subdir": "vision/cifar10/csv",
        "task": "vision",
    },
    "cifar100_csv": {
        "builder": "build_cifar100_csv",
        "input_dim": 3 * 32 * 32,
        "num_classes": 100,
        "subdir": "vision/cifar100/csv",
        "task": "vision",
    },

    # ========== NLP Datasets ==========
    # Add NLP dataset specs here in the future
}
