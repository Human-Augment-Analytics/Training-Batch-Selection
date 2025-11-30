DATASET_SPECS = {
     "mnist": {
         "builder": "build_mnist",
         "image_size": 28,
         "input_dim": 28 * 28,
         "num_classes": 10,
         "in_channels": 1,
         "subdir": "vision/MNIST",
    },
     "mnist_csv": {
         "builder": "build_mnist_csv",
         "image_size": 28,
         "input_dim": 28 * 28,
         "num_classes": 10,
         "in_channels": 1,
         "subdir": "vision/MNIST/csv",
    },
    "qmnist": {
        "builder": "build_qmnist",
        "image_size": 28,
        "input_dim": 28 * 28,
        "num_classes": 10,
        "in_channels": 1,
        "subdir": "vision/qmnist",
    },
    "cifar10_flat": {
        "builder": "build_cifar10_flat", 
        "image_size": 32,
        "input_dim": 1 * 32 * 32,
        "num_classes": 10,
        "in_channels": 1,
        "subdir": "vision/cifar10",
    },
    "cifar10": {
        "builder": "build_cifar10", 
        "image_size": 32,
        "input_dim": 3 * 32 * 32,
        "num_classes": 10,
        "in_channels": 3,
        "subdir": "vision/cifar10",
    },
    "cifar100": {
        "builder": "build_cifar100", 
        "image_size": 32,
        "input_dim": 3 * 32 * 32,
        "num_classes": 100,
        "in_channels": 3,
        "subdir": "vision/cifar100",
    },
    "imagenet": {
        "builder": "build_imagenet",
        "input_dim": 3 * 224 * 224,
        "num_classes": 1000,
        "in_channels": 3,
        "image_size": 224,
        "subdir": "vision/ImageNet",
    },

    # add more …
}


