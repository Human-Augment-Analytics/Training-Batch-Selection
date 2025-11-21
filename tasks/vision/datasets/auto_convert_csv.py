"""Automatic CSV conversion utility for vision datasets"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision.datasets import MNIST, QMNIST, CIFAR10, CIFAR100


def ensure_mnist_csv(root_dir, force_regenerate=False):
    """
    Automatically ensure MNIST CSV files exist.
    If they don't exist, convert from torchvision format.
    
    Args:
        root_dir: Root directory where datasets are stored
        force_regenerate: If True, regenerate even if CSV exists
        
    Returns:
        tuple: (train_csv_path, test_csv_path)
    """
    # Two possible locations for CSVs
    locations = [
        Path(root_dir) / "vision" / "MNIST" / "csv",  # For benchmark_datasets
        Path("trainer") / "data" / "vision",           # For vision.py
    ]
    
    train_csv = "mnist_train.csv"
    test_csv = "mnist_test.csv"
    
    # Check if CSVs already exist in any location
    csv_exists = False
    existing_location = None
    
    for loc in locations:
        if (loc / train_csv).exists() and (loc / test_csv).exists():
            csv_exists = True
            existing_location = loc
            break
    
    if csv_exists and not force_regenerate:
        print(f"✔️ MNIST CSVs found at: {existing_location}")
        return str(existing_location / train_csv), str(existing_location / test_csv)
    
    # Need to generate CSVs
    print("🔄 Converting MNIST to CSV format...")
    
    # Ensure all locations exist
    for loc in locations:
        loc.mkdir(parents=True, exist_ok=True)
    
    # Load MNIST from torchvision
    # Use MNIST subfolder so torchvision downloads to datasets/vision/MNIST/
    mnist_dir = Path(root_dir) / "vision" / "MNIST"
    try:
        train_dataset = MNIST(root=str(mnist_dir), train=True, download=False)
        test_dataset = MNIST(root=str(mnist_dir), train=False, download=False)
        print(f"✔️ Loaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test")
    except:
        print("⚠️  MNIST not found. Downloading...")
        train_dataset = MNIST(root=str(mnist_dir), train=True, download=True)
        test_dataset = MNIST(root=str(mnist_dir), train=False, download=True)
        print(f"✔️ Downloaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Convert training set
    print(f"  Converting training set ({len(train_dataset)} samples)...")
    train_data = []
    for idx, (img, label) in enumerate(train_dataset):
        if idx % 10000 == 0 and idx > 0:
            print(f"    Progress: {idx}/{len(train_dataset)}")
        pixels = np.array(img).flatten()
        row = [label] + pixels.tolist()
        train_data.append(row)
    
    columns = ['label'] + [f'pixel{i}' for i in range(784)]
    train_df = pd.DataFrame(train_data, columns=columns)
    
    # Convert test set
    print(f"  Converting test set ({len(test_dataset)} samples)...")
    test_data = []
    for idx, (img, label) in enumerate(test_dataset):
        if idx % 2000 == 0 and idx > 0:
            print(f"    Progress: {idx}/{len(test_dataset)}")
        pixels = np.array(img).flatten()
        row = [label] + pixels.tolist()
        test_data.append(row)
    
    test_df = pd.DataFrame(test_data, columns=columns)
    
    # Save to all locations
    for loc in locations:
        train_path = loc / train_csv
        test_path = loc / test_csv
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"✔️ Saved CSVs to: {loc}")
    
    print("🎉 CSV conversion complete!")
    
    # Return primary location (first one)
    primary_loc = locations[0]
    return str(primary_loc / train_csv), str(primary_loc / test_csv)


def ensure_qmnist_csv(root_dir, force_regenerate=False):
    """
    Automatically ensure QMNIST CSV files exist.
    If they don't exist, convert from torchvision format.

    Args:
        root_dir: Root directory where datasets are stored
        force_regenerate: If True, regenerate even if CSV exists

    Returns:
        tuple: (train_csv_path, test_csv_path)
    """
    # Two possible locations for CSVs
    locations = [
        Path(root_dir) / "vision" / "QMNIST" / "csv",  # For benchmark_datasets
        Path("trainer") / "data" / "vision",            # For vision.py
    ]

    train_csv = "qmnist_train.csv"
    test_csv = "qmnist_test.csv"

    # Check if CSVs already exist in any location
    csv_exists = False
    existing_location = None

    for loc in locations:
        if (loc / train_csv).exists() and (loc / test_csv).exists():
            csv_exists = True
            existing_location = loc
            break

    if csv_exists and not force_regenerate:
        print(f"QMNIST CSVs found at: {existing_location}")
        return str(existing_location / train_csv), str(existing_location / test_csv)

    # need to generate CSVs
    print("Converting QMNIST to CSV format...")

    # ensure all locations exist
    for loc in locations:
        loc.mkdir(parents=True, exist_ok=True)

    # load QMNIST from torchvision
    # Use QMNIST subfolder so torchvision downloads to datasets/vision/QMNIST/
    qmnist_dir = Path(root_dir) / "vision" / "QMNIST"
    try:
        train_dataset = QMNIST(root=str(qmnist_dir), train=True, download=False)
        test_dataset = QMNIST(root=str(qmnist_dir), train=False, download=False)
        print(f"Loaded QMNIST: {len(train_dataset)} train, {len(test_dataset)} test")
    except:
        print("QMNIST not found. Downloading...")
        train_dataset = QMNIST(root=str(qmnist_dir), train=True, download=True)
        test_dataset = QMNIST(root=str(qmnist_dir), train=False, download=True)
        print(f"Downloaded QMNIST: {len(train_dataset)} train, {len(test_dataset)} test")

    # convert training set
    print(f"  Converting training set ({len(train_dataset)} samples)...")
    train_data = []
    for idx, (img, label) in enumerate(train_dataset):
        if idx % 10000 == 0 and idx > 0:
            print(f"    Progress: {idx}/{len(train_dataset)}")
        pixels = np.array(img).flatten()
        row = [label] + pixels.tolist()
        train_data.append(row)

    columns = ['label'] + [f'pixel{i}' for i in range(784)]
    train_df = pd.DataFrame(train_data, columns=columns)

    # convert test set
    print(f"  Converting test set ({len(test_dataset)} samples)...")
    test_data = []
    for idx, (img, label) in enumerate(test_dataset):
        if idx % 10000 == 0 and idx > 0:
            print(f"    Progress: {idx}/{len(test_dataset)}")
        pixels = np.array(img).flatten()
        row = [label] + pixels.tolist()
        test_data.append(row)

    test_df = pd.DataFrame(test_data, columns=columns)

    # save to all locations
    for loc in locations:
        train_path = loc / train_csv
        test_path = loc / test_csv

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Saved CSVs to: {loc}")

    print("CSV conversion complete!")

    # return primary location (first one)
    primary_loc = locations[0]
    return str(primary_loc / train_csv), str(primary_loc / test_csv)


def ensure_cifar10_csv(root_dir, force_regenerate=False):
    """
    Automatically ensure CIFAR10 CSV files exist.
    If they don't exist, convert from torchvision format.

    Args:
        root_dir: Root directory where datasets are stored
        force_regenerate: If True, regenerate even if CSV exists

    Returns:
        tuple: (train_csv_path, test_csv_path)
    """
    # Two possible locations for CSVs
    locations = [
        Path(root_dir) / "vision" / "cifar10" / "csv",  # For benchmark_datasets
        Path("trainer") / "data" / "vision",             # For vision.py
    ]

    train_csv = "cifar10_train.csv"
    test_csv = "cifar10_test.csv"

    # check if CSVs already exist in any location
    csv_exists = False
    existing_location = None

    for loc in locations:
        if (loc / train_csv).exists() and (loc / test_csv).exists():
            csv_exists = True
            existing_location = loc
            break

    if csv_exists and not force_regenerate:
        print(f"CIFAR10 CSVs found at: {existing_location}")
        return str(existing_location / train_csv), str(existing_location / test_csv)

    # need to generate CSVs
    print("Converting CIFAR10 to CSV format...")
    print("Note: This will create ~615 MB of CSV files")

    # ensure all locations exist
    for loc in locations:
        loc.mkdir(parents=True, exist_ok=True)

    # load CIFAR10 from torchvision
    # Use cifar10 subfolder so torchvision downloads to datasets/vision/cifar10/
    cifar_dir = Path(root_dir) / "vision" / "cifar10"
    try:
        train_dataset = CIFAR10(root=str(cifar_dir), train=True, download=False)
        test_dataset = CIFAR10(root=str(cifar_dir), train=False, download=False)
        print(f"Loaded CIFAR10: {len(train_dataset)} train, {len(test_dataset)} test")
    except:
        print("CIFAR10 not found. Downloading...")
        train_dataset = CIFAR10(root=str(cifar_dir), train=True, download=True)
        test_dataset = CIFAR10(root=str(cifar_dir), train=False, download=True)
        print(f"Downloaded CIFAR10: {len(train_dataset)} train, {len(test_dataset)} test")

    # convert training set
    print(f"  Converting training set ({len(train_dataset)} samples)...")
    train_data = []
    for idx, (img, label) in enumerate(train_dataset):
        if idx % 10000 == 0 and idx > 0:
            print(f"    Progress: {idx}/{len(train_dataset)}")
        # CIFAR10 images are 32x32x3 RGB
        pixels = np.array(img).flatten()  # flatten to 3072 values
        row = [label] + pixels.tolist()
        train_data.append(row)

    # 3072 columns for RGB image (32*32*3)
    columns = ['label'] + [f'pixel{i}' for i in range(3072)]
    train_df = pd.DataFrame(train_data, columns=columns)

    # convert test set
    print(f"  Converting test set ({len(test_dataset)} samples)...")
    test_data = []
    for idx, (img, label) in enumerate(test_dataset):
        if idx % 2000 == 0 and idx > 0:
            print(f"    Progress: {idx}/{len(test_dataset)}")
        pixels = np.array(img).flatten()
        row = [label] + pixels.tolist()
        test_data.append(row)

    test_df = pd.DataFrame(test_data, columns=columns)

    # save to all locations
    print("Saving CSV files (this may take a minute)...")
    for loc in locations:
        train_path = loc / train_csv
        test_path = loc / test_csv

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Saved CSVs to: {loc}")

    print("CSV conversion complete!")

    # return primary location (first one)
    primary_loc = locations[0]
    return str(primary_loc / train_csv), str(primary_loc / test_csv)


def ensure_cifar100_csv(root_dir, force_regenerate=False):
    """
    Automatically ensure CIFAR100 CSV files exist.
    If they don't exist, convert from torchvision format.

    Args:
        root_dir: Root directory where datasets are stored
        force_regenerate: If True, regenerate even if CSV exists

    Returns:
        tuple: (train_csv_path, test_csv_path)
    """
    # Two possible locations for CSVs
    locations = [
        Path(root_dir) / "vision" / "cifar100" / "csv",  # For benchmark_datasets
        Path("trainer") / "data" / "vision",              # For vision.py
    ]

    train_csv = "cifar100_train.csv"
    test_csv = "cifar100_test.csv"

    # check if CSVs already exist in any location
    csv_exists = False
    existing_location = None

    for loc in locations:
        if (loc / train_csv).exists() and (loc / test_csv).exists():
            csv_exists = True
            existing_location = loc
            break

    if csv_exists and not force_regenerate:
        print(f"CIFAR100 CSVs found at: {existing_location}")
        return str(existing_location / train_csv), str(existing_location / test_csv)

    # need to generate CSVs
    print("Converting CIFAR100 to CSV format...")
    print("Note: This will create ~615 MB of CSV files")

    # ensure all locations exist
    for loc in locations:
        loc.mkdir(parents=True, exist_ok=True)

    # load CIFAR100 from torchvision
    # Use cifar100 subfolder so torchvision downloads to datasets/vision/cifar100/
    cifar_dir = Path(root_dir) / "vision" / "cifar100"
    try:
        train_dataset = CIFAR100(root=str(cifar_dir), train=True, download=False)
        test_dataset = CIFAR100(root=str(cifar_dir), train=False, download=False)
        print(f"Loaded CIFAR100: {len(train_dataset)} train, {len(test_dataset)} test")
    except:
        print("CIFAR100 not found. Downloading...")
        train_dataset = CIFAR100(root=str(cifar_dir), train=True, download=True)
        test_dataset = CIFAR100(root=str(cifar_dir), train=False, download=True)
        print(f"Downloaded CIFAR100: {len(train_dataset)} train, {len(test_dataset)} test")

    # convert training set
    print(f"  Converting training set ({len(train_dataset)} samples)...")
    train_data = []
    for idx, (img, label) in enumerate(train_dataset):
        if idx % 10000 == 0 and idx > 0:
            print(f"    Progress: {idx}/{len(train_dataset)}")
        # CIFAR100 images are 32x32x3 RGB (same as CIFAR10)
        pixels = np.array(img).flatten()  # flatten to 3072 values
        row = [label] + pixels.tolist()
        train_data.append(row)

    # 3072 columns for RGB image (32*32*3)
    columns = ['label'] + [f'pixel{i}' for i in range(3072)]
    train_df = pd.DataFrame(train_data, columns=columns)

    # convert test set
    print(f"  Converting test set ({len(test_dataset)} samples)...")
    test_data = []
    for idx, (img, label) in enumerate(test_dataset):
        if idx % 2000 == 0 and idx > 0:
            print(f"    Progress: {idx}/{len(test_dataset)}")
        pixels = np.array(img).flatten()
        row = [label] + pixels.tolist()
        test_data.append(row)

    test_df = pd.DataFrame(test_data, columns=columns)

    # save to all locations
    print("Saving CSV files (this may take a minute)...")
    for loc in locations:
        train_path = loc / train_csv
        test_path = loc / test_csv

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Saved CSVs to: {loc}")

    print("CSV conversion complete!")

    # return primary location (first one)
    primary_loc = locations[0]
    return str(primary_loc / train_csv), str(primary_loc / test_csv)


if __name__ == "__main__":
    # can be run standalone for MNIST, QMNIST, CIFAR10, or CIFAR100
    import sys

    if len(sys.argv) < 2:
        print("Usage: python auto_convert_csv.py [mnist|qmnist|cifar10|cifar100] [root_dir]")
        print("Example: python auto_convert_csv.py cifar10 datasets/")
        sys.exit(1)

    dataset_type = sys.argv[1].lower()
    root = sys.argv[2] if len(sys.argv) > 2 else "datasets"

    if dataset_type == "mnist":
        ensure_mnist_csv(root)
    elif dataset_type == "qmnist":
        ensure_qmnist_csv(root)
    elif dataset_type == "cifar10":
        ensure_cifar10_csv(root)
    elif dataset_type == "cifar100":
        ensure_cifar100_csv(root)
    else:
        print(f"Unknown dataset type: {dataset_type}")
        print("Supported: mnist, qmnist, cifar10, cifar100")
        sys.exit(1)
