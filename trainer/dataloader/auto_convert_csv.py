"""Automatic MNIST CSV conversion utility"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision.datasets import MNIST


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
    mnist_dir = Path(root_dir) / "vision"
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


if __name__ == "__main__":
    # Can be run standalone
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    ensure_mnist_csv(root)
