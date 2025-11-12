"""Convert MNIST from torchvision format to CSV format for legacy pipeline"""
import os
import pandas as pd
import numpy as np
from torchvision.datasets import MNIST
from pathlib import Path

def convert_mnist_to_csv():
    """Convert MNIST dataset to CSV format"""
    
    # Paths
    datasets_root = Path("datasets/vision")
    output_dir = Path("trainer/data/vision")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Converting MNIST to CSV format...")
    print()
    
    # Load MNIST datasets
    try:
        train_dataset = MNIST(root=str(datasets_root), train=True, download=False)
        test_dataset = MNIST(root=str(datasets_root), train=False, download=False)
    except:
        print("❌ MNIST not found. Downloading...")
        train_dataset = MNIST(root=str(datasets_root), train=True, download=True)
        test_dataset = MNIST(root=str(datasets_root), train=False, download=True)
    
    print(f"✔️ Loaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test")
    print()
    
    # Convert training set
    print("Converting training set...")
    train_data = []
    for idx, (img, label) in enumerate(train_dataset):
        if idx % 10000 == 0:
            print(f"  Processing {idx}/{len(train_dataset)}...")
        
        # Convert PIL Image to numpy array and flatten
        pixels = np.array(img).flatten()
        row = [label] + pixels.tolist()
        train_data.append(row)
    
    # Create DataFrame and save
    columns = ['label'] + [f'pixel{i}' for i in range(784)]
    train_df = pd.DataFrame(train_data, columns=columns)
    train_csv_path = output_dir / "mnist_train.csv"
    train_df.to_csv(train_csv_path, index=False)
    print(f"✔️ Saved: {train_csv_path} ({len(train_df)} rows)")
    print()
    
    # Convert test set
    print("Converting test set...")
    test_data = []
    for idx, (img, label) in enumerate(test_dataset):
        if idx % 2000 == 0:
            print(f"  Processing {idx}/{len(test_dataset)}...")
        
        pixels = np.array(img).flatten()
        row = [label] + pixels.tolist()
        test_data.append(row)
    
    test_df = pd.DataFrame(test_data, columns=columns)
    test_csv_path = output_dir / "mnist_test.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"✔️ Saved: {test_csv_path} ({len(test_df)} rows)")
    print()
    
    print("🎉 Conversion complete!")
    print()
    print("You can now run:")
    print("  python -m trainer.pipelines.vision.vision")

if __name__ == "__main__":
    convert_mnist_to_csv()
