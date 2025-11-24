#!/usr/bin/env python3
"""
Download and setup CINIC-10 dataset.

CINIC-10 is an augmented extension of CIFAR-10 with ImageNet images.
Total: 270,000 images (90,000 train + 90,000 valid + 90,000 test)

This script:
1. Downloads CINIC-10 from Edinburgh DataShare
2. Extracts it to datasets/vision/cinic-10/
3. Optionally converts to CSV format for training

Usage:
    python scripts/setup_cinic10.py                    # Download only
    python scripts/setup_cinic10.py --convert-csv      # Download + convert to CSV
    python scripts/setup_cinic10.py --csv-only         # Convert existing data to CSV
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_cinic10(datasets_root):
    """Download CINIC-10 dataset."""
    cinic_dir = Path(datasets_root) / "vision" / "cinic-10"

    # Check if already downloaded
    if (cinic_dir / "train").exists() and (cinic_dir / "test").exists():
        print(f"✓ CINIC-10 already downloaded at: {cinic_dir}")
        return cinic_dir

    print("=" * 70)
    print("DOWNLOADING CINIC-10 DATASET")
    print("=" * 70)
    print(f"Destination: {cinic_dir}")
    print("Size: ~1.7 GB compressed, ~2.5 GB uncompressed")
    print()

    # Create directory
    cinic_dir.mkdir(parents=True, exist_ok=True)

    # Download URL
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    tar_file = cinic_dir.parent / "CINIC-10.tar.gz"

    print(f"Downloading from: {url}")
    print(f"This may take several minutes depending on your connection...")
    print()

    # Download using wget or curl
    try:
        # Try wget first
        result = subprocess.run(
            ["wget", "-O", str(tar_file), url],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Download complete using wget")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fall back to curl
            result = subprocess.run(
                ["curl", "-L", "-o", str(tar_file), url],
                check=True,
                capture_output=True,
                text=True
            )
            print("✓ Download complete using curl")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Error: Neither wget nor curl found.")
            print("Please install wget or curl, or download manually from:")
            print(f"  {url}")
            print(f"  Extract to: {cinic_dir}")
            return None

    # Extract
    print()
    print("Extracting archive...")
    try:
        subprocess.run(
            ["tar", "-xzf", str(tar_file), "-C", str(cinic_dir.parent)],
            check=True
        )
        print(f"✓ Extracted to: {cinic_dir}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error extracting: {e}")
        return None

    # Clean up tar file
    print()
    print("Cleaning up...")
    tar_file.unlink()
    print(f"✓ Removed temporary file: {tar_file}")

    # Verify structure
    print()
    print("Verifying dataset structure...")
    expected_dirs = ["train", "test", "valid"]
    expected_classes = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    for split in expected_dirs:
        split_dir = cinic_dir / split
        if not split_dir.exists():
            print(f"✗ Warning: Missing {split} directory")
            continue

        for class_name in expected_classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"✗ Warning: Missing {split}/{class_name}")
            else:
                num_images = len(list(class_dir.glob("*.png")))
                if num_images == 0:
                    print(f"✗ Warning: No images in {split}/{class_name}")

    print("✓ Dataset structure verified")
    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Location: {cinic_dir}")
    print()

    return cinic_dir


def convert_to_csv(datasets_root):
    """Convert CINIC-10 to CSV format."""
    print()
    print("=" * 70)
    print("CONVERTING TO CSV FORMAT")
    print("=" * 70)
    print()

    try:
        from tasks.vision.datasets.auto_convert_csv import ensure_cinic10_csv
        train_csv, test_csv = ensure_cinic10_csv(str(datasets_root))

        print()
        print("=" * 70)
        print("CONVERSION COMPLETE")
        print("=" * 70)
        print(f"Training CSV: {train_csv}")
        print(f"Test CSV: {test_csv}")
        print()
        return True
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and setup CINIC-10 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download only
  python scripts/setup_cinic10.py

  # Download and convert to CSV
  python scripts/setup_cinic10.py --convert-csv

  # Convert existing data to CSV
  python scripts/setup_cinic10.py --csv-only

  # Use custom datasets directory
  python scripts/setup_cinic10.py --datasets-root /path/to/datasets --convert-csv
        """
    )

    parser.add_argument(
        '--datasets-root',
        type=str,
        default='datasets',
        help='Root directory for datasets (default: datasets)'
    )
    parser.add_argument(
        '--convert-csv',
        action='store_true',
        help='Convert to CSV format after downloading'
    )
    parser.add_argument(
        '--csv-only',
        action='store_true',
        help='Only convert to CSV (skip download)'
    )

    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)

    # Step 1: Download (unless csv-only)
    if not args.csv_only:
        cinic_dir = download_cinic10(datasets_root)
        if cinic_dir is None:
            print("\n✗ Setup failed during download")
            return 1
    else:
        cinic_dir = datasets_root / "vision" / "cinic-10"
        if not cinic_dir.exists():
            print(f"✗ Error: CINIC-10 not found at {cinic_dir}")
            print("Run without --csv-only to download first")
            return 1

    # Step 2: Convert to CSV (if requested)
    if args.convert_csv or args.csv_only:
        success = convert_to_csv(datasets_root)
        if not success:
            print("\n✗ Setup failed during CSV conversion")
            return 1

    # Success message
    print()
    print("✓ CINIC-10 setup complete!")
    print()
    print("Next steps:")
    if not (args.convert_csv or args.csv_only):
        print("  1. Convert to CSV format:")
        print("     python scripts/setup_cinic10.py --csv-only")
        print()
    print("  2. Run experiments:")
    print("     python -m tasks.vision.run_all --dataset cinic10_csv")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
