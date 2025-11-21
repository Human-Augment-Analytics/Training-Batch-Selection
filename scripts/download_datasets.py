#!/usr/bin/env python3
"""
Simple Dataset Downloader - College Project Version
Just downloads datasets, nothing fancy!

Usage:
    python simple_dataset_loader.py list
    python simple_dataset_loader.py download mnist
"""

import os
import sys
from pathlib import Path

# Try importing stuff
try:
    import yaml
except:
    print("Need PyYAML! Run: pip install pyyaml")
    sys.exit(1)

try:
    from torchvision import datasets
    from torchvision.datasets.utils import download_url
    import PIL
except:
    print("Need torch stuff! Run: pip install torch torchvision pillow")
    sys.exit(1)


# Which config to use?
CONFIG_FILE = "config/dataset_config_enhanced.yaml"


def load_config():
    """Load the YAML config"""
    if not os.path.exists(CONFIG_FILE):
        print(f"Can't find {CONFIG_FILE}!")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def get_dataset_path(dataset_name, category, config):
    """Figure out where dataset should go"""
    # Get datasets root from config, with environment variable fallback
    root_from_config = config['paths'].get('datasets_root', 'datasets')
    root_from_env = os.environ.get('DATASETS_ROOT', root_from_config)
    root = Path(root_from_env)

    if category == 'vision':
        subdir = config['paths'].get('vision_subdir', 'vision')
    else:
        subdir = config['paths'].get('nlp_subdir', 'nlp')

    return root / subdir / dataset_name


def check_dataset_exists(dataset_path, validation_config):
    """Check if dataset is already downloaded"""
    if not dataset_path.exists():
        return False

    # Check for required dirs
    for dirname in validation_config.get('check_dirs', []):
        if not (dataset_path / dirname).exists():
            return False

    # Check for required files (with fallback support)
    check_files = validation_config.get('check_files', [])
    fallback_dirs = validation_config.get('fallback_check_dirs', [])

    if check_files:
        files_exist = False
        for filename in check_files:
            filepath = dataset_path / filename
            if filepath.exists() and filepath.stat().st_size > 0:
                files_exist = True
                break

        # If no files found, check fallback directories
        if not files_exist and fallback_dirs:
            for dirname in fallback_dirs:
                if (dataset_path / dirname).exists():
                    files_exist = True
                    break

        if not files_exist:
            return False

    # If nothing specified, just check if folder has stuff
    if not validation_config.get('check_dirs') and not check_files:
        if not any(dataset_path.iterdir()):
            return False

    return True


def download_torchvision_dataset(dataset_name, dataset_config, dataset_path):
    """Download using torchvision (works for CIFAR, MNIST, etc.)"""
    import socket
    download_config = dataset_config['download']
    class_name = download_config['class_name']

    print(f"Downloading {class_name} from torchvision...")

    # Increase socket timeout for large downloads
    default_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(300)  # 5 minutes

    try:
        # Get the dataset class
        dataset_class = getattr(datasets, class_name)

        # Download each split
        splits = download_config.get('splits', {})
        for split_name, split_config in splits.items():
            param_name = split_config['param_name']
            param_value = split_config['param_value']

            print(f"  - {split_name} split...")

            # Call torchvision download with retry
            kwargs = {
                'root': str(dataset_path),
                param_name: param_value,
                'download': True,
            }

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    dataset_class(**kwargs)
                    break  # Success
                except (socket.timeout, TimeoutError, ConnectionError) as e:
                    if attempt < max_retries - 1:
                        print(f"    ⚠️  Timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                    else:
                        print(f"    ❌ Failed after {max_retries} attempts")
                        raise

        print(f"✔️ Downloaded {dataset_name}")
        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"\n💡 Tip: This dataset may be available on shared storage.")
        print(f"   Check: /storage/ice-shared/cs8903onl/lw-batch-selection/datasets/vision/{dataset_name}/")
        return False

    finally:
        # Restore original timeout
        socket.setdefaulttimeout(default_timeout)


def download_url_dataset(dataset_name, dataset_config, dataset_path):
    """Download from a URL"""
    import urllib.request
    download_config = dataset_config['download']

    # Create dataset directory if it doesn't exist
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Check if single URL or multiple sources
    if 'url' in download_config:
        # Single URL (e.g., CINIC-10, Tiny ImageNet)
        url = download_config['url']
        print(f"Downloading from {url}...")

        filename = Path(url).name
        filepath = dataset_path / filename
        download_url(url, root=str(dataset_path), filename=filename)
        print(f"✔️ Downloaded {filename}")

        # Extract archive if needed
        archive_config = download_config.get('archive', {})
        if archive_config:
            import tarfile
            import zipfile
            import shutil

            archive_format = archive_config.get('format', '')
            print(f"Extracting {archive_format} archive...")

            if 'tar' in archive_format:
                with tarfile.open(filepath, 'r:*') as tar:
                    tar.extractall(path=dataset_path)
            elif archive_format == 'zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)

            print(f"✔️ Extracted archive")

            # Handle nested directory flattening
            flatten_dir = archive_config.get('flatten_nested_dir')
            if flatten_dir:
                nested_path = dataset_path / flatten_dir
                if nested_path.exists() and nested_path.is_dir():
                    # Move contents up one level
                    for item in nested_path.iterdir():
                        shutil.move(str(item), str(dataset_path / item.name))
                    nested_path.rmdir()
                    print(f"✔️ Flattened directory structure")

            # Remove archive file if requested
            if archive_config.get('remove_after_extract', False):
                filepath.unlink()
                print(f"✔️ Removed archive file")

    elif 'sources' in download_config:
        # Multiple sources (e.g., E2E NLG with train/val/test CSVs)
        sources = download_config['sources']
        print(f"Downloading {len(sources)} files...")

        for split_name, source_config in sources.items():
            url = source_config['url']
            output_filename = source_config.get('output', Path(url).name)
            output_path = dataset_path / output_filename

            print(f"  Downloading {split_name} from {url}...")
            urllib.request.urlretrieve(url, output_path)
            print(f"    ✔️ Saved to {output_path}")
    else:
        print(f"❌ No URL or sources found in config for {dataset_name}")
        return False

    print(f"✔️ Downloaded {dataset_name}")
    return True


def download_huggingface_dataset(dataset_name, dataset_config, dataset_path):
    """Download from HuggingFace"""
    try:
        from datasets import load_dataset
    except:
        print("Need HuggingFace datasets! Run: pip install datasets")
        return False

    download_config = dataset_config['download']
    hf_dataset = download_config['dataset']
    subset = download_config.get('subset', None)
    splits = download_config.get('splits', ['train', 'validation', 'test'])
    save_images = download_config.get('save_images', False)

    print(f"Downloading {hf_dataset} ({subset}) from HuggingFace...")

    # Create dataset directory if it doesn't exist
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Load and save each split
    for split in splits:
        try:
            print(f"  Downloading {split} split...")
            if subset:
                dataset = load_dataset(hf_dataset, subset, split=split)
            else:
                dataset = load_dataset(hf_dataset, split=split)

            # Check if this is an image dataset that needs special handling
            export_config = dataset_config.get('export', {})
            export_format = export_config.get('format', 'jsonl')

            if save_images or export_format == 'parquet':
                # Save images to disk in organized folders
                split_dir = dataset_path / split
                split_dir.mkdir(parents=True, exist_ok=True)

                print(f"    Saving {len(dataset)} images to {split_dir}...")
                for idx, item in enumerate(dataset):
                    # Save image if present
                    if 'image' in item and item['image'] is not None:
                        img = item['image']
                        img_path = split_dir / f"{idx:05d}.jpg"
                        img.save(str(img_path))

                    # Save mask if present (for segmentation datasets)
                    if 'mask' in item and item['mask'] is not None:
                        mask = item['mask']
                        mask_path = split_dir / f"{idx:05d}_mask.png"
                        mask.save(str(mask_path))

                print(f"    ✔️ Saved {len(dataset)} images to {split_dir}")
            else:
                # Regular text dataset - save as JSONL
                filename_template = export_config.get('filename_template', '{split}.jsonl')
                output_file = dataset_path / filename_template.format(split=split)
                dataset.to_json(str(output_file))
                print(f"    ✔️ Saved to {output_file}")

        except Exception as e:
            print(f"    ❌ Failed to download {split}: {e}")

    print(f"✔️ Dataset saved to {dataset_path}")
    return True


def download_kaggle_dataset(dataset_name, dataset_config, dataset_path):
    """Download from Kaggle using Kaggle API"""
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle API not installed!")
        print("\nTo use Kaggle datasets, you need to:")
        print("1. Install Kaggle API: pip install kaggle")
        print("2. Set up Kaggle credentials:")
        print("   - Go to https://www.kaggle.com/settings/account")
        print("   - Click 'Create New Token' to download kaggle.json")
        print("   - Place it in ~/.kaggle/kaggle.json")
        print("   - Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

    download_config = dataset_config['download']
    kaggle_dataset = download_config['dataset']
    unzip = download_config.get('unzip', True)
    archive_config = download_config.get('archive', {})

    # Create dataset directory if it doesn't exist
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Check if archive already exists
    archive_files = archive_config.get('files', [])
    existing_archive = None
    for archive_file in archive_files:
        archive_path = dataset_path / archive_file
        if archive_path.exists():
            existing_archive = archive_path
            break

    if existing_archive:
        print(f"Found existing archive: {existing_archive}")
        print(f"Extracting {existing_archive.name}...")

        try:
            import zipfile
            with zipfile.ZipFile(existing_archive, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            print(f"✔️ Extracted to {dataset_path}")

            # Optionally remove archive after extraction
            if archive_config.get('remove_after_extract', False):
                existing_archive.unlink()
                print(f"✔️ Removed archive {existing_archive.name}")

            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False

    print(f"Downloading {kaggle_dataset} from Kaggle...")

    try:
        # Download using Kaggle API
        kaggle.api.dataset_download_files(
            kaggle_dataset,
            path=str(dataset_path),
            unzip=unzip
        )

        if unzip:
            print(f"✔️ Downloaded and extracted to {dataset_path}")
        else:
            print(f"✔️ Downloaded to {dataset_path}")

        return True

    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have accepted the dataset terms on Kaggle website")
        print("2. Check your Kaggle API credentials are set up correctly")
        print(f"3. Visit: https://www.kaggle.com/datasets/{kaggle_dataset}")
        return False


def show_manual_instructions(dataset_name, dataset_config):
    """Show manual download instructions"""
    download_config = dataset_config['download']
    instructions = download_config.get('instructions', 'Manual download required.')

    print("\n" + "="*60)
    print(f"MANUAL DOWNLOAD: {dataset_name}")
    print("="*60)
    print(instructions)
    print("="*60 + "\n")
    return False


def download_dataset(dataset_name, dataset_config, dataset_path):
    """Main download function - routes to correct method"""
    method = dataset_config['download']['method']

    # Make directory
    dataset_path.mkdir(parents=True, exist_ok=True)

    if method == 'torchvision':
        return download_torchvision_dataset(dataset_name, dataset_config, dataset_path)
    elif method == 'url':
        return download_url_dataset(dataset_name, dataset_config, dataset_path)
    elif method == 'huggingface':
        return download_huggingface_dataset(dataset_name, dataset_config, dataset_path)
    elif method == 'kaggle':
        return download_kaggle_dataset(dataset_name, dataset_config, dataset_path)
    elif method == 'manual':
        return show_manual_instructions(dataset_name, dataset_config)
    else:
        print(f"Unknown method: {method}")
        return False


def list_datasets():
    """List all datasets"""
    config = load_config()

    print("\n" + "="*80)
    print("  DATASETS")
    print("="*80 + "\n")

    # Vision datasets
    print("VISION:")
    for name, ds_config in config.get('vision_datasets', {}).items():
        path = get_dataset_path(name, 'vision', config)
        exists = check_dataset_exists(path, ds_config.get('validation', {}))

        status = "✔️" if exists else "❌"
        size = ds_config.get('estimated_size_mb', 0)
        size_str = f"{size}MB" if size < 1024 else f"{size/1024:.1f}GB"

        print(f"  {status} {name:20s} {size_str:>10s}  {ds_config.get('description', '')}")

    # NLP datasets
    print("\nNLP:")
    for name, ds_config in config.get('nlp_datasets', {}).items():
        path = get_dataset_path(name, 'nlp', config)
        exists = check_dataset_exists(path, ds_config.get('validation', {}))

        status = "✔️" if exists else "❌"
        size = ds_config.get('estimated_size_mb', 0)
        size_str = f"{size}MB" if size < 1024 else f"{size/1024:.1f}GB"

        print(f"  {status} {name:20s} {size_str:>10s}  {ds_config.get('description', '')}")

    print()


def download_one(dataset_name, auto_yes=False):
    """Download a specific dataset"""
    config = load_config()

    # Find the dataset
    dataset_config = None
    category = None

    if dataset_name in config.get('vision_datasets', {}):
        dataset_config = config['vision_datasets'][dataset_name]
        category = 'vision'
    elif dataset_name in config.get('nlp_datasets', {}):
        dataset_config = config['nlp_datasets'][dataset_name]
        category = 'nlp'
    else:
        print(f"Dataset '{dataset_name}' not found!")
        print("Run 'list' to see available datasets")
        return

    # Get path
    dataset_path = get_dataset_path(dataset_name, category, config)

    # Check if already exists (fully extracted)
    validation_config = dataset_config.get('validation', {})
    if check_dataset_exists(dataset_path, validation_config):
        # For Kaggle datasets with archives, check if we only have zip but not extracted files
        download_config = dataset_config.get('download', {})
        if download_config.get('method') == 'kaggle' and 'archive' in download_config:
            archive_files = download_config['archive'].get('files', [])
            fallback_dirs = validation_config.get('fallback_check_dirs', [])

            # Check if we have fallback dirs (extracted content)
            has_extracted = False
            for dirname in fallback_dirs:
                if (dataset_path / dirname).exists():
                    has_extracted = True
                    break

            # If no extracted dirs but have archive, proceed to extract
            if not has_extracted:
                for archive_file in archive_files:
                    if (dataset_path / archive_file).exists():
                        # Don't return early - let it proceed to extraction
                        break
            else:
                print(f"'{dataset_name}' already downloaded at {dataset_path}")
                return
        else:
            print(f"'{dataset_name}' already downloaded at {dataset_path}")
            return

    # Confirm
    if not auto_yes:
        size = dataset_config.get('estimated_size_mb', 0)
        size_str = f"{size}MB" if size < 1024 else f"{size/1024:.1f}GB"

        response = input(f"Download {dataset_name} (~{size_str})? [y/N]: ").lower()
        if response not in ['y', 'yes']:
            print("Cancelled.")
            return

    # Download!
    print(f"\nDownloading {dataset_name}...")
    success = download_dataset(dataset_name, dataset_config, dataset_path)

    if success:
        print(f"\n✔️ Done! Dataset saved to: {dataset_path}")
    else:
        print(f"\n❌ Download failed or requires manual setup")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python simple_dataset_loader.py list")
        print("  python simple_dataset_loader.py download <dataset_name>")
        print("  python simple_dataset_loader.py download <dataset_name> --yes")
        return

    command = sys.argv[1]

    if command == 'list':
        list_datasets()

    elif command == 'download':
        if len(sys.argv) < 3:
            print("Specify dataset name!")
            print("Run 'list' to see options")
            return

        dataset_name = sys.argv[2]
        auto_yes = '--yes' in sys.argv or '-y' in sys.argv

        download_one(dataset_name, auto_yes)

    else:
        print(f"Unknown command: {command}")
        print("Use 'list' or 'download'")


if __name__ == '__main__':
    main()
