#!/usr/bin/env python3
"""
Download Engines - How we actually download datasets

Just 4 simple engines that handle all datasets:
- TorchvisionEngine: For MNIST, CIFAR-10, etc. (torchvision library)
- URLEngine: Download from a URL
- HuggingFaceEngine: For NLP datasets from HuggingFace
- ManualEngine: Show instructions for manual downloads
"""

import csv
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict
from urllib.error import URLError

try:
    from torchvision import datasets
    from torchvision.datasets.utils import download_url
except ImportError:
    print("Warning: torchvision not available")
    datasets = None
    download_url = None

try:
    from PIL import Image
except ImportError:
    Image = None


class DownloadEngine:
    """Base class - all engines inherit from this"""

    def __init__(self, config: Dict[str, Any], dataset_root: Path):
        self.config = config
        self.dataset_root = dataset_root
        self.download_config = config.get("download", {})

    def download(self) -> bool:
        """Download the dataset. Returns True if successful."""
        raise NotImplementedError

    def is_downloaded(self) -> bool:
        """Check if dataset is already downloaded."""
        validation = self.config.get("validation", {})
        check_dirs = validation.get("check_dirs", [])
        check_files = validation.get("check_files", [])

        # Check directories
        for dir_name in check_dirs:
            if not (self.dataset_root / dir_name).exists():
                return False

        # Check files
        for file_name in check_files:
            file_path = self.dataset_root / file_name
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False

        # If no specific checks, check if directory has content
        if not check_dirs and not check_files:
            if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
                return False

        return True


class TorchvisionEngine(DownloadEngine):
    """Download MNIST, CIFAR-10, etc. using torchvision"""

    def download(self) -> bool:
        """Download dataset via torchvision."""
        if datasets is None:
            raise ImportError("torchvision is required for this dataset")

        class_name = self.download_config.get("class_name")
        if not class_name:
            raise ValueError("torchvision datasets require 'class_name' in config")

        dataset_class = getattr(datasets, class_name, None)
        if dataset_class is None:
            raise ValueError(f"Unknown torchvision dataset: {class_name}")

        print(f"[torchvision] Downloading {class_name}...")

        # Get splits configuration
        splits = self.download_config.get("splits", {})

        # Download each split
        datasets_dict = {}
        for split_name, split_config in splits.items():
            param_name = split_config.get("param_name", "train")
            param_value = split_config.get("param_value", True)

            print(f"[torchvision] Downloading {split_name} split...")

            # Build kwargs for dataset
            kwargs = {
                "root": str(self.dataset_root),
                param_name: param_value,
                "download": True,
                "transform": None,
            }

            # Add extra parameters if specified
            if "year" in self.download_config:
                kwargs["year"] = self.download_config["year"]

            # Download
            dataset = dataset_class(**kwargs)
            datasets_dict[split_name] = dataset

        # Export if needed
        export_config = self.config.get("export", {})
        export_format = export_config.get("format", "raw")

        if export_format == "class_folders":
            self._export_class_folders(datasets_dict, export_config)
        elif export_format == "raw":
            print(f"[torchvision] Keeping raw format")

        return True

    def _export_class_folders(self, datasets_dict: Dict, export_config: Dict):
        """Export dataset to class-sorted folders."""
        print(f"[export] Converting to class-folders format...")

        image_ext = export_config.get("image_extension", "png")

        for split_name, dataset in datasets_dict.items():
            split_dir = self.dataset_root / split_name

            # Skip if already exported
            if split_dir.exists() and any(split_dir.iterdir()):
                print(f"[export] {split_name} split already exported, skipping...")
                continue

            split_dir.mkdir(parents=True, exist_ok=True)

            # Get classes
            if hasattr(dataset, 'classes'):
                classes = dataset.classes
            elif 'num_classes' in export_config:
                classes = [str(i) for i in range(export_config['num_classes'])]
            else:
                raise ValueError("Cannot determine class names for export")

            # Create class directories
            for class_name in classes:
                (split_dir / str(class_name)).mkdir(exist_ok=True)

            # Export images
            class_counts = {str(c): 0 for c in classes}

            print(f"[export] Exporting {split_name} split ({len(dataset)} samples)...")
            for image, label in dataset:
                class_name = str(classes[label]) if len(classes) > label else str(label)
                count = class_counts[class_name]
                image_path = split_dir / class_name / f"{count:05d}.{image_ext}"

                if Image and hasattr(image, 'save'):
                    image.save(image_path)
                else:
                    print(f"Warning: Cannot save image (PIL not available)")

                class_counts[class_name] = count + 1

        print(f"[export] Class-folders export complete!")


class URLEngine(DownloadEngine):
    """Download from a URL (for custom datasets)"""

    def download(self) -> bool:
        """Download dataset from URL."""
        if download_url is None:
            raise ImportError("torchvision is required for URL downloads")

        # Single file download
        if "url" in self.download_config:
            url = self.download_config["url"]
            archive_config = self.download_config.get("archive", {})

            filename = Path(url).name
            archive_path = self.dataset_root / filename

            print(f"[url] Downloading from {url}...")
            download_url(url, root=str(self.dataset_root), filename=filename)

            # Extract if it's an archive
            if archive_config:
                self._extract_archive(archive_path, archive_config)

        # Multiple file downloads
        elif "sources" in self.download_config:
            sources = self.download_config["sources"]

            for source_name, source_config in sources.items():
                url = source_config["url"]
                output_name = source_config.get("output", Path(url).name)

                print(f"[url] Downloading {source_name}: {url}...")
                download_url(url, root=str(self.dataset_root), filename=output_name)

            # Post-processing
            export_config = self.config.get("export", {})
            if export_config.get("convert_csv_to_jsonl"):
                self._convert_csv_to_jsonl(export_config)

        return True

    def _extract_archive(self, archive_path: Path, archive_config: Dict):
        """Extract archive file."""
        format_type = archive_config.get("format", "tar.gz")
        extract_to = self.dataset_root / archive_config.get("extract_to", ".")

        print(f"[extract] Extracting {archive_path.name}...")

        if format_type in ("tar.gz", "tgz", "tar"):
            with tarfile.open(archive_path) as tar:
                tar.extractall(path=extract_to)
        elif format_type == "zip":
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(path=extract_to)

        # Flatten nested directory if specified
        flatten_dir = archive_config.get("flatten_nested_dir")
        if flatten_dir:
            nested = extract_to / flatten_dir
            if nested.exists():
                for item in nested.iterdir():
                    shutil.move(str(item), str(extract_to / item.name))
                nested.rmdir()

        # Remove archive
        if archive_config.get("remove_after_extract", True):
            archive_path.unlink()

    def _convert_csv_to_jsonl(self, export_config: Dict):
        """Convert CSV files to JSONL."""
        print(f"[convert] Converting CSV to JSONL...")

        sources = self.download_config.get("sources", {})
        for source_name, source_config in sources.items():
            csv_file = self.dataset_root / source_config.get("output")
            jsonl_file = csv_file.with_suffix(".jsonl")

            with open(csv_file, 'r', encoding='utf-8') as cf:
                reader = csv.DictReader(cf)
                with open(jsonl_file, 'w', encoding='utf-8') as jf:
                    for idx, row in enumerate(reader):
                        # Process row
                        if export_config.get("lowercase_keys"):
                            row = {k.lower(): v for k, v in row.items()}
                        if export_config.get("add_index"):
                            row["idx"] = idx

                        jf.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Remove CSV if specified
            if export_config.get("remove_csv_after", True):
                csv_file.unlink()

            print(f"[convert] Created {jsonl_file.name}")


class HuggingFaceEngine(DownloadEngine):
    """Download NLP datasets from HuggingFace"""

    def download(self) -> bool:
        """Download dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")

        dataset_name = self.download_config.get("dataset")
        subset = self.download_config.get("subset")
        name = self.download_config.get("name")
        max_samples = self.download_config.get("max_samples")

        print(f"[huggingface] Loading {dataset_name}...")

        # Load dataset
        kwargs = {}
        if subset:
            dataset_name = f"{dataset_name}/{subset}" if "/" not in dataset_name else dataset_name
            kwargs["name"] = subset
        if name:
            kwargs["name"] = name

        # Load specific splits or all
        splits = self.download_config.get("splits")
        if splits:
            datasets_dict = {}
            for split in splits:
                print(f"[huggingface] Loading {split} split...")
                ds = load_dataset(dataset_name, split=split, **kwargs)
                datasets_dict[split] = ds
        else:
            split = self.download_config.get("split", "train")
            dataset = load_dataset(dataset_name, split=split, **kwargs)
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            datasets_dict = {split: dataset}

        # Export
        export_config = self.config.get("export", {})
        export_format = export_config.get("format", "jsonl")

        if export_format == "jsonl":
            for split_name, ds in datasets_dict.items():
                filename_template = export_config.get("filename_template", "{split}.jsonl")
                output_file = export_config.get("output_file")

                if output_file:
                    output_path = self.dataset_root / output_file
                else:
                    output_path = self.dataset_root / filename_template.format(split=split_name)

                print(f"[huggingface] Saving {split_name} to {output_path.name}...")
                ds.to_json(str(output_path))

        return True


class ManualEngine(DownloadEngine):
    """Just show instructions (for datasets that need manual download)"""

    def download(self) -> bool:
        """Display manual download instructions."""
        instructions = self.download_config.get("instructions", "")
        instructions = instructions.format(dataset_root=self.dataset_root)

        print("\n" + "="*80)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*80)
        print(instructions)
        print("="*80 + "\n")

        return False  # Manual downloads are never "complete" automatically


# Map download methods to engine classes
ENGINE_REGISTRY = {
    "torchvision": TorchvisionEngine,
    "url": URLEngine,
    "huggingface": HuggingFaceEngine,
    "manual": ManualEngine,
}


def get_engine(method: str, config: Dict[str, Any], dataset_root: Path) -> DownloadEngine:
    """Get the right engine based on download method"""
    engine_class = ENGINE_REGISTRY.get(method)
    if engine_class is None:
        raise ValueError(f"Unknown download method: {method}")
    return engine_class(config, dataset_root)