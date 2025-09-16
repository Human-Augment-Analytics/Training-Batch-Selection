# Pretraining Dataloader

This script helps you **download, split, and tokenize** a dataset from Hugging Face.
It is useful for preparing text datasets for pretraining large language models.

---

## Quickstart:
    python -m trainer.data.loader --max_rows 600000  --split_examples 10000 --dataset_name cerebras/SlimPajama-627B --dataset_col text

---

## Features

* Download a subset of a Hugging Face dataset.
* Save raw text to disk in one or multiple files.
* Split large datasets into smaller chunks.
* Tokenize text with a Hugging Face tokenizer.
* Save tokenized arrays (`.npy`) for efficient loading.

At a high-level, the script does the following:

1. Downloads a dataset from huggingface, into the trainer/data/downloads directory.
2. Splits the downloaded dataset into 'Segments' in trainer/data/segments, to allow parallel tokenization.
3. Tokenizes the Segments in parallel

---

## Usage

```bash
python setup_dataset.py [FLAGS]
```

---

## Flags

### `--dataset_name`

* **Type:** `str`
* **Default:** `tiiuae/falcon-refinedweb`
* Hugging Face dataset name to download.

### `--max_rows`

* **Type:** `int`
* **Default:** `50_000`
* Maximum number of rows to download.
  Use this to sample a subset instead of the full dataset.

### `--batch_size`

* **Type:** `int`
* **Default:** `10_000`
* Number of rows to buffer before writing them to disk.
  Larger batch size = fewer I/O operations but higher memory usage.

### `--split_examples`

* **Type:** `int`
* **Default:** `10_000`
* Maximum number of examples per split file.
  Each file will contain at most this many examples.

### `--tokenizer_model`

* **Type:** `str`
* **Default:** `meta-llama/Meta-Llama-3-8B-Instruct`
* Hugging Face tokenizer to use for tokenization.

### `--output_file`

* **Type:** `str`
* **Default:** `falcon-refinedweb.txt`
* Filename to save the downloaded raw text.

### `--split_prefix`

* **Type:** `str`
* **Default:** `split`
* Prefix for split files.
  Example: `split_0.txt`, `split_1.txt`, …

### `--tokenized_dir`

* **Type:** `str`
* **Default:** `tokenized_data`
* Directory where tokenized `.npy` files will be saved.

### `--dataset_col`

* **Type:** `str`
* **Default:** `content`
* Name of the dataset column containing the text.

---

## Example

Download 100k rows, split into chunks of 5k, and tokenize with LLaMA-3-70B tokenizer:

```bash
python setup_dataset.py \
  --dataset_name tiiuae/falcon-refinedweb \
  --max_rows 100000 \
  --batch_size 20000 \
  --split_examples 5000 \
  --tokenizer_model meta-llama/Meta-Llama-3-70B \
  --output_file data.txt \
  --split_prefix chunk \
  --tokenized_dir tokenized_out \
  --dataset_col text
```

---

## Workflow

1. **Download** the dataset from Hugging Face.
2. **Save raw text** into an output file.
3. **Split** large datasets into multiple smaller files.
4. **Tokenize** the text using the specified tokenizer.
5. **Save tokens** as `.npy` files for efficient training input.
