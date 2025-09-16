#!/usr/bin/env python
import os
import io
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import glob
import concurrent.futures


def sanitize_dataset_name(dataset_name):
    # Replace characters that are not valid in file names.
    return dataset_name.replace("/", "_")

def download_dataset(dataset_name, output_filename, max_rows, batch_size, flag, column):
    """
    Downloads up to `max_rows` examples from a streaming Hugging Face dataset and writes each example 
    to a text file (with each example ending with a delimiter) in batches.
    If flag is 'y', then the downloads directory is cleared before writing.
    """
    save_path = './trainer/data/downloads/'
    output_file_path = os.path.join(save_path, output_filename)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if flag.lower() == 'y':
        print('deleting file: ', output_file_path)
        os.remove(output_file_path)

    # Load the dataset in streaming mode.
    ds = load_dataset(dataset_name, split="train", streaming=True)
    
    texts = []            # Buffer for collected examples.
    buffer = io.StringIO()  # In-memory text buffer.
    
    # Create or clear the output file.
    with open(output_file_path, "w", encoding="utf-8") as f:
        pass

    # Iterate over the dataset and write examples in batches.
    for item in tqdm(ds.take(max_rows), total=max_rows, desc="Downloading dataset"):
        text = item[column].strip()
        if not text:
            continue
        texts.append(text)
        
        if len(texts) >= batch_size:
            # Append each example with its delimiter.
            buffer.write("\n".join([ex + "\n<|endofexample|>" for ex in texts]) + "\n")
            with open(output_file_path, "a", encoding="utf-8") as f:
                f.write(buffer.getvalue())
            texts = []
            buffer.truncate(0)
            buffer.seek(0)
    
    # Write any remaining examples.
    if texts:
        buffer.write("\n".join([ex + "\n<|endofexample|>" for ex in texts]) + "\n")
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(buffer.getvalue())
    buffer.close()
    print(f"Downloaded dataset saved to '{output_file_path}'.")

def split_file(input_filename, split_prefix, split_examples, flag, dataset_name):
    """
    Splits the file `input_filename` into multiple files, each containing up to `split_examples` full examples.
    The output files are stored in a dataset-specific directory under data/segments.
    If flag is 'y', the dataset-specific segments directory is cleared first.
    Returns the list of split file paths.
    """
    split_files = []
    example_counter = 0
    file_counter = 1

    dsafe = sanitize_dataset_name(dataset_name)
    segments_dir = os.path.join(os.getcwd(), "trainer", "data", "segments", dsafe)
    
    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)
    else:
        if flag.lower() == 'y':
            for file_path in glob.glob(os.path.join(segments_dir, '*')):
                os.remove(file_path)
    
    current_filename = os.path.join(segments_dir, f"{split_prefix}_{file_counter}.txt")
    out_file = open(current_filename, "w", encoding="utf-8")
    split_files.append(current_filename)

    # Read the entire input file and split by the delimiter.
    with open(input_filename, "r", encoding="utf-8") as infile:
        data = infile.read()
    
    # Split on the delimiter and filter out any empty examples.
    examples = [ex.strip() for ex in data.split("<|endofexample|>") if ex.strip()]
    
    # Write full examples into output files.
    for ex in tqdm(examples, desc="Splitting file by examples"):
        out_file.write(ex + "\n<|endofexample|>\n")
        example_counter += 1

        if example_counter >= split_examples:
            out_file.close()
            file_counter += 1
            example_counter = 0
            current_filename = os.path.join(segments_dir, f"{split_prefix}_{file_counter}.txt")
            out_file = open(current_filename, "w", encoding="utf-8")
            split_files.append(current_filename)
    
    out_file.close()
    print(f"File '{input_filename}' split into {len(split_files)} files in directory '{segments_dir}'.")
    return split_files

def process_split_file(split_file, tokenized_data_dir, tokenizer_model):
    """
    Worker function: tokenizes the examples in a single split file and saves
    the tokenized output as a NumPy array.
    """
    # Each worker loads its own tokenizer instance.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, token="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")
    tokenized_examples = []
    with open(split_file, "r", encoding="utf-8") as f:
        data = f.read()
    # Split on the delimiter and remove any empty examples.
    examples = [ex.strip() for ex in data.split("<|endofexample|>") if ex.strip()]
    
    # (Optional: you can wrap the loop in tqdm if you want progress on each file,
    # but in parallel that may mix output from multiple processes.)
    for ex in examples:
        tokens = tokenizer.encode(ex, add_special_tokens=True)
        tokenized_examples.append(tokens)
    
    tokenized_array = np.array(tokenized_examples, dtype=object)
    base_name = os.path.basename(split_file)
    np_filename = os.path.join(tokenized_data_dir, base_name.replace(".txt", ".npy"))
    np.save(np_filename, tokenized_array)
    return np_filename

def tokenize_files(split_files, tokenized_data_dir, tokenizer_model, flag, dataset_name):
    """
    For each file in `split_files`, tokenizes the full examples (separated by the delimiter)
    using the specified tokenizer and saves the tokenized outputs as NumPy arrays.
    The tokenized files are stored in a dataset-specific directory under data/tokenized.
    If flag is 'y', the directory is cleared before writing.
    This version processes the files in parallel.
    """
    dsafe = sanitize_dataset_name(dataset_name)
    tokenized_data_dir = os.path.join(os.getcwd(), "trainer", "data", "tokenized", dsafe)
    
    if not os.path.exists(tokenized_data_dir):
        os.makedirs(tokenized_data_dir)
    else:
        if flag.lower() == 'y':
            for file_path in glob.glob(os.path.join(tokenized_data_dir, '*')):
                os.remove(file_path)
    
    # Use a process pool to parallelize tokenization over split_files.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks to the executor.
        futures = {executor.submit(process_split_file, sf, tokenized_data_dir, tokenizer_model): sf for sf in split_files}
        # Collect results as they complete.
        for future in concurrent.futures.as_completed(futures):
            try:
                np_filename = future.result()
                print(f"Tokenized data saved to '{np_filename}'.")
            except Exception as exc:
                sf = futures[future]
                print(f"Tokenization generated an exception for file {sf}: {exc}")

def main():
    parser = argparse.ArgumentParser(description="Pretraining dataset setup: download, split, and tokenize.")
    parser.add_argument("--dataset_name", type=str, default="tiiuae/falcon-refinedweb",
                        help="Hugging Face dataset name.")
    parser.add_argument("--max_rows", type=int, default=50_000,
                        help="Maximum number of rows to download from the dataset.")
    parser.add_argument("--batch_size", type=int, default=10_000,
                        help="Number of rows to buffer before writing to disk.")
    parser.add_argument("--split_examples", type=int, default=10000,
                        help="Maximum number of examples per split file.")
    parser.add_argument("--tokenizer_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Tokenizer model to use (from Hugging Face).")
    parser.add_argument("--output_file", type=str, default="falcon-refinedweb.txt",
                        help="Output filename for the downloaded text.")
    parser.add_argument("--split_prefix", type=str, default="split",
                        help="Prefix for the split text files.")
    parser.add_argument("--tokenized_dir", type=str, default="tokenized_data",
                        help="Directory to save tokenized .npy files.")
    parser.add_argument("--dataset_col", type=str, default="content",
                        help="Column name having text in the dataset.")

    args = parser.parse_args()
    

    dsafe = sanitize_dataset_name(args.dataset_name)
    # Ask the user whether to delete existing data for each step.
        # Accept data loader settings
    flag_download = input(f"Do you want to download data? (y/n): ")
    if flag_download.lower() == 'y':
        flag_download_del = input(f"Do you want to delete the existing file at data/downloads/{dsafe}.txt? (y/n): ")
    flag_split = input(f"Do you want to split downloaded data? (y/n): ")
    if flag_split.lower() == 'y':
        flag_split_del = input(f"Do you want to delete the existing files at data/segments/{dsafe}? (y/n): ")
    flag_tokenize = input(f"Do you want to tokenize segmented data? (y/n): ")
    if flag_tokenize.lower() == 'y':
        flag_tok_del = input(f"Do you want to delete the existing tokenized data at data/tokenized/{dsafe}? (y/n): ")

    
    # Step 1: Download dataset.
    if flag_download.lower() == 'y':
        print("Step 1: Downloading dataset...")
        download_dataset(args.dataset_name, f"{dsafe}.txt", args.max_rows, args.batch_size, flag_download_del, args.dataset_col)
    
    # Step 2: Split the downloaded file.
    if flag_split.lower() == 'y':
        print("\nStep 2: Splitting the file...")
        download_path = os.path.join("./trainer/data/downloads/", f"{dsafe}.txt")
        split_files = split_file(download_path, args.split_prefix, args.split_examples, flag_split_del, args.dataset_name)
    
    # Step 3: Tokenize each split file.
    if flag_tokenize.lower() == 'y':
        print("\nStep 3: Tokenizing split files...")
        tokenize_files(split_files, args.tokenized_dir, args.tokenizer_model, flag_tok_del, args.dataset_name)
    
    print("\nPretraining setup complete!")

if __name__ == '__main__':
    main()

#python -m trainer.data.loader --max_rows 1000000  --split_examples 10000
# python -m trainer.data.loader --max_rows 600000  --split_examples 10000 --dataset_name cerebras/SlimPajama-627B --dataset_col text
#python -m trainer.data.loader --max_rows 1000000  --split_examples 10000 --dataset_name bigscience-data/roots_en_uncorpus