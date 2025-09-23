import os
import glob
import numpy as np

def count_total_tokens(tokenized_dir):
    """
    Calculates the total number of tokens in all tokenized .npy files within the specified directory.

    Args:
        tokenized_dir (str): The directory where tokenized .npy files are stored.

    Returns:
        int: Total number of tokens across all examples.
    """
    total_tokens = 0

    # Look for all .npy files in the tokenized directory.
    npy_files = glob.glob(os.path.join(tokenized_dir, "*.npy"))
    for file_path in npy_files:
        # Load the numpy array (allow_pickle=True for object arrays)
        tokenized_array = np.load(file_path, allow_pickle=True)
        # Each element is assumed to be a list of token ids.
        for tokens in tokenized_array:
            total_tokens += len(tokens)
    
    print(f"Total number of tokens: {total_tokens}")
    return total_tokens

def count_total_words(segments_dir):
    """
    Calculates the total number of words in all .txt files within the specified segments directory.

    Args:
        segments_dir (str): The directory where segment .txt files are stored.

    Returns:
        int: Total number of words across all files.
    """
    total_words = 0

    # Look for all .txt files in the segments directory.
    txt_files = glob.glob(os.path.join(segments_dir, "*.txt"))
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            # Read the entire file content.
            text = f.read()
            # Split the text by whitespace to get words.
            words = text.split()
            total_words += len(words)
    
    print(f"Total number of words: {total_words}")
    return total_words

if __name__ == "__main__":
    # Adjust these paths if needed.
    tokenized_directory = os.path.join(os.getcwd(), "src", "data", "tokenized")
    segments_directory = os.path.join(os.getcwd(), "src", "data", "segments")
    
    print("Counting tokens in tokenized data...")
    count_total_tokens(tokenized_directory)
    
    print("\nCounting words in segments data...")
    count_total_words(segments_directory)
