import glob
import torch
import numpy as np
from tqdm import tqdm
from trainer.dataloader.base_dataloader import BaseDataset


######################################
# Pretraining Dataloader...
# Dataset for Tokenized Data from Multiple Directories
######################################
class TokenizedDataset(BaseDataset):
    def __init__(self, tokenized_root, seq_length, eos_token=None, max_samples=10_000_000, stride=1):
        """
        Loads tokenized data from all .npy files under tokenized_root (recursively),
        concatenates them into one long token stream, and slices them into fixed-length sequences.
        Optionally, an EOS token is inserted between examples.
        The dataset length is capped by max_samples.
        """
        self.seq_length = seq_length
        self.stride = stride
        all_tokens = []
        file_list = glob.glob(os.path.join(tokenized_root, '**', '*.npy'), recursive=True)
        print(f"[TokenizedDataset] Found {len(file_list)} .npy files under {tokenized_root}")
        for file in tqdm(file_list, desc="Loading tokenized files"):
            arr = np.load(file, allow_pickle=True)
            for example in arr:
                all_tokens.extend(example)
                if eos_token is not None:
                    all_tokens.append(eos_token)
        self.tokens = all_tokens
        total = (len(self.tokens) - self.seq_length) // self.stride
        if total > max_samples:
            self.length = max_samples
            print(f"[TokenizedDataset] Capping dataset length to {self.length} samples (from {total}).")
        else:
            self.length = total
            print(f"[TokenizedDataset] Total samples: {self.length}")


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.stride
        x = torch.tensor(self.tokens[start:start+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.tokens[start+1:start+self.seq_length+1], dtype=torch.long)
        return x, y
