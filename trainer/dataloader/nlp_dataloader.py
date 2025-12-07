import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizer


class IMDBDataset(Dataset):
    """
    Wrapper for HuggingFace IMDB dataset with BERT tokenization.

    This dataset loads the IMDB sentiment classification dataset and tokenizes
    the text using BERT tokenizer.

    Args:
        split: Dataset split ('train' or 'test')
        max_length: Maximum sequence length for tokenization (default: 128)
        subset_size: Optional limit on dataset size for testing/debugging
    """

    def __init__(self, split='train', max_length=128, subset_size=None):
        print(f"Loading {split} dataset...")
        self.dataset = load_dataset('stanfordnlp/imdb', split=split)

        # Use subset if specified
        if subset_size is not None:
            indices = list(range(min(subset_size, len(self.dataset))))
            self.dataset = self.dataset.select(indices)
            print(f"Using subset of {len(self.dataset)} samples")

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
