#Created by ChatGPT

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

# Constants
MAX_LENGTH = 256
BATCH_SIZE = 16


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])  # Ensure it's an integer (0 or 1)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # (max_length,)
            'label': torch.tensor(label, dtype=torch.long)
        }

def prepare_imdb_dataloaders(csv_path, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
                              test_size=0.2, val_size=0.1, data_percentage=1.0):
    """
    Prepare IMDB dataloaders with optional data sampling percentage.

    Args:
        csv_path (str): Path to the CSV file.
        tokenizer: Tokenizer object with encode_plus method.
        batch_size (int): Batch size.
        max_length (int): Maximum sequence length.
        test_size (float): Fraction of data to reserve for testing.
        val_size (float): Fraction of data to reserve for validation.
        data_percentage (float): Percentage (0-1] of total data to use.

    Returns:
        Tuple of DataLoaders: (train_loader, val_loader, test_loader)
    """
    assert 0 < data_percentage <= 1.0, "data_percentage must be between 0 and 1."

    # Load CSV
    df = pd.read_csv(csv_path).dropna()

    # Sample the data
    if data_percentage < 1.0:
        df = df.sample(frac=data_percentage, random_state=42).reset_index(drop=True)

    # Split into train + temp (val + test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=(test_size + val_size),
        stratify=df['label'],
        random_state=42
    )

    # Further split temp into val and test
    val_relative_size = val_size / (val_size + test_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=(1 - val_relative_size),
        stratify=temp_labels,
        random_state=42
    )

    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

