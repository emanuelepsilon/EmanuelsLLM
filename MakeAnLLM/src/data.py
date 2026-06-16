from pathlib import Path

import torch


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "input.txt"


def load_text(path=DATA_PATH):
    """Read the whole text dataset into one Python string."""
    return path.read_text(encoding="utf-8")


text = load_text()
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for index, char in enumerate(chars)}


def encode(text_to_encode):
    """Turn a string into a list of integer token IDs."""
    token_ids = []

    for char in text_to_encode:
        if char not in stoi:
            raise ValueError(f"Character {char!r} is not in the vocabulary.")

        token_ids.append(stoi[char])

    return token_ids


def decode(token_ids):
    """Turn a list of integer token IDs back into a string."""
    chars_to_join = []

    for token_id in token_ids:
        if token_id not in itos:
            raise ValueError(f"Token ID {token_id!r} is not in the vocabulary.")

        chars_to_join.append(itos[token_id])

    return "".join(chars_to_join)


def get_batch(train_data, val_data, split, block_size, batch_size):
    """Create one batch of input tokens and target tokens."""
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        raise ValueError("split must be 'train' or 'val'.")

    data = torch.as_tensor(data, dtype=torch.long)

    if len(data) <= block_size:
        raise ValueError("data must be longer than block_size.")

    start_indexes = torch.randint(0, len(data) - block_size, (batch_size,))

    x = torch.stack([
        data[start:start + block_size]
        for start in start_indexes.tolist()
    ])
    y = torch.stack([
        data[start + 1:start + block_size + 1]
        for start in start_indexes.tolist()
    ])

    return x, y


def print_dataset_stats():
    """Print a small summary of the dataset and vocabulary."""
    print(f"Dataset file: {DATA_PATH}")
    print(f"Total characters: {len(text)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary characters: {repr(''.join(chars))}")


if __name__ == "__main__":
    print_dataset_stats()
