from pathlib import Path

import torch

from data import decode, vocab_size
from model import ToyTransformerLM


MAX_NEW_TOKENS = 500
BLOCK_SIZE = 8
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "transformer.pt"


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("No checkpoint found. Run src/train.py first.")

    model = ToyTransformerLM(vocab_size, BLOCK_SIZE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    start_token = torch.zeros((1, 1), dtype=torch.long)

    with torch.no_grad():
        generated_tokens = model.generate(start_token, MAX_NEW_TOKENS)

    generated_text = decode(generated_tokens[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()
