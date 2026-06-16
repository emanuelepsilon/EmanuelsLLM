from pathlib import Path

import torch

from data import encode, get_batch, itos, load_text, stoi, vocab_size
from model import ToyTransformerLM


BLOCK_SIZE = 64
BATCH_SIZE = 16
MAX_STEPS = 2000
EVAL_EVERY = 200
EVAL_STEPS = 20
LEARNING_RATE = 3e-4
N_EMBD = 64
NUM_HEADS = 4
NUM_BLOCKS = 4

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "transformer.pt"


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate train and validation loss without updating the model."""
    losses = {}
    model.eval()

    for split in ["train", "val"]:
        split_losses = []

        for _ in range(EVAL_STEPS):
            x, y = get_batch(train_data, val_data, split, BLOCK_SIZE, BATCH_SIZE)
            logits, loss = model(x, y)
            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def main():
    torch.manual_seed(1337)

    text = load_text()
    token_ids = torch.tensor(encode(text), dtype=torch.long)

    split_index = int(0.9 * len(token_ids))
    train_data = token_ids[:split_index]
    val_data = token_ids[split_index:]

    model_config = {
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "num_heads": NUM_HEADS,
        "num_blocks": NUM_BLOCKS,
    }

    model = ToyTransformerLM(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_STEPS + 1):
        if step % EVAL_EVERY == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        x, y = get_batch(train_data, val_data, "train", BLOCK_SIZE, BATCH_SIZE)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Saved model to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
