from pathlib import Path

import torch

from model import ToyTransformerLM


CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "transformer.pt"
MAX_NEW_TOKENS = 160
TEMPERATURE = 1.0
TOP_K = 1


def encode(prompt, stoi):
    """Turn a prompt string into token IDs using the saved vocabulary."""
    token_ids = []

    for char in prompt:
        if char not in stoi:
            raise ValueError(f"Character {char!r} is not in the saved vocabulary.")

        token_ids.append(stoi[char])

    return token_ids


def decode(token_ids, itos):
    """Turn generated token IDs back into text using the saved vocabulary."""
    return "".join(itos[token_id] for token_id in token_ids)


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("No checkpoint found. Run src/train.py first.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is old. Run src/train.py again to save the new format.")

    return checkpoint


def main():
    checkpoint = load_checkpoint()

    model = ToyTransformerLM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    print("Tiny character-level Q&A continuation tool")
    print("This is not a real instruction-following chatbot.")
    print("It continues text in the Question/Answer style it was trained on.")
    print("Type a prompt, or type exit to quit.")

    while True:
        prompt = input("\nprompt> ")

        if prompt.lower() == "exit":
            break

        formatted_prompt = f"Question: {prompt}\nAnswer:"

        try:
            prompt_ids = encode(formatted_prompt, stoi)
        except ValueError as error:
            print(error)
            continue

        if not prompt_ids:
            print("Please type at least one character.")
            continue

        idx = torch.tensor([prompt_ids], dtype=torch.long)

        with torch.no_grad():
            generated_ids = model.generate(
                idx,
                MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
            )

        generated_text = decode(generated_ids[0].tolist(), itos)
        answer = generated_text[len(formatted_prompt):]
        answer = answer.split("\n\nQuestion:")[0]
        print("\n" + answer.strip())


if __name__ == "__main__":
    main()
