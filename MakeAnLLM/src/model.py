import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One self-attention head."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        batch_size, time_steps, channels = x.shape

        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores * keys.shape[-1] ** -0.5

        attention_scores = attention_scores.masked_fill(
            self.tril[:time_steps, :time_steps] == 0,
            float("-inf"),
        )
        attention_weights = F.softmax(attention_scores, dim=-1)

        out = attention_weights @ values
        return out


class MultiHeadAttention(nn.Module):
    """Several self-attention heads running in parallel."""

    def __init__(self, n_embd, num_heads, head_size, block_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        head_outputs = [
            head(x)
            for head in self.heads
        ]
        concatenated = torch.cat(head_outputs, dim=-1)
        out = self.projection(concatenated)
        return out


class FeedForward(nn.Module):
    """A small neural network applied to each token position."""

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer block: communication, thinking, and skip paths."""

    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads.")

        head_size = n_embd // num_heads

        self.self_attention = MultiHeadAttention(
            n_embd,
            num_heads,
            head_size,
            block_size,
        )
        self.feed_forward = FeedForward(n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class ToyTransformerLM(nn.Module):
    """A small character-level transformer language model."""

    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd=32,
        num_heads=4,
        num_blocks=2,
    ):
        super().__init__()

        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, num_heads, block_size)
            for _ in range(num_blocks)
        ])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, time_steps = idx.shape

        if time_steps > self.block_size:
            raise ValueError("Input sequence is longer than block_size.")

        token_embeddings = self.token_embedding_table(idx)
        positions = torch.arange(time_steps, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch_size, time_steps, vocab_size = logits.shape
            logits_for_loss = logits.view(batch_size * time_steps, vocab_size)
            targets_for_loss = targets.view(batch_size * time_steps)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.block_size:]
            logits, loss = self(idx_context)
            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                top_values, top_indexes = torch.topk(logits, top_k)
                filtered_logits = torch.full_like(logits, float("-inf"))
                logits = filtered_logits.scatter(1, top_indexes, top_values)

            probabilities = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx
