# data.py
import torch
import tiktoken


class DataLoaderLite:
    """
    Very simple sequential dataloader for language modeling.
    We intentionally make it easy to overfit.
    """

    def __init__(self, filename="input.txt", B=8, T=256, device="cuda"):
        self.device = device
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding("gpt2")
        self.enc = enc

        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long, device=device)

        # how many full B*T blocks we can get (minus one for targets)
        self.n_batches = (len(self.tokens) - 1) // (B * T)
        self.i = 0

        print(f"Loaded {len(self.tokens)} tokens from {filename}")
        print(f"With B={B}, T={T}, we have {self.n_batches} batches per epoch")

    def next_batch(self):
        if self.i >= self.n_batches:
            self.i = 0

        start = self.i * self.B * self.T
        end = start + self.B * self.T + 1
        chunk = self.tokens[start:end]  # length B*T + 1

        x = chunk[:-1].view(self.B, self.T)
        y = chunk[1:].view(self.B, self.T)

        self.i += 1
        return x, y
