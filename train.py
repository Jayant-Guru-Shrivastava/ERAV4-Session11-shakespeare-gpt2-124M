# train.py
import os
import time
import torch

from model import GPTConfig, GPT
from data import DataLoaderLite

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # hyperparameters (tune if needed)
    B = 8
    T = 256
    learning_rate = 3e-4
    max_steps = 20000          # you can stop early when loss < target
    log_interval = 10
    target_loss = 0.099999

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # data
    train_loader = DataLoaderLite(
        filename="input.txt",
        B=B,
        T=T,
        device=device,
    )

    # model
    config = GPTConfig(
        block_size=T,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
    )
    model = GPT(config).to(device)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,  # no L2 regularization
    )

    # training loop
    step = 0
    best_loss = float("inf")
    start_time = time.time()

    with open("logs/train.log", "w") as f:
        f.write("step\tloss\n")

    while step < max_steps:
        model.train()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        step += 1
        loss_val = loss.item()
        best_loss = min(best_loss, loss_val)

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"step {step:6d} | loss {loss_val:.6f} | best {best_loss:.6f} | {elapsed:.1f}s")
            with open("logs/train.log", "a") as f:
                f.write(f"{step}\t{loss_val:.6f}\n")

        # early stop when we overfit enough
        if loss_val < target_loss:
            print(f"Target loss {target_loss} reached at step {step}, loss={loss_val:.6f}")
            break

    # save checkpoint
    ckpt_path = "checkpoints/shakespeare-gpt2-124M.pt"
    torch.save(model.state_dict(), ckpt_path)
    print("Saved checkpoint to", ckpt_path)

    # generate sample output
    print("\n=== SAMPLE GENERATION ===")
    import tiktoken
    enc = train_loader.enc  # reuse same encoder

    start_text = "ROMEO:"
    start_tokens = enc.encode(start_text)
    x = torch.tensor([start_tokens], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=200, temperature=0.8, top_k=50)

    out_tokens = out[0].tolist()
    decoded = enc.decode(out_tokens)
    print(decoded)

    # also save sample to file
    with open("sample_output.txt", "w", encoding="utf-8") as f:
        f.write(decoded)
    print("Saved sample_output.txt")


if __name__ == "__main__":
    main()
