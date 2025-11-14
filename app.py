# app.py
import torch
import tiktoken
import gradio as gr

from model import GPTConfig, GPT

device = "cpu"  # HF Spaces usually CPU

# model config (must match training)
config = GPTConfig(
    block_size=256,
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768,
)

model = GPT(config)
state_dict = torch.load("shakespeare-gpt2-124M.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

enc = tiktoken.get_encoding("gpt2")


def generate_text(prompt, max_new_tokens, temperature, top_k):
    if not prompt.strip():
        prompt = "ROMEO:"

    tokens = enc.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=int(top_k) if top_k > 0 else None,
        )

    out_tokens = out[0].tolist()
    text = enc.decode(out_tokens)
    return text


iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, label="Prompt", value="ROMEO:"),
        gr.Slider(50, 500, value=200, step=10, label="Max new tokens"),
        gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature"),
        gr.Slider(0, 100, value=50, step=5, label="Top-k (0 = none)"),
    ],
    outputs=gr.Textbox(lines=15, label="Generated text"),
    title="Shakespeare GPT-2 124M",
    description="Decoder-only 124M GPT-2 model trained to overfit on input.txt",
)

if __name__ == "__main__":
    iface.launch()
