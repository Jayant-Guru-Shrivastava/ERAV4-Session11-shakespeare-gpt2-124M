# Shakespeare GPT-2 (124M) — Overfit Assignment ✅

**Goal:** Train a **decoder‑only 124M** (GPT‑2 small) model on `input.txt` and achieve **loss < 0.099999**, then share:
- A **GitHub repo** with code, training logs, sample outputs, and a screenshot of the HuggingFace Space.
- A **HuggingFace Space** demo that runs the trained model.

This repo contains a minimal GPT‑2 small implementation (decoder‑only), a tiny dataloader over `input.txt`, and a Gradio app for HF Spaces.

---

## ✅ Result

- **Best training loss:** `~0.094433`
- **Reached target:** `loss < 0.099999`  
- **When:** at **step 4578**
- **Device:** GPU (Colab)
- **Tokens in dataset:** `~338,025`

Training excerpt:

```
...
step   4560 | loss 0.134991 | best 0.100690
step   4570 | loss 0.128361 | best 0.100096
Target loss 0.099999 reached at step 4578, loss=0.094433
Saved checkpoint to checkpoints/shakespeare-gpt2-124M.pt
```

Sample generation (truncated):

```
ROMEO:
Pl had my husband and thy do not his fault
And do thee forsware in Exll'd than easy
Of her, O,
If that way,
Since thou touch'd to't, for all his queen,
Had oft best
 brieve a groans,
Did not in loving lord, but one sweet king
At love my heir, nor any worse than the highness.
```

> Full sample saved in `sample_output.txt`.

---

## 📦 Repository Structure

```
.
├─ input.txt                          # training text (tokenized with tiktoken gpt2)
├─ model.py                           # GPT‑2 small (decoder‑only) implementation
├─ data.py                            # tiny sequential dataloader over input.txt
├─ train.py                           # trains until loss < 0.099999, logs & saves ckpt
├─ app.py                             # Gradio app for HuggingFace Spaces
├─ requirements.txt                   # torch, tiktoken, gradio
├─ logs/
│   └─ train.log                      # step\tloss log
├─ checkpoints/
│   └─ shakespeare-gpt2-124M.pt       # trained weights (NOT committed to GitHub)
└─ screenshots/
    └─ hf_space.png                   # screenshot of the running HF Space
```

> **Note:** GitHub has a 100MB per‑file limit. Do **not** commit the `.pt` weights to GitHub. Upload them to the **HuggingFace Space** instead.

---

## 🧠 Model

- **Architecture:** Decoder‑only Transformer (GPT‑2 small)
- **Parameters:** ~124M
- **Config:**
  - `n_layer = 12`
  - `n_head  = 12`
  - `n_embd  = 768`
  - `block_size = 256`
  - `vocab_size = 50257` (tiktoken *gpt2* encoding)
- **Tokenizer:** `tiktoken` (*gpt2*) — matches model vocab (50257)

---

## 📚 Dataset

- Single file: `input.txt` (Shakespeare‑style text supplied in the assignment)
- Tokens: ~**338,025**
- Tokenization: `tiktoken.get_encoding('gpt2')`

---

## ⚙️ Training Details

- **Batch size:** `B = 8`
- **Sequence length:** `T = 256`
- **Optimizer:** `AdamW(lr=1e-4, weight_decay=0.0)`
- **Dropout:** `0.0` (disabled to intentionally overfit)
- **Target loss:** `< 0.099999` (early‑stop)
- **Steps run:** Early stop at **~4578** when target was reached
- **Logging:** Every 10 steps to `logs/train.log`
- **Checkpoint:** Saved to `checkpoints/shakespeare-gpt2-124M.pt`

> We intentionally **removed regularization** (dropout = 0, weight_decay = 0) to allow the model to **memorize** the tiny dataset and reach a very low loss.

---

## ▶️ Reproduce Training (Local / Colab)

### Install
```bash
pip install torch tiktoken
```

### Run
```bash
python train.py
```

- The script will:
  - Load `input.txt`, tokenize with *tiktoken gpt2*
  - Train a GPT‑2 small (decoder‑only)
  - Log to `logs/train.log`
  - **Early‑stop** at loss `< 0.099999`
  - Save weights to `checkpoints/shakespeare-gpt2-124M.pt`
  - Generate a sample into `sample_output.txt`

> **GPU recommended.** On CPU it will be slow.

---

## 🌐 HuggingFace Space (Demo)

Create a new Space (SDK: **Gradio**) and upload:

- `model.py`
- `app.py`
- `requirements.txt`
- `shakespeare-gpt2-124M.pt`  ← **upload the trained checkpoint here**

Ensure `app.py` loads the exact filename:

```python
state_dict = torch.load("shakespeare-gpt2-124M.pt", map_location="cpu")
```

**Space URL:**  
`https://huggingface.co/spaces/<your-username>/<your-space-name>`  ← replace with your live link.

Add a screenshot of the running Space to the GitHub repo:
```
screenshots/hf_space.png
```

---

## 📑 Files You Should Commit to GitHub

- ✅ Code: `model.py`, `data.py`, `train.py`, `app.py`, `requirements.txt`
- ✅ Data stub: `input.txt` (if allowed by assignment)
- ✅ Logs: `logs/train.log`
- ✅ Samples: `sample_output.txt`
- ✅ Screenshot: `screenshots/hf_space.png`
- ✅ This `README.md`

**Do not commit**:
- ❌ `checkpoints/shakespeare-gpt2-124M.pt` (too large for GitHub; keep it in HF Space)

---

## 🛠️ Troubleshooting

- **Loss doesn’t go down:** make sure dropout is `0.0` and AdamW `weight_decay=0.0`. Let it run for enough steps (thousands). LR `1e-4` is a good long‑run value.
- **Weights not found in Space:** verify the filename in `app.py` matches the uploaded checkpoint.
- **Tokenizer mismatch:** ensure you use `tiktoken` *gpt2* with `vocab_size=50257` in the model.

---

## 📜 Acknowledgements

This project uses a minimal GPT‑2‑style decoder‑only architecture and the `tiktoken` *gpt2* tokenizer for simplicity and reproducibility.

---

## ✅ Submission Checklist

- [x] Loss `< 0.099999` achieved (`0.094433`)
- [x] GitHub repo with logs + sample outputs + screenshot + README
- [x] HF Space live demo with uploaded weights
