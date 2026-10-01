"""Project 2, step 1: look at every weight before writing any code.

Run from the repo root:   python -m scripts.ch2_inspect_weights

Prints layer 0 in full (every other layer is identical), the non-layer
weights, and the total. Your job: reproduce the total BY HAND in
notes/ch2.md using section 2.8, before you look at the last line.
"""
import torch
from transformers import AutoModelForCausalLM

from tinyserve.model_loader import MODEL_ID

m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
sd = m.state_dict()

print("=== config ===")
c = m.config
for k in ["hidden_size", "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "intermediate_size", "vocab_size",
          "rms_norm_eps", "tie_word_embeddings"]:
    print(f"{k:22} {getattr(c, k)}")

print("\n=== layer 0 (all 24 layers look like this) ===")
layer0 = 0
for name, t in sd.items():
    if name.startswith("model.layers.0."):
        print(f"{name:45} {str(tuple(t.shape)):18} {t.numel():>12,}")
        layer0 += t.numel()
print(f"{'layer 0 total':45} {'':18} {layer0:>12,}")

print("\n=== outside the layers ===")
for name, t in sd.items():
    if not name.startswith("model.layers."):
        print(f"{name:45} {str(tuple(t.shape)):18} {t.numel():>12,}")

# parameters() de-duplicates tied tensors; state_dict() may list lm_head too
total = sum(p.numel() for p in m.parameters())
print(f"\nunique parameters: {total:,}")
print("is lm_head the same tensor as embed_tokens?",
      m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr())
