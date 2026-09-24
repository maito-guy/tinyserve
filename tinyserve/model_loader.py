"""Load the reference model and tokenizer.

This file is deliberately finished for you: it is plumbing, not learning.
Everything in generate.py you write yourself.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    # T4 has no fast bf16; use fp16 on CUDA, fp32 elsewhere.
    return torch.float16 if device.type == "cuda" else torch.float32


def load(model_id: str = MODEL_ID):
    """Return (model, tokenizer, device). Model is in eval mode, no grad."""
    device = pick_device()
    dtype = pick_dtype(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, device


def chat_prompt(tok, user_message: str, system: str | None = None) -> str:
    """Apply the model's chat template and return the raw string.

    Print the result once and look at the special tokens: that is step 3
    of Project 1.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
