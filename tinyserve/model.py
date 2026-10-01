"""Project 2: your own Qwen2.5 forward pass, in plain PyTorch.

Hugging Face is used ONLY to download the weights. Every piece of math the
model does is written by you in this file.

Order to work in (each has its own test in tests/test_model.py):

    1. rms_norm          section 2.3     pytest -k rms
    2. rope_cos_sin      section 2.6     pytest -k rope
       apply_rope
    3. mlp               section 2.7     pytest -k mlp
    4. attention         sections 2.4-2.5
       block             section 2.2     pytest -k layer
    5. forward                           pytest -k logits
    6. KVCache.write/read                pytest -k cache

Conventions used everywhere in this file (batch size is always 1, so there
is no batch dimension at all):

    ids      shape [T]                  token ids
    x        shape [T, d]               one row per token
    q        shape [h, T, d_head]       heads first
    k, v     shape [n_kv, T, d_head]

nn.Linear stores its weight as [out_features, in_features]. So a linear
layer is   y = x @ W.T + b   (note the .T). Forgetting it is trap #1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# Config and weights (done for you: plumbing)
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    d: int          # hidden size
    L: int          # layers
    h: int          # query heads
    n_kv: int       # key/value heads
    d_head: int     # size of one head
    d_ff: int       # MLP inner size
    V: int          # vocabulary size
    eps: float      # RMSNorm epsilon
    rope_theta: float

    @classmethod
    def from_hf(cls, c) -> "Config":
        d_head = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        theta = getattr(c, "rope_theta", None)
        if theta is None:  # newer transformers keep it in rope_parameters
            theta = (getattr(c, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
        return cls(
            d=c.hidden_size,
            L=c.num_hidden_layers,
            h=c.num_attention_heads,
            n_kv=c.num_key_value_heads,
            d_head=d_head,
            d_ff=c.intermediate_size,
            V=c.vocab_size,
            eps=c.rms_norm_eps,
            rope_theta=float(theta),
        )


def load_weights(hf_model) -> dict[str, torch.Tensor]:
    """Copy the HF weights into a flat dict with short names.

    Per layer i you get:
        f"{i}.attn_norm"               [d]
        f"{i}.q_w" [h*d_head, d]       f"{i}.q_b" [h*d_head]
        f"{i}.k_w" [n_kv*d_head, d]    f"{i}.k_b" [n_kv*d_head]
        f"{i}.v_w" [n_kv*d_head, d]    f"{i}.v_b" [n_kv*d_head]
        f"{i}.o_w" [d, h*d_head]       (no bias on o_proj)
        f"{i}.mlp_norm"                [d]
        f"{i}.gate" [d_ff, d]   f"{i}.up" [d_ff, d]   f"{i}.down" [d, d_ff]
    Plus: "embed" [V, d], "final_norm" [d], "lm_head" [V, d] (same tensor as
    embed for Qwen2.5-0.5B, because the weights are tied).
    """
    sd = hf_model.state_dict()
    n_layers = hf_model.config.num_hidden_layers
    W: dict[str, torch.Tensor] = {
        "embed": sd["model.embed_tokens.weight"],
        "final_norm": sd["model.norm.weight"],
    }
    W["lm_head"] = sd.get("lm_head.weight", W["embed"])
    for i in range(n_layers):
        p = f"model.layers.{i}."
        W[f"{i}.attn_norm"] = sd[p + "input_layernorm.weight"]
        for name in ("q", "k", "v"):
            W[f"{i}.{name}_w"] = sd[p + f"self_attn.{name}_proj.weight"]
            W[f"{i}.{name}_b"] = sd[p + f"self_attn.{name}_proj.bias"]
        W[f"{i}.o_w"] = sd[p + "self_attn.o_proj.weight"]
        W[f"{i}.mlp_norm"] = sd[p + "post_attention_layernorm.weight"]
        W[f"{i}.gate"] = sd[p + "mlp.gate_proj.weight"]
        W[f"{i}.up"] = sd[p + "mlp.up_proj.weight"]
        W[f"{i}.down"] = sd[p + "mlp.down_proj.weight"]
    return W


# --------------------------------------------------------------------------- #
# 1. RMSNorm (section 2.3)
# --------------------------------------------------------------------------- #
def rms_norm(x: torch.Tensor, g: torch.Tensor, eps: float) -> torch.Tensor:
    """x: [..., d], g: [d]  ->  same shape as x.

    RMSNorm(x) = ( x / sqrt(mean(x^2 over the last dim) + eps) ) * g

    Do the math in float32 and cast back to x's dtype at the end, then
    multiply by g. Worked check: x=[3,4], g=[1,1], eps=0 -> [0.849, 1.131].
    Hint: mean over the last dim with keepdim=True, and torch.rsqrt.
    """
    Dtype = x.dtype
    x = x.float()
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(mean_sq + eps)
    return (x * g).to(Dtype)


# --------------------------------------------------------------------------- #
# 2. RoPE (section 2.6)
# --------------------------------------------------------------------------- #
def rope_cos_sin(
    positions: torch.Tensor, d_head: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """positions: [T] (integer positions)  ->  (cos, sin), each [T, d_head].

    Steps:
      inv_freq[i] = 1 / theta^(2i / d_head)     for i = 0 .. d_head/2 - 1
      angles      = positions (as a column) * inv_freq (as a row)  -> [T, d_head/2]
      angles      = concatenate(angles, angles) on the last dim     -> [T, d_head]
      return cos(angles), sin(angles)

    Why the concatenate: HF uses the "split-half" layout. Dimension j is
    paired with dimension j + d_head/2 (not with j+1), so both halves need
    the same angles. Hint: torch.arange(0, d_head, 2) / d_head gives 2i/d_head.
    """
    i = torch.arange(0, d_head, 2).float()
    inv_freq = 1.0 / (theta ** (i / d_head))
    angles = positions.float()[:, None] * inv_freq[None, :]   # [T, d_head/2]
    angles = torch.cat([angles, angles], dim=-1)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [n_heads, T, d_head], cos/sin: [T, d_head]  ->  same shape as x.

    Split x in half along the last dim: x1 = first half, x2 = second half.
    rotate_half(x) = concatenate(-x2, x1)
    return x * cos + rotate_half(x) * sin

    (cos and sin broadcast over the heads dimension automatically.)
    Check: at position 0, cos=1 and sin=0, so x must come back unchanged.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# --------------------------------------------------------------------------- #
# 3. MLP (section 2.7)
# --------------------------------------------------------------------------- #
def mlp(x: torch.Tensor, W: dict, i: int) -> torch.Tensor:
    """x: [T, d]  ->  [T, d].   Layer i's SwiGLU MLP:

        down( silu(x @ gate.T) * (x @ up.T) )

    Weights: W[f"{i}.gate"], W[f"{i}.up"], W[f"{i}.down"]. No biases.
    torch.nn.functional.silu exists; use it.
    """
    gate = x @ W[f"{i}.gate"].T
    up = x @ W[f"{i}.up"].T
    h = torch.nn.functional.silu(gate) * up
    return h @ W[f"{i}.down"].T

# --------------------------------------------------------------------------- #
# 4. Attention and the block (sections 2.2, 2.4, 2.5)
# --------------------------------------------------------------------------- #
def attention(
    x: torch.Tensor,
    W: dict,
    i: int,
    cfg: Config,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache: "KVCache | None" = None,
    start_pos: int = 0,
) -> torch.Tensor:
    """x: [T, d] (already normalized)  ->  [T, d].

    The T tokens in x sit at positions start_pos .. start_pos + T - 1.
    cos/sin are already computed for exactly those positions.

    Steps:
      1. q = x @ q_w.T + q_b  -> [T, h*d_head] -> view [T, h, d_head]
                                -> transpose to [h, T, d_head]
         k, v the same with n_kv heads          -> [n_kv, T, d_head]
      2. q = apply_rope(q, cos, sin);  k = apply_rope(k, cos, sin)
         (v is NOT rotated)
      3. If cache is not None:
             cache.write(i, start_pos, k, v)
             k, v = cache.read(i, start_pos + T)     -> [n_kv, S, d_head]
         where S = start_pos + T is the total number of tokens so far.
         (If cache is None, S = T and k, v stay as they are.)
      4. GQA: each KV head serves h // n_kv query heads.
             k = k.repeat_interleave(h // n_kv, dim=0)   -> [h, S, d_head]
             v the same
      5. scores = q @ k.transpose(-1, -2) / sqrt(d_head)  -> [h, T, S]
      6. Causal mask. Query row t is at position start_pos + t; key column
         s is at position s. Block (set to -inf) wherever s > start_pos + t.
         Build a [T, S] boolean mask and use scores.masked_fill(mask, -inf).
      7. probs = softmax(scores in float32, dim=-1), cast back to v's dtype
      8. out = probs @ v -> [h, T, d_head] -> transpose to [T, h, d_head]
                         -> reshape to [T, h*d_head]
      9. return out @ o_w.T      (no bias)

    Write the softmax attention by hand (no F.scaled_dot_product_attention):
    this is the one function in the book you should be able to write on a
    whiteboard.
    """
    T = x.shape[0]
    q = x @ W[f"{i}.q_w"].T + W[f"{i}.q_b"]
    k = x @ W[f"{i}.k_w"].T + W[f"{i}.k_b"]
    v = x @ W[f"{i}.v_w"].T + W[f"{i}.v_b"]
    q = q.view(T, cfg.h, cfg.d_head).transpose(0, 1)  # [h, T, d_head]
    k = k.view(T, cfg.n_kv, cfg.d_head).transpose(0, 1)  # [n_kv, T, d_head]
    v = v.view(T, cfg.n_kv, cfg.d_head).transpose(0, 1)  # [n_kv, T, d_head]
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    if cache is not None:
        cache.write(i, start_pos, k, v)
        k, v = cache.read(i, start_pos + T) 
    group = cfg.h // cfg.n_kv                   # 14 // 2 = 7
    k = k.repeat_interleave(group, dim=0)       # [14, S, 64]
    v = v.repeat_interleave(group, dim=0)       # [14, S, 64]

    # 6. scores: every query against every key, per head
    scores = q @ k.transpose(-1, -2) / math.sqrt(cfg.d_head)    # [14, T, S]

    # 7. causal mask: block keys that come after the query
    S = k.shape[1]
    q_pos = start_pos + torch.arange(T, device=x.device)        # [T]
    k_pos = torch.arange(S, device=x.device)                    # [S]
    mask = k_pos[None, :] > q_pos[:, None]                       # [T, S], True = blocked
    scores = scores.masked_fill(mask, float("-inf"))  # [14, T, S]
    # 8. softmax over the keys, in float32
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)

    # 9. blend values, glue heads back, output projection
    out = probs @ v                                              # [14, T, 64]
    out = out.transpose(0, 1).reshape(T, cfg.h * cfg.d_head)     # [T, 896]
    return out @ W[f"{i}.o_w"].T
def block(
    x: torch.Tensor,
    W: dict,
    i: int,
    cfg: Config,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache: "KVCache | None" = None,
    start_pos: int = 0,
) -> torch.Tensor:
    """One transformer layer, section 2.2:

        x = x + attention( rms_norm(x, attn_norm) )
        x = x + mlp( rms_norm(x, mlp_norm) )
    """
    # TODO
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# 5. The whole model
# --------------------------------------------------------------------------- #
def forward(
    ids: torch.Tensor,
    W: dict,
    cfg: Config,
    cache: "KVCache | None" = None,
    start_pos: int = 0,
    return_hidden: bool = False,
):
    """ids: [T]  ->  logits [T, V]   (and a list of hidden states if asked).

    Steps:
      1. x = W["embed"][ids]                                -> [T, d]
      2. positions = arange(start_pos, start_pos + T)
         cos, sin = rope_cos_sin(positions, d_head, rope_theta)
         (compute once, reuse in every layer)
      3. hidden = [x];  for each layer i: x = block(...); hidden.append(x)
      4. x = rms_norm(x, W["final_norm"], eps)
      5. logits = x @ W["lm_head"].T                          -> [T, V]
      Return logits, or (logits, hidden) if return_hidden.

    hidden has L + 1 entries: the embeddings, then the output of each layer.
    The layer-by-layer test compares it with Hugging Face's hidden states,
    so when something is wrong it tells you the first layer that drifts.
    """
    # TODO
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# 6. Preallocated KV cache (Project 2, step 4)
# --------------------------------------------------------------------------- #
class KVCache:
    """One big tensor allocated up front:

        data: [L, 2, n_kv, max_len, d_head]     (index 0 = K, 1 = V)

    Heads come before positions so that a read returns [n_kv, S, d_head],
    exactly the shape attention() works with, with no transposes.
    (The book writes the shape as [L, 2, max_len, n_kv, d_head]; same
    numbers, different order.)
    """

    def __init__(self, cfg: Config, max_len: int, dtype: torch.dtype, device) -> None:
        self.max_len = max_len
        self.data = torch.zeros(
            cfg.L, 2, cfg.n_kv, max_len, cfg.d_head, dtype=dtype, device=device
        )

    def write(self, layer: int, start_pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: [n_kv, T, d_head]. Store them at positions start_pos .. start_pos+T-1
        of this layer. One slice assignment each."""
        # TODO
        raise NotImplementedError

    def read(self, layer: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (k, v) for positions 0 .. end-1 of this layer, each
        [n_kv, end, d_head]. Slices, not copies."""
        # TODO
        raise NotImplementedError

    @property
    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()


# --------------------------------------------------------------------------- #
# Greedy generation with your model (done for you: same pattern as Project 1)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_generate(
    ids: torch.Tensor,
    W: dict,
    cfg: Config,
    max_new_tokens: int,
    use_cache: bool = True,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """ids: [T] -> [T + n]. Greedy only; this is for correctness tests."""
    ids = ids.clone()

    if not use_cache:
        for _ in range(max_new_tokens):
            nxt = forward(ids, W, cfg)[-1].argmax()
            ids = torch.cat([ids, nxt.view(1)])
            if eos_token_id is not None and nxt.item() == eos_token_id:
                break
        return ids

    cache = KVCache(cfg, len(ids) + max_new_tokens, W["embed"].dtype, ids.device)
    logits = forward(ids, W, cfg, cache=cache, start_pos=0)          # prefill
    pos = len(ids)
    nxt = logits[-1].argmax()
    ids = torch.cat([ids, nxt.view(1)])
    for _ in range(max_new_tokens - 1):                                # decode
        if eos_token_id is not None and nxt.item() == eos_token_id:
            break
        logits = forward(nxt.view(1), W, cfg, cache=cache, start_pos=pos)
        pos += 1
        nxt = logits[-1].argmax()
        ids = torch.cat([ids, nxt.view(1)])
    return ids
