"""Project 2 tests. Run in the order of the TODOs:

    pytest tests/test_model.py -v -k rms      # 1
    pytest tests/test_model.py -v -k rope     # 2
    pytest tests/test_model.py -v -k mlp      # 3
    pytest tests/test_model.py -v -k layer    # 4  (attention + block)
    pytest tests/test_model.py -v -k logits   # 5
    pytest tests/test_model.py -v -k cache    # 6

Everything runs on CPU in float32, so it works on a laptop without a GPU.
Tests marked "no model" run instantly; the others load Qwen2.5-0.5B (~2 GB
of RAM in float32) once and reuse it.
"""

import math

import pytest
import torch

from tinyserve import model as M
from tinyserve.model_loader import MODEL_ID

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Shared fixtures: the HF model in float32 on CPU, and your weights/config
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def hf():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
    )
    return m.eval()


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def mine(hf):
    return M.load_weights(hf), M.Config.from_hf(hf.config)


PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "Inference is the phase after training where the weights are frozen.",
]


def _ids(tok, text):
    return tok(text, return_tensors="pt").input_ids[0]


# --------------------------------------------------------------------------- #
# 0. Config sanity (passes immediately; compare with the table in 2.1)
# --------------------------------------------------------------------------- #
def test_config_matches_section_2_1(mine):
    _, cfg = mine
    assert (cfg.d, cfg.L, cfg.h, cfg.n_kv, cfg.d_head, cfg.d_ff, cfg.V) == (
        896, 24, 14, 2, 64, 4864, 151936,
    )


# --------------------------------------------------------------------------- #
# 1. RMSNorm
# --------------------------------------------------------------------------- #
def test_rms_worked_example():  # no model
    out = M.rms_norm(torch.tensor([3.0, 4.0]), torch.tensor([1.0, 1.0]), eps=0.0)
    assert torch.allclose(out, torch.tensor([0.8485, 1.1314]), atol=1e-3)


def test_rms_matches_hf(hf, mine):
    W, cfg = mine
    x = torch.randn(5, cfg.d) * 3
    with torch.no_grad():
        ref = hf.model.norm(x)
    assert torch.allclose(M.rms_norm(x, W["final_norm"], cfg.eps), ref, atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. RoPE
# --------------------------------------------------------------------------- #
def test_rope_shapes():  # no model
    cos, sin = M.rope_cos_sin(torch.arange(10), 64, 1_000_000.0)
    assert cos.shape == (10, 64) and sin.shape == (10, 64)


def test_rope_position_zero_is_identity():  # no model
    cos, sin = M.rope_cos_sin(torch.arange(1), 64, 1_000_000.0)
    x = torch.randn(3, 1, 64)
    assert torch.allclose(M.apply_rope(x, cos, sin), x, atol=1e-6)


def test_rope_preserves_length():  # no model: rotation never stretches a vector
    cos, sin = M.rope_cos_sin(torch.arange(50), 64, 1_000_000.0)
    x = torch.randn(2, 50, 64)
    y = M.apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_rope_depends_only_on_relative_position():  # no model: section 2.6
    theta, dh = 1_000_000.0, 64
    q, k = torch.randn(1, 1, dh), torch.randn(1, 1, dh)

    def score(p, m):
        cq, sq = M.rope_cos_sin(torch.tensor([p]), dh, theta)
        ck, sk = M.rope_cos_sin(torch.tensor([m]), dh, theta)
        return (M.apply_rope(q, cq, sq) * M.apply_rope(k, ck, sk)).sum()

    assert torch.allclose(score(5, 2), score(105, 102), atol=1e-4)
    assert torch.allclose(score(9, 0), score(40, 31), atol=1e-4)


# --------------------------------------------------------------------------- #
# 3. MLP
# --------------------------------------------------------------------------- #
def test_mlp_matches_hf(hf, mine):
    W, cfg = mine
    x = torch.randn(5, cfg.d)
    with torch.no_grad():
        ref = hf.model.layers[0].mlp(x)
    assert torch.allclose(M.mlp(x, W, 0), ref, atol=1e-4)


# --------------------------------------------------------------------------- #
# 4. Attention + block, checked layer by layer
# --------------------------------------------------------------------------- #
def test_layer_by_layer_matches_hf(hf, tok, mine):
    W, cfg = mine
    ids = _ids(tok, PROMPTS[0])
    with torch.no_grad():
        ref = hf(ids[None], output_hidden_states=True).hidden_states
        _, hidden = M.forward(ids, W, cfg, return_hidden=True)

    assert torch.allclose(hidden[0], ref[0][0], atol=1e-5), "embeddings differ"
    # ref[i+1] is the output of layer i. The last HF entry has the final norm
    # applied, so it is skipped here; the logits test covers the end.
    for i in range(cfg.L - 1):
        a, b = hidden[i + 1], ref[i + 1][0]
        diff = (a - b).abs().max().item()
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-3), (
            f"first drift at layer {i}: max abs diff {diff:.3g}. "
            f"The bug is inside block() for layer {i} (attention or MLP)."
        )


# --------------------------------------------------------------------------- #
# 5. Full logits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_hf(hf, tok, mine, prompt):
    W, cfg = mine
    ids = _ids(tok, prompt)
    with torch.no_grad():
        ref = hf(ids[None]).logits[0]
        got = M.forward(ids, W, cfg)
    assert got.shape == ref.shape
    diff = (got - ref).abs().max().item()
    assert diff < 2e-3, f"max abs logit diff {diff:.3g}"


# --------------------------------------------------------------------------- #
# 6. KV cache
# --------------------------------------------------------------------------- #
def _tiny_cfg():
    return M.Config(d=8, L=2, h=4, n_kv=2, d_head=4, d_ff=16, V=10, eps=1e-6, rope_theta=10000.0)


def test_cache_write_then_read():  # no model
    cfg = _tiny_cfg()
    c = M.KVCache(cfg, max_len=10, dtype=torch.float32, device="cpu")
    k1, v1 = torch.randn(2, 3, 4), torch.randn(2, 3, 4)   # 3 tokens at 0..2
    k2, v2 = torch.randn(2, 1, 4), torch.randn(2, 1, 4)   # 1 token at 3
    c.write(1, 0, k1, v1)
    c.write(1, 3, k2, v2)
    k, v = c.read(1, 4)
    assert k.shape == (2, 4, 4) and v.shape == (2, 4, 4)
    assert torch.equal(k[:, :3], k1) and torch.equal(k[:, 3:], k2)
    assert torch.equal(v[:, :3], v1) and torch.equal(v[:, 3:], v2)
    assert c.data[0].abs().sum() == 0, "wrote into the wrong layer"


def test_cache_bytes_match_section_1_6(mine):
    _, cfg = mine
    c = M.KVCache(cfg, max_len=500, dtype=torch.float16, device="cpu")
    assert c.nbytes == 500 * 12_288   # 12 KiB per token for Qwen2.5-0.5B


def test_cache_prefill_then_decode_matches_full_forward(tok, mine):
    W, cfg = mine
    ids = _ids(tok, PROMPTS[2])
    with torch.no_grad():
        full = M.forward(ids, W, cfg)                        # no cache
        cache = M.KVCache(cfg, len(ids), torch.float32, "cpu")
        M.forward(ids[:-1], W, cfg, cache=cache, start_pos=0)  # prefill all but last
        last = M.forward(ids[-1:], W, cfg, cache=cache, start_pos=len(ids) - 1)
    assert torch.allclose(last[-1], full[-1], atol=1e-3)


def test_cache_generation_equals_no_cache(tok, mine):
    W, cfg = mine
    ids = _ids(tok, PROMPTS[0])
    a = M.greedy_generate(ids, W, cfg, 30, use_cache=False)
    b = M.greedy_generate(ids, W, cfg, 30, use_cache=True)
    assert torch.equal(a, b), (tok.decode(a), tok.decode(b))
