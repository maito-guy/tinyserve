"""Run with: pytest tests/ -v

Tests are ordered by the order you should implement things. The first two
need no GPU and no model download.
"""

import pytest
import torch

from tinyserve import generate as G


# ---- 1. sampler math (no model needed) ------------------------------------ #
def test_softmax_matches_section_1_4_table():
    logits = torch.tensor([2.0, 1.0, 0.0])
    got = G.softmax_probs(logits, temperature=1.0)
    assert torch.allclose(got, torch.tensor([0.665, 0.245, 0.090]), atol=2e-3)

    got = G.softmax_probs(logits, temperature=0.5)
    assert torch.allclose(got, torch.tensor([0.867, 0.117, 0.016]), atol=2e-3)

    got = G.softmax_probs(logits, temperature=2.0)
    assert torch.allclose(got, torch.tensor([0.506, 0.307, 0.186]), atol=2e-3)


def test_softmax_does_not_overflow_in_fp16():
    # e^50 overflows float16; max-subtraction must save you.
    logits = torch.tensor([50.0, 49.0, 0.0], dtype=torch.float16)
    got = G.softmax_probs(logits, temperature=1.0)
    assert torch.isfinite(got).all()
    assert abs(got.sum().item() - 1.0) < 1e-3


def test_greedy_is_argmax():
    logits = torch.tensor([0.1, 5.0, 0.3, 0.2])
    assert G.sample(logits, temperature=0.0) == 1


def test_top_k_drops_tail():
    logits = torch.tensor([2.0, 1.0, 0.0])
    gen = torch.Generator().manual_seed(0)
    picks = {G.sample(logits, temperature=1.0, top_k=2, generator=gen) for _ in range(200)}
    assert 2 not in picks  # the smallest logit can never be chosen


def test_top_p_keeps_two_of_three():
    # cumulative: 0.665, 0.910 -> p=0.9 keeps exactly two tokens
    logits = torch.tensor([2.0, 1.0, 0.0])
    gen = torch.Generator().manual_seed(0)
    picks = {G.sample(logits, temperature=1.0, top_p=0.9, generator=gen) for _ in range(200)}
    assert picks <= {0, 1} and picks == {0, 1}


# ---- 2 & 3. generation loops (downloads the model; slow-ish) --------------- #
@pytest.fixture(scope="module")
def loaded():
    from tinyserve.model_loader import load

    return load()


def test_naive_and_cached_agree_under_greedy(loaded):
    model, tok, device = loaded
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)
    a = G.generate_naive(model, ids, max_new_tokens=20, temperature=0.0)
    b = G.generate_cached(model, ids, max_new_tokens=20, temperature=0.0)
    assert torch.equal(a, b), (tok.decode(a[0]), tok.decode(b[0]))


def test_cached_appends_exactly_n_tokens(loaded):
    model, tok, device = loaded
    ids = tok("Hello", return_tensors="pt").input_ids.to(device)
    out = G.generate_cached(model, ids, max_new_tokens=7, temperature=0.0)
    assert out.shape[1] == ids.shape[1] + 7
