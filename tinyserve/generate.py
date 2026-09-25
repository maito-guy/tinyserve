"""Project 1: hand-written generation.

Fill in every function marked TODO. Do not call model.generate() anywhere.
Run `pytest tests/` as you go: the tests tell you when each piece is right.

Order to work in:
  1. sample()            -- pure tensor math, test against the softmax table in 1.4
  2. generate_naive()    -- full re-run every step, use_cache=False
  3. generate_cached()   -- past_key_values, feed only the new token
"""

from __future__ import annotations

import torch
from bench.timing import gpu_timer   # at the top of the file


# --------------------------------------------------------------------------- #
# 1. Sampling
# --------------------------------------------------------------------------- #
def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Turn one row of logits (shape [V]) into a token id.

    Rules (section 1.4 and 1.5):
      * temperature == 0  -> greedy argmax, no randomness.
      * Divide logits by temperature BEFORE softmax.
      * Do the softmax in float32 and subtract the max first.
      * top_k: keep the k largest, set the rest to -inf.
      * top_p: sort descending, keep the smallest set whose cumulative
        probability >= p, set the rest to -inf. (Always keep at least 1.)
      * Then torch.multinomial(probs, 1, generator=generator).

    Check: logits [2, 1, 0] at temperature 1 must give probabilities
    [0.665, 0.245, 0.090].
    """
    if temperature == 0:
        return int(torch.argmax(logits).item())
    logits = logits.clone().float()   # never mutate the caller's tensor
    neg_inf = float("-inf")
    if top_p is not None and top_p < 1.0:
        probs = softmax_probs(logits, temperature)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=0)  
        cum_before = cum - sorted_probs
        remove = cum_before >= top_p  
        logits[sorted_indices[remove]] = neg_inf
    if top_k is not None and top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < threshold, neg_inf)

    probs = softmax_probs(logits, temperature)
    return int(torch.multinomial(probs, 1, generator=generator).item())
def softmax_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Helper you may want: float32 softmax with temperature. Exposed so the
    tests can check your numbers against the table in section 1.4."""
    p = logits.float() / temperature
    p = p - p.max()
    z = torch.exp(p)
    return z / z.sum()
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_naive(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    step_times: list[float] | None = None,
) -> torch.Tensor:
    """Return input_ids with max_new_tokens appended (shape [1, T + n]).

    Each step: call model(ids, use_cache=False), take logits[0, -1],
    sample, append. Stop early on eos_token_id.

    If step_times is a list, append the wall time of each step to it
    (use bench.timing.gpu_timer so the numbers are real).
    """
    input_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        with gpu_timer(step_times):
            out = model(input_ids, use_cache=False)
            logits = out.logits[0, -1]
            next_id = sample(logits, temperature=temperature)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=input_ids.device)], dim=1
            )
        if eos_token_id is not None and next_id == eos_token_id:
            break
    return input_ids

# --------------------------------------------------------------------------- #
# 3. Cached loop: feed only the new token, reuse past_key_values
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_cached(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    step_times: list[float] | None = None,
    ttft: list[float] | None = None,
) -> torch.Tensor:
    """Same contract as generate_naive, but:

      * First call (prefill): model(input_ids, use_cache=True). Record its
        duration into ttft if given. Keep out.past_key_values.
      * Every later call (decode): model(next_id[None, None],
        past_key_values=cache, use_cache=True). Feed ONE token.

    Under temperature=0 this must produce exactly the same tokens as
    generate_naive. That equality is your correctness test.
    """
    ids = input_ids.clone()          # the full sequence we return
    device = ids.device

    with gpu_timer(ttft):
        out = model(ids, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0, -1]
        next_id = sample(logits, temperature=temperature)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    if eos_token_id is not None and next_id == eos_token_id:
        return ids
    
    for _ in range(max_new_tokens - 1):
        with gpu_timer(step_times):
            out = model(
                torch.tensor([[next_id]], device=device),
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            logits = out.logits[0, -1]
            next_id = sample(logits, temperature=temperature)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if eos_token_id is not None and next_id == eos_token_id:
            break
    return ids