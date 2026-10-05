"""Project 2, step 6: your forward pass vs Hugging Face, cached decode.

    python -m bench.ch2_compare --prompt-len 512 --out-len 128
"""
import argparse

import torch

from bench.run import make_prompt_ids
from bench.timing import gpu_timer, percentiles, warmup
from tinyserve import model as M
from tinyserve.generate import generate_cached
from tinyserve.model_loader import load


@torch.no_grad()
def time_mine(ids, W, cfg, n):
    """Prefill + (n-1) decode steps with your model and your cache."""
    ttft, steps = [], []
    cache = M.KVCache(cfg, len(ids) + n, W["embed"].dtype, ids.device)
    with gpu_timer(ttft):
        nxt = M.forward(ids, W, cfg, cache=cache, start_pos=0)[-1].argmax()
    pos = len(ids)
    for _ in range(n - 1):
        with gpu_timer(steps):
            nxt = M.forward(nxt.view(1), W, cfg, cache=cache, start_pos=pos)[-1].argmax()
        pos += 1
    return ttft[0], steps


def time_hf(model, ids, n):
    ttft, steps = [], []
    generate_cached(model, ids[None], n, step_times=steps, ttft=ttft)
    return ttft[0], steps


def report(name, ttft, steps):
    p = percentiles(steps)
    total = ttft + sum(steps)
    print(f"{name:14} TTFT {ttft*1e3:7.1f} ms   ITL p50 {p['p50']*1e3:6.1f} ms   "
          f"p99 {p['p99']*1e3:6.1f} ms   {(1 + len(steps)) / total:6.1f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--out-len", type=int, default=128)
    args = ap.parse_args()

    model, tok, device = load()                 # fp16 on GPU
    W, cfg = M.load_weights(model), M.Config.from_hf(model.config)
    ids2d = make_prompt_ids(tok, device, args.prompt_len)
    ids = ids2d[0]
    print(f"device={device} dtype={W['embed'].dtype} prompt={len(ids)} out={args.out_len}\n")

    warmup(lambda: time_hf(model, ids[:16], 4))
    warmup(lambda: time_mine(ids[:16], W, cfg, 4))

    report("Hugging Face", *time_hf(model, ids, args.out_len))
    report("mine", *time_mine(ids, W, cfg, args.out_len))


if __name__ == "__main__":
    main()