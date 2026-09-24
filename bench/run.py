"""Project 1 step 7-8: the benchmark harness you reuse in every chapter.

Usage (after implementing generate.py):

    python -m bench.run --prompt-len 512 --out-len 256 --mode both --plot

Inputs:  prompt length, output length, mode (naive | cached | both)
Outputs: TTFT, mean/p50/p99 ITL, tokens/s, and a per-step latency plot
         saved to reports/ch1_step_latency.png

The argparse and plotting are done. The TODO is assembling the metrics from
the step times, which is the point of the exercise: you should be able to
say exactly what each number means (section 1.8).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bench.timing import percentiles, warmup
from tinyserve import generate as G
from tinyserve.model_loader import load


def make_prompt_ids(tok, device, prompt_len: int) -> torch.Tensor:
    """A prompt of (approximately) prompt_len tokens. Repeating text is fine
    for timing; the model does not need to say anything sensible."""
    text = "The history of computing is long and full of surprises. " * 200
    ids = tok(text, return_tensors="pt").input_ids[:, :prompt_len]
    return ids.to(device)


def run_one(mode: str, model, ids: torch.Tensor, out_len: int) -> dict:
    """Run one generation and return a metrics dict.

    TODO: fill in the metrics.
      steps:    per-step wall time list (seconds), from step_times
      ttft:     for cached mode, the prefill time; for naive mode the first
                step IS the prefill+first token, so use steps[0]
      itl:      the remaining steps
      tokens/s: generated tokens / total time
    Return: {"mode", "ttft_ms", "itl_mean_ms", "itl_p50_ms", "itl_p99_ms",
             "tokens_per_s", "steps_ms": [...]}
    """
    steps: list[float] = []
    ttft: list[float] = []
    if mode == "naive":
        G.generate_naive(model, ids, out_len, step_times=steps)
    elif mode == "cached":
        G.generate_cached(model, ids, out_len, step_times=steps, ttft=ttft)
    else:
        raise ValueError(mode)

    # TODO: compute the metrics from `steps` and `ttft`
    raise NotImplementedError


def plot(results: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in results:
        ax.plot(range(len(r["steps_ms"])), r["steps_ms"], label=r["mode"])
    ax.set_xlabel("decode step")
    ax.set_ylabel("step latency (ms)")
    ax.set_title("Per-step latency: naive vs KV-cached")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--out-len", type=int, default=256)
    p.add_argument("--mode", choices=["naive", "cached", "both"], default="both")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--json", type=Path, default=Path("reports/ch1_bench.json"))
    args = p.parse_args()

    model, tok, device = load()
    ids = make_prompt_ids(tok, device, args.prompt_len)
    print(f"device={device} prompt_tokens={ids.shape[1]} out_len={args.out_len}")

    modes = ["naive", "cached"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        warmup(lambda: run_one(mode, model, ids[:, :16], 4))
        r = run_one(mode, model, ids, args.out_len)
        results.append(r)
        summary = {k: v for k, v in r.items() if k != "steps_ms"}
        print(json.dumps(summary, indent=2))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2))
    if args.plot:
        plot(results, Path("reports/ch1_step_latency.png"))


if __name__ == "__main__":
    main()
