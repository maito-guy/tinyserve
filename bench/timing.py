"""Timing helpers that give correct GPU numbers.

Why this exists: CUDA calls return to Python before the GPU has finished.
If you time with plain time.perf_counter() around a model call, you measure
how long the CPU took to *issue* the work, not how long the GPU took to do
it. torch.cuda.synchronize() blocks until the GPU is done, so timestamps
taken after it are real.

Finished for you: bad timing would silently corrupt every later chapter.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def gpu_timer(out: list[float] | None):
    """Append the elapsed seconds of the block to `out` (if not None)."""
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    if out is not None:
        out.append(time.perf_counter() - t0)


def percentiles(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p99": float("nan")}
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p99": float(np.percentile(a, 99)),
    }


def warmup(fn, n: int = 2) -> None:
    """Run fn() n times and discard. First calls pay for kernel loading."""
    for _ in range(n):
        fn()
    sync()
