# Chapter 1 notes

## Predictions (write BEFORE running anything)

- Naive loop, 512-token prompt, 256 output tokens: the last step should be
  slower than the first by a factor of about ____ because ____.
- Cached loop: steps should get slower by about ____ per step, because
  each step reads ____ more entries of the KV cache.
- Expected batch-1 decode speed on my GPU (bandwidth / model bytes): ____ tok/s.
  (You will find out in Chapter 3 why the real number is lower.)

## Measurements

I predicted naive steps would grow 1.5× and the cache would give a huge speed-up. Instead naive was nearly flat and the cache only doubled throughput. Cached decode ran at 33 ms per token, 10× slower than the bandwidth bound. My guess: a fixed per-step overhead (~30 ms) dominates both loops on a model this small, so removing work barely shows. Chapter 3 should explain this.
