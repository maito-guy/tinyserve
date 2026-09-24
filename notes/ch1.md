# Chapter 1 notes

## Predictions (write BEFORE running anything)

- Naive loop, 512-token prompt, 256 output tokens: the last step should be
  slower than the first by a factor of about ____ because ____.
- Cached loop: steps should get slower by about ____ per step, because
  each step reads ____ more entries of the KV cache.
- Expected batch-1 decode speed on my GPU (bandwidth / model bytes): ____ tok/s.
  (You will find out in Chapter 3 why the real number is lower.)

## Measurements

| run | prompt | out | TTFT ms | ITL p50 ms | ITL p99 ms | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| naive  | 512 | 256 | | | | |
| cached | 512 | 256 | | | | |

## Characters per token (step 2)

| prompt | chars | tokens | chars/token |
| --- | --- | --- | --- |
| short English question | | | |
| English paragraph | | | |
| same paragraph, Arabic | | | |
| Python function | | | |
| table of numbers | | | |

## What surprised me

(two paragraphs, same day)
