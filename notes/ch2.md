# Chapter 2 notes

## Step 1: parameter count by hand (before running the inspect script)

| Part | Calculation | Parameters |
| --- | --- | --- |
| W_q + bias | | |
| W_k, W_v + biases | | |
| W_o | | |
| MLP (gate, up, down) | | |
| 2 RMSNorm gains | | |
| One layer | | |
| 24 layers | | |
| Embedding (tied with LM head) | | |
| Final norm | | |
| Total | | |

Script says: ____________   Match? ___

## Predictions

- KV cache bytes after a 500-token prompt: ____
- FLOPs for one decode step at context 500: ____
- My forward pass vs Hugging Face speed (cached decode): ____ x slower / faster

## Bugs I hit (and which test caught them)

-

## Results

| | Hugging Face | Mine |
| --- | --- | --- |
| Max logit diff (fp32) | | |
| Cached decode ITL p50 (T4) | | |

## What surprised me
