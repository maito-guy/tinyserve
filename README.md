# tinyserve

A from-scratch LLM inference engine, built one chapter at a time while
working through *LLM Inference from First Principles*. Final target: a
paged-KV, continuously batched, OpenAI-compatible server benchmarked
against vLLM, SGLang and llama.cpp.

Model: Qwen2.5-0.5B-Instruct. Hardware: a free Colab/Kaggle T4.

## Layout

```
tinyserve/   the engine (ch1: generate loops; ch2: own forward pass; ...)
bench/       benchmark harness, reused every chapter
tests/       correctness tests; run before every commit
notes/       one markdown file per chapter: predictions, results, surprises
reports/     plots and tables (generated, mostly gitignored)
```

## Setup (local)

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v -k "softmax or greedy or top"   # no GPU, no download
```

On Colab, see notes/colab_setup.md.

## Progress

- [ ] ch1: naive + cached generate, hand-written sampler, benchmark harness
- [ ] ch2: own forward pass matching HF logits, preallocated KV cache
- [ ] ch3: roofline, torch.compile, batch sweep
- [ ] ch4: scheduler, paged cache, prefix cache, chunked prefill, int8/int4, speculative
- [ ] ch5: OpenAI-compatible endpoint, race against vLLM/SGLang/llama.cpp
- [ ] capstone: Docker, CI, report
