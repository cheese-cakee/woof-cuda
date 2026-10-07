# woof-cuda

Lossless speculative decoding for Conway Research's Woof models on a 6 GB laptop GPU.

Woof 4B runs at **123 tok/s** on an RTX 4050 Laptop, **2.14x** faster than plain decoding, with greedy output that is
**token-for-token identical** to plain decoding on every prompt of a 20-prompt task and browser-agent suite. Stock
llama.cpp speculation on the same model changes 30 of those 60 outputs.

It is built from three pieces:

- **Exact import of the MLX release.** Woof ships only as MLX affine 4-bit weights. A converter maps them into a new
  native GGUF type, `Q4_64` (4.5 bits/weight, one fp16 scale and bias per 64 weights), and proves every weight
  bit-exact against the original.
- **A batch-invariant CUDA verify kernel.** Verifying 1 to 64 draft tokens produces logits bit-identical to decoding
  them one at a time, so speculation cannot change the answer, while still streaming the weights at memory speed.
- **An exact serving profile for a hybrid model.** Fixed-split attention, per-token recurrent-state snapshots for the
  24 Gated DeltaNet layers, and a bounded verify window, all on top of llama.cpp's DFlash and n-gram speculation.

Independent project. Not affiliated with or endorsed by Conway Research.

## Results

RTX 4050 Laptop 6 GB (sm_89, 187 GB/s measured), CUDA 12.6, WSL2 Ubuntu 24.04. 20 prompts (16 task types plus 4
browser-agent DOM tasks, up to 14K prompt tokens), greedy, thinking off, 128 output tokens, 3 alternating rounds.
Rates are median over prompts; speedups are geometric means of matched per-prompt ratios, median [min-max] over rounds.

**Woof 4B**

| mode | decode tok/s | speedup | outputs equal to plain |
|---|---:|---:|---:|
| plain (native exact) | 56.2 [55.9-58.7] | 1.00x | reference |
| DFlash, 7 draft tokens | **122.8** [122.2-123.7] | **2.14x** [2.07-2.15] | **60/60** |
| DFlash, 15 draft tokens | 82.3 | 1.55x | 60/60 |

Speedup by task kind with 7 draft tokens: edit 2.82x, transform 2.47x, browser agent 2.13x, writing 1.54x, document
QA 1.26x. Total suite request time, including the long DOM prefills, falls from 52.7 s to 38.4 s. Longer outputs
(512 tokens) also match plain decoding exactly.

**Woof 2B**

| mode | decode tok/s | speedup | outputs equal to plain |
|---|---:|---:|---:|
| plain (native exact) | 97.5 [96.6-102.5] | 1.00x | reference |
| n-gram lookup, 8 tokens | 100.6 | 1.15x overall, 1.56x on edits | **60/60** |

**Reference points on the same GPU (Woof 4B)**

| engine | decode tok/s | notes |
|---|---:|---|
| MLX 0.32.3 (CUDA backend) | 46.5 | out of memory on prompts above ~1.1K tokens |
| stock llama.cpp, exact-weight Q4_1 | 55.5 | 79% of the bandwidth roofline |
| this repo, plain | 56.2 | f32 weight arithmetic, no activation quantization |
| this repo, DFlash 7 | 122.8 | lossless |

MLX: greedy reference runner, short prompts. Stock llama.cpp: `llama-bench` tg128, 7 runs. This repo: the suite
harness above. Methods differ, so treat this table as orientation; the matched comparisons are the tables above.

Raw data, per-prompt token sequences and build manifests are in [`results/`](results/).

## How it works

Batch-1 decoding is bound by memory bandwidth: Woof 4B streams 2.4-2.6 GB of weights per token, and the large
matrix-vector kernels already run at 96-99% of the 187 GB/s roofline. Verifying n tokens at once costs almost the same
as decoding one (1.11x for 8 tokens, 1.38x for 32 on this GPU), so speculation is the lever. It is only lossless if the
verify pass computes exactly what decoding would have, and in stock llama.cpp it does not: kernel selection, warp
counts, attention split-K and activation quantization all depend on batch size.

This repo fixes the arithmetic instead of the tolerance. The native kernel dequantizes in f32, walks K in the same
order for every column count, reduces with one fixed tree, and gets its speed from shared-memory staging, `cp.async`
pipelining and register tiling. The rest of the graph is pinned the same way.

Full design, data and validation: [`docs/design.md`](docs/design.md).

## Quickstart

Linux or WSL2 with an NVIDIA GPU (built and tested on sm_89; set `CUDA_ARCH` for others), CUDA 12.x, CMake, GCC,
Python 3.12. Everything lives under `$WOOF_LAB` (default `~/woof-lab`). Converting the 4B needs ~35 GB of temporary
disk for f32 intermediates, which the script deletes afterwards.

```bash
scripts/setup_env.sh                     # Python venv
scripts/build_llama_woof.sh              # stock + patched llama.cpp at the pinned base commit
tools/convert/download_models.sh         # pinned Woof 2B/4B releases + DFlash draft, all hash-checked
tools/convert/convert_model.sh woof-4b   # MLX -> exact Q4_1 -> exact native Q4_64
tools/convert/convert_model.sh woof-2b

scripts/serve_woof.sh dflash7            # OpenAI-compatible server and web UI on http://127.0.0.1:8080
```

Modes: `exact` (plain native), `dflash7`, `lookup8`, `lookup16`, `2b`, `2b-lookup8`, `2b-lookup16`, and `stock` /
`2b-stock` for the unmodified llama.cpp baseline. Run `stock` and `dflash7` on the same prompt to see the difference;
`exact` and `dflash7` produce the same text.

## Reproducing the measurements

```bash
. ~/woof-lab/venv/bin/activate
python tools/bench/resume_suite.py woof-4b configs/woof-4b-suite.json ~/woof-lab/results/woof-4b.json --runs 3
python tools/bench/resume_suite.py woof-2b configs/woof-2b-suite.json ~/woof-lab/results/woof-2b.json --runs 3
tools/bench/measure_verify_cost.sh       # c(n) curve

tools/exactness/build_oracles.sh         # logit parity, cross-snapshot parity, CUDA graph replay
```

The suite runner alternates configuration order between rounds, runs a canary first, saves after every
configuration (resumable), and reports per-prompt token differences against the first configuration. Close other GPU
applications before timing; laptop GPUs share VRAM and power with the desktop.

## What "exact" means here

Speculative greedy output equals plain greedy output of this native build, with logits bit-identical for verify
widths 1-64, including after rollback, on the tested prompts. Native arithmetic is a faithful f32 evaluation of the
released weights; it is not bitwise equal to MLX or to stock llama.cpp (which quantizes activations to 8 bits).
Sampled decoding, other GPUs, other KV-cache types and verify widths above 64 are outside the claim.

## Layout

```
patches/woof-native-exact.patch   llama.cpp changes (base commit in patches/BASE_COMMIT)
  ggml-cuda/woof_verify_matvec.cuh   batch-invariant native f32 matvec (serving path)
  ggml-cuda/woof-native-affine.cuh   Q4_64 dispatch, exact gather, fallbacks
  ggml-cuda/woof_tc_matvec.cuh       tensor-core variants (experimental, different arithmetic)
tools/convert/     MLX -> GGUF exact import, Q4_64 repack, tests
tools/exactness/   logit parity, cross-snapshot parity, graph replay oracles
tools/bench/       20-prompt suite, matched multi-round runner, verify-cost curve
tools/reference/   MLX (CUDA) reference runner and token comparison
configs/           benchmark configurations
results/           raw results behind every number above
docs/design.md     design and validation
```

## Next

- A draft trained on Woof's own outputs (the zero-shot Qwen3.5-4B draft accepts 47.6% of tokens at depth 7).
- Native Windows build of the serving path.
- Remove the remaining ~1 ms of host-side staging between decode steps.
- Underdog-27B on 6 GB: currently 2.8 tok/s plain and 3.4 tok/s with DFlash2 on a CPU/GPU split (not exact); the
  bottleneck is CPU-side verification.

## Licenses

Code in this repository: MIT. The patch applies to [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
Woof models: Apache-2.0, Conway Research. DFlash draft: [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)
(MIT), GGUF by [EntityDeletr](https://huggingface.co/EntityDeletr/Qwen3.5-4B-DFlash-GGUF) (Apache-2.0).
