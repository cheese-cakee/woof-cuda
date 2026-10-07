# Design

How Woof's MLX release becomes a native CUDA format, why speculative decoding normally changes greedy outputs, and
how this implementation makes verification bit-identical to decoding while keeping it fast.

## 1. Target and hardware budget

Woof 4B is a post-trained Qwen3.5-4B hybrid: 32 layers, 24 Gated DeltaNet (linear attention) and 8 full-attention
layers (16 query / 4 KV heads, head_dim 256, output gate), hidden 2560, SiLU MLP 9216, tied 248320 x 2560 embedding.
Woof 2B is a Llama-architecture model: 42 layers, hidden 2048, GQA 16/2, MLP 6144, untied 130560-token vocabulary.
Both ship only as MLX affine 4-bit weights (group 64, bf16 scale and bias).

Measured on the RTX 4050 Laptop (AD107, sm_89, 96-bit GDDR6):

| quantity | value |
|---|---|
| read bandwidth (128-bit loads, 1-2 GB buffer) | 187 GB/s, 97% of nominal 192 GB/s |
| Woof 4B weights read per decode token (Q4_1 / native 4.5-bit) | 2.63 GB / 2.38 GB |
| decode roofline at 187 GB/s (Q4_1 / native) | 71 / ~78 tok/s |
| stock llama.cpp decode, Woof 4B | 55.5 tok/s (17.8 ms/token, 79% of roofline) |

Nsight profile of one stock decode token: the large matrix-vector kernels already run at 96-99% of the roofline; the
remaining 3.8 ms splits across host-side input staging between tokens (1.1 ms), one under-performing 9216 x 2560
gate/up shape (0.8 ms), small kernels (0.7 ms), and recurrent/attention work (0.7 ms). Plain decode therefore has a
ceiling near 70 tok/s. The step change has to come from reading the weights fewer times per generated token.

## 2. Verification is almost free

Cost of one forward pass over n tokens relative to one token (Woof 4B, context depth 512, stock build):

| n | 1 | 4 | 8 | 9 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| c(n) | 1.00 | 1.04 | 1.11 | 1.21 | 1.28 | 1.38 |

Batch-1 decode is memory-bound, so verifying a draft of k tokens costs little more than producing one. Speculative
decoding is the dominant lever. The jump at n = 9 is llama.cpp switching from its matrix-vector kernel (MMVQ, <= 8
columns) to its tiled kernel (MMQ).

## 3. Why stock speculative decoding changes greedy output

Greedy speculation is lossless only if the logits computed while verifying a draft equal the logits plain decoding
would compute token by token. In stock llama.cpp they do not, because the arithmetic depends on how many tokens are in
the batch:

- MMVQ chooses its warp count by column count (4 warps for 1-4 columns, 2 for 5-8), which changes the reduction tree.
- Above 8 columns, matrix multiplication moves from MMVQ to MMQ: different tiling and accumulation order.
- Flash attention picks its kernel and its split-K factor from the batch shape.
- Activations are quantized to Q8_1 for the integer dot products, so even the weight arithmetic is approximate.

Near-ties then flip. On the 20-prompt suite, stock llama.cpp with a DFlash draft changes 30 of 60 greedy outputs,
and stock n-gram lookup changes 6-36 of 60 depending on the configuration.

## 4. Native 4.5-bit format: `Q4_64`

An MLX group of 64 weights is `w = scale * q + bias` with 4-bit codes `q`. `Q4_64` stores exactly that:

```c
#define QK4_64 64
typedef struct {
    ggml_half d;      // scale
    ggml_half m;      // bias
    uint8_t   qs[32]; // 64 4-bit codes
} block_q4_64;        // 36 bytes = 4.5 bits/weight
```

Import pipeline (`tools/convert/`):

1. `dequantize_to_hf.py` reads the MLX safetensors with numpy (bf16 handled bit-exactly), computes `scale*q + bias`
   in f32 and undoes mlx-lm's load-time rewrites for Qwen3.5 (conv1d axis order, +1 norm shift).
2. The official `convert_hf_to_gguf.py` handles names, metadata, the Llama Q/K row permutation and Qwen3.5's V-head
   reordering.
3. `repack_q4_1.py` matches every f32 tensor back to its MLX source by hashing rows, or 64-column groups when the
   converter permutes columns (Qwen3.5 `out_proj`), re-encodes it as Q4_1 (each MLX group becomes two 32-value blocks
   with the same scale/bias), and keeps it only if the Q4_1 dequantization equals the f32 tensor bit for bit.
4. `repack_native_affine.py` folds each pair of Q4_1 blocks back into one `Q4_64` block and re-verifies every tensor
   after writing.

Details that matter for exactness:

- bf16 scale/bias must be exactly representable in fp16. MLX stores tiny placeholder scales (~1e-7) for groups whose
  codes are all zero; the scale is irrelevant there and is set to 0, which is exact.
- Woof 2B's embedding table keeps 180 rare-token rows with scales fp16 cannot hold, so `token_embd` stays f32. It is
  gathered row by row from host memory and costs no VRAM or decode bandwidth.
- Woof 4B's config declares an MTP layer that the release does not ship; the converter would emit a phantom block and
  the file would not load. The repacker drops it from the metadata (block count and per-layer arrays).

Result: 4B 249/249 and 2B 295/296 quantized tensors bit-exact; 4B file 2.64 GB (Q4_1) -> 2.38 GB.

## 5. Batch-invariant verify kernel

`woof_verify_matvec.cuh` computes `y = W x` for n = 1..64 input columns with one invariant: the arithmetic producing
any output element does not depend on n.

- Weights are dequantized in f32 as `d*q + m` (exact: the product of an fp16 scale and a 4-bit integer fits in f32).
  Activations stay f32; there is no activation quantization.
- Each lane walks the same groups of K in the same ascending order for every column, accumulates with explicitly
  rounded FMAs, and the warp reduces with one fixed pairwise shuffle tree.
- Speed comes from data movement only: persistent blocks process row tiles, each warp keeps several rows and columns
  in registers, and 512-value input chunks are staged through shared memory with 16-byte `cp.async` copies and an
  XOR-swizzled layout. Column tiles reuse a row's weights while they are still in L2.
- A dispatcher picks the tile shape by column count (rows/warp, columns/tile, warps/block, chunk, pipeline stages);
  the shape changes how work is scheduled, never which products are summed in which order.

Supported fast path: contiguous `Q4_64` weights, f32 activations with 16-byte-aligned columns, K a multiple of 256,
up to 64 columns. Other shapes fall back to an expanded Q4_1 path (misaligned or strided inputs are first staged into
aligned storage). The fallback is correct but not part of the exactness guarantee; the verify window is bounded so
the model never leaves the fast path.

Two tensor-core variants (`woof_tc_matvec.cuh`, fp16 activations, one or two fp16 planes) are batch-invariant within
themselves but change the arithmetic relative to the f32 kernel, so they are kept as experiments, not as the serving
path.

## 6. Exact mode

`WOOF_EXACT_SPEC=1` makes the rest of the graph batch-invariant too:

| component | exact-mode behavior |
|---|---|
| weight matmuls | native f32 kernel above (`WOOF_NATIVE_KERNEL=f32`) |
| flash attention | vector kernel for <= 64 query rows with a fixed split count (`WOOF_FA_SPLITS`, validated 1..32; default profile 8) independent of batch shape |
| Gated DeltaNet rollback | per-token recurrent-state snapshots sized to the draft length, so a rejected suffix is undone by restoring a snapshot, not by replaying a forward pass |
| draft length | bounded to 63 so verification stays inside the 64-column fast path |
| fusions | enabled; the fusions used apply identically for one and many tokens (verified by the oracles below) |

## 7. How exactness is checked

- `tools/exactness/logit_parity.cpp`: decodes a fixed token sequence one token at a time, then verifies the same
  tokens as one batch of width n, and compares every f32 logit. Widths 1, 2, 4, 5, 8, 9, 16, 32, 33, 64, plus rollback
  of a rejected suffix and continued decoding.
- `tools/exactness/cross_snapshot_parity.cpp`: the same comparison between contexts configured with different
  snapshot counts (0 vs 7, 15, 16, 32), i.e. a plain server versus a speculative one.
- `tools/exactness/graph_replay.cpp`: CUDA graph capture and replay with changed inputs, checked against the CPU
  backend, including padded and offset views.
- `test-backend-ops`: native type cases, including misaligned and strided inputs (58/58 pass).
- End to end: every speculative configuration's greedy tokens are compared with plain decoding on the full suite.

Results are in `results/exactness/` (logit CSVs report the count of unequal f32 entries, which is 0 for every
supported width) and `results/4b`, `results/2b` (0/60 differing outputs for every configuration).

## 8. Scope and limits

- "Exact" means speculative greedy output equals plain greedy output of this native build. Native arithmetic is a
  faithful f32 evaluation of the released weights, but it is not bitwise equal to MLX or to stock llama.cpp, which
  quantizes activations.
- Greedy, single sequence, f16 KV cache, verify widths up to 64, measured on sm_89. Sampled speculation, other GPUs and
  other cache types are not covered.
- The DFlash draft was trained for Qwen3.5-4B and is used zero-shot (47.6% acceptance at 7 draft tokens). A draft
  trained on Woof's own outputs is the obvious next gain.
- The 27B configuration is a CPU/GPU split baseline; its speculative outputs are not exact.
