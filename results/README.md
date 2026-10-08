# Results

Raw outputs behind the numbers in the top-level README. Suite files are written by `tools/bench/resume_suite.py`:
per configuration, per round, per prompt: generated token ids, decode and prefill rates, draft/accepted counts, plus
a `differences` list against the first configuration. Configuration names inside these files are the ones used when
they were recorded (`reviewed_*` = this repo's patched build, `native_*` = native `Q4_64` weights, `stock_*` =
unmodified llama.cpp); the portable equivalents are in `configs/`.

| path | contents |
|---|---|
| `4b/suite-reviewed.json`, `.log` | Woof 4B, 20 prompts x 3 rounds: plain, DFlash 7/15, lookup 16, earlier exact build |
| `4b/final-statistics.json` | per-configuration medians/ranges of decode ratio, tok/s and request time |
| `4b/suite-reviewed-long.json` | 4 prompts, 512-token outputs, plain vs DFlash 7 |
| `2b/suite-2b.json`, `suite-2b-long.json` | Woof 2B: plain, stock, lookup 8/16/32; long-output check |
| `27b/suite-27b.json` | Underdog-27B CPU/GPU split, plain vs DFlash2 (2 complete rounds, 4 short prompts; not exact) |
| `exactness/parity-f32-*.csv` | logit parity by verify width (count of unequal f32 logits, argmax differences, rollback); `-2b-` files are Woof 2B. Regenerate with `tools/exactness/run_oracles.sh` |
| `exactness/cross-*.csv` | parity between contexts with 0 and 7/15/16/32 recurrent snapshots (2B: 0 vs 32) |
| `exactness/short-split32.csv`, `graph-api.csv` | fixed attention-split bound on a short context; CUDA graph capture/replay API trace |
| `exactness/acceptance-proof.json` | exit status of every acceptance check |
| `exactness/build-manifest.json`, `source-manifest.json` | toolchain, flags, source and binary hashes of the measured build |
| `exactness/*.validation.json` | per-tensor bit-exact validation of the native GGUF files |
| `baseline/verify-cost-*.json` | c(n): forward-pass time for n = 1..32 tokens |
| `baseline/bench-*-q4_1-exact.md` | stock llama.cpp llama-bench, exact-weight Q4_1 |
| `baseline/mlx-cuda-*.json`, `llama-plain-*.json` | MLX (CUDA backend) greedy reference and matching llama.cpp tokens |
