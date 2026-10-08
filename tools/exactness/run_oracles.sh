#!/usr/bin/env bash
# Runs the exactness oracles under the serving profile. Fails if any verify width <= 64 differs from one-token decoding.
# Usage: run_oracles.sh [woof-4b|woof-2b ...]   (default: both)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
lab=${WOOF_LAB:-$HOME/woof-lab}
bin=$lab/llama.cpp-woof/build-cuda/bin
out=$lab/results/exactness
mkdir -p "$out"
export WOOF_EXACT_SPEC=1 WOOF_NATIVE_KERNEL=f32 WOOF_FA_SPLITS=8 GGML_CUDA_DISABLE_FUSION=0
if [ "$#" -eq 0 ]; then set -- woof-4b woof-2b; fi

for model in "$@"; do
  for prompt in function_edit dom_click; do
    gguf=$lab/gguf/$model-native-affine.gguf
    ids=$here/prompts/$model-$prompt.ids
    echo "== $model $prompt: verify width 1..65 vs one-token decode, rollback up to 32"
    "$bin/woof-logit-parity" "$gguf" "$ids" "$out/parity-$model-$prompt.csv" 32
    echo "== $model $prompt: 32-snapshot (speculative) context vs 0-snapshot (plain) context"
    "$bin/woof-cross-snapshot-parity" "$gguf" "$ids" "$out/cross-$model-$prompt.csv" 32 0
  done
done
echo "== CUDA graph replay with changed inputs vs CPU backend"
"$bin/woof-graph-replay"
echo "ALL ORACLES PASS (width 65 is outside the exact fast path and is reported, not required)"
