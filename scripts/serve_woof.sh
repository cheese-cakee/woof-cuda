#!/usr/bin/env bash
# OpenAI-compatible server on 127.0.0.1:8080. Usage: serve_woof.sh <mode> [extra llama-server flags]
set -euo pipefail
lab=${WOOF_LAB:-$HOME/woof-lab}
mode=${1:-exact}
if [ "$#" -gt 0 ]; then shift; fi

binary="$lab/llama.cpp-woof/build-cuda/bin/llama-server"
model="$lab/gguf/woof-4b-native-affine.gguf"
flags=()
server_flags=(-ngl 99 -fa on -c 16384 -np 1 --temp 0 --top-k 1)
# Exact profile: native f32 kernel, verify bit-identical to decode, fusion on, fixed attention split count.
export WOOF_EXACT_SPEC=1 WOOF_NATIVE_KERNEL=f32 WOOF_FA_SPLITS=8 GGML_CUDA_DISABLE_FUSION=0

case "$mode" in
  exact) ;;
  2b) model="$lab/gguf/woof-2b-native-affine.gguf" ;;
  dflash7)
    flags=(--spec-type draft-dflash -md "$lab/models/drafts/Qwen3.5-4B-DFlash.gguf" -ngld 99 --spec-draft-n-max 7)
    ;;
  lookup8|lookup16|2b-lookup8|2b-lookup16)
    count=${mode##*lookup}
    if [[ $mode == 2b-* ]]; then model="$lab/gguf/woof-2b-native-affine.gguf"; fi
    flags=(--spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m "$count" --spec-draft-n-max "$count")
    ;;
  stock|2b-stock)
    binary="$lab/llama.cpp/build-cuda/bin/llama-server"
    model="$lab/gguf/woof-4b-Q4_1-exact.gguf"
    if [ "$mode" = 2b-stock ]; then model="$lab/gguf/woof-2b-Q4_1-exact.gguf"; fi
    unset WOOF_EXACT_SPEC WOOF_NATIVE_KERNEL WOOF_FA_SPLITS GGML_CUDA_DISABLE_FUSION
    ;;
  *)
    echo "usage: serve_woof.sh [exact|dflash7|lookup8|lookup16|2b|2b-lookup8|2b-lookup16|stock|2b-stock] [llama-server flags]" >&2
    exit 2
    ;;
esac
exec "$binary" -m "$model" "${server_flags[@]}" --reasoning off --host 127.0.0.1 --port 8080 "${flags[@]}" "$@"
