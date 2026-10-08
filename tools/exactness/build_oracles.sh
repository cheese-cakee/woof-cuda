#!/usr/bin/env bash
# Compiles the exactness oracles against the patched llama.cpp build.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
llama=${WOOF_LAB:-$HOME/woof-lab}/llama.cpp-woof
bin=$llama/build-cuda/bin
for tool in logit_parity cross_snapshot_parity graph_replay; do
  g++ -O2 -std=c++17 -I"$llama/include" -I"$llama/ggml/include" "$here/$tool.cpp" \
    -L"$bin" -lllama -lggml -lggml-base -Wl,-rpath,"$bin" -o "$bin/woof-${tool//_/-}"
  echo "built $bin/woof-${tool//_/-}"
done
