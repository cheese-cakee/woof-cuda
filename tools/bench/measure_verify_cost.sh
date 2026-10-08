#!/usr/bin/env bash
# c(n) = cost of an n-token forward pass relative to one token, at context depth 512 (stock build, exact Q4_1 GGUF).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
lab=${WOOF_LAB:-$HOME/woof-lab}
for m in ${@:-woof-4b woof-2b}; do
  "$lab/llama.cpp/build-cuda/bin/llama-bench" -m "$lab/gguf/$m-Q4_1-exact.gguf" -ngl 99 -fa 1 \
    -p 1,2,3,4,5,6,8,9,10,12,16,24,32 -n 0 -d 512 -r 7 -o json 2>/dev/null > "$lab/results/verify-cost-$m.json"
  echo "== $m"
  python3 "$here/summarize_verify_cost.py" "$lab/results/verify-cost-$m.json"
done
