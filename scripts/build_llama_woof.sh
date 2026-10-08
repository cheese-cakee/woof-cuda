#!/usr/bin/env bash
# Builds two llama.cpp trees at the pinned base commit: stock ($WOOF_LAB/llama.cpp) and patched ($WOOF_LAB/llama.cpp-woof).
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
lab=${WOOF_LAB:-$HOME/woof-lab}
base=$(cat "$repo/patches/BASE_COMMIT")
arch=${CUDA_ARCH:-89}
nvcc=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
mkdir -p "$lab"

checkout() {  # dir
  if [ ! -d "$1" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$1"
  fi
  git -C "$1" fetch --depth 1 origin "$base" 2>/dev/null || git -C "$1" fetch origin
  git -C "$1" checkout --detach "$base"
}

build() {  # dir
  cmake -S "$1" -B "$1/build-cuda" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_CUDA_COMPILER="$nvcc" \
    -DLLAMA_BUILD_UI=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "$1/build-cuda" -j"${JOBS:-6}" --target llama-server llama-bench llama-cli test-backend-ops
}

checkout "$lab/llama.cpp"
build "$lab/llama.cpp"

checkout "$lab/llama.cpp-woof"
if ! git -C "$lab/llama.cpp-woof" apply --reverse --check "$repo/patches/woof-native-exact.patch" 2>/dev/null; then
  git -C "$lab/llama.cpp-woof" apply --whitespace=nowarn "$repo/patches/woof-native-exact.patch"
fi
build "$lab/llama.cpp-woof"
echo "stock:   $lab/llama.cpp/build-cuda/bin"
echo "patched: $lab/llama.cpp-woof/build-cuda/bin"
