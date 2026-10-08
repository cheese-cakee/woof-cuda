#!/usr/bin/env bash
# MLX 4-bit -> f32 HF -> f32 GGUF (official converter) -> exact Q4_1 GGUF -> exact native Q4_64 GGUF.
# Usage: convert_model.sh <woof-2b|woof-4b>
set -euo pipefail
name=$1
here=$(cd "$(dirname "$0")" && pwd)
lab=${WOOF_LAB:-$HOME/woof-lab}
work=$lab/work
. "$lab/venv/bin/activate"
mkdir -p "$work" "$lab/gguf"

export PYTHONPATH=$lab/llama.cpp/gguf-py
python "$here/dequantize_to_hf.py" "$lab/models/$name-mlx" "$work/$name-f32-hf"
python "$lab/llama.cpp/convert_hf_to_gguf.py" "$work/$name-f32-hf" --outtype f32 --outfile "$work/$name-f32.gguf"
python "$here/repack_q4_1.py" "$lab/models/$name-mlx" "$work/$name-f32.gguf" "$lab/gguf/$name-Q4_1-exact.gguf"

export PYTHONPATH=$lab/llama.cpp-woof/gguf-py
python "$here/repack_native_affine.py" "$lab/gguf/$name-Q4_1-exact.gguf" "$lab/gguf/$name-native-affine.gguf"

# The f32 intermediates are ~2x the model size each and are not needed after conversion.
rm -r "$work/$name-f32-hf" "$work/$name-f32.gguf"
ls -la "$lab/gguf/$name-Q4_1-exact.gguf" "$lab/gguf/$name-native-affine.gguf"
