#!/usr/bin/env bash
# Python environment for conversion, benchmarks and the MLX reference: $WOOF_LAB/venv.
set -euo pipefail
lab=${WOOF_LAB:-$HOME/woof-lab}
mkdir -p "$lab"/{models,gguf,results}
cd "$lab"
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q --retries 10 numpy safetensors huggingface_hub sentencepiece protobuf jinja2 "transformers>=5.0"
pip install -q --retries 10 --index-url https://download.pytorch.org/whl/cpu torch
# Optional MLX-on-CUDA reference; it JIT-compiles with NVRTC 12.9 and needs matching headers via CUDA_HOME.
pip install -q --retries 10 "mlx[cuda12]" mlx-lm nvidia-cuda-runtime-cu12==12.9.79 || echo "MLX reference not installed (optional)"
