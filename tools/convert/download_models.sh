#!/usr/bin/env bash
# Resumable plain-HTTPS download of pinned Woof revisions, verified against release-provenance.json.
set -euo pipefail
mkdir -p "${WOOF_LAB:-$HOME/woof-lab}/models"
cd "${WOOF_LAB:-$HOME/woof-lab}/models"
fetch() {  # repo revision dir file
  local url="https://huggingface.co/ConwayResearch/$1/resolve/$2/$4"
  for i in $(seq 1 20); do
    curl -sSL --fail -C - --retry 5 --retry-all-errors --connect-timeout 20 --speed-limit 20000 --speed-time 60 -o "$3/$4" "$url" && return 0
    echo "retry $i $4"; sleep 5
  done
  return 1
}
verify_model() {  # dir
  [ -f "$1/release-provenance.json" ] || return 1
  python3 - "$1" <<'PY'
import hashlib, json, sys, pathlib
d = pathlib.Path(sys.argv[1]); prov = json.loads((d / "release-provenance.json").read_text())
bad = 0
for name, info in prov["files"].items():
    path = d / name
    h = hashlib.file_digest(path.open("rb"), "sha256").hexdigest() if path.exists() else "missing"
    print("OK  " if h == info["sha256"] else "BAD ", name, h[:12])
    bad += h != info["sha256"]
sys.exit(1 if bad else 0)
PY
}
get_model() {  # repo revision dir
  if verify_model "$3"; then return 0; fi
  mkdir -p "$3"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja \
           model.safetensors.index.json release-provenance.json LICENSE model.safetensors; do
    fetch "$1" "$2" "$3" "$f" || { echo "FAILED $1 $f"; return 1; }
  done
  verify_model "$3"
}
rm -rf woof-2b-mlx/.cache woof-4b-mlx/.cache
get_model Underdog-Woof-2B-1.1 73c8b5ca3abffb2358270f4bc86bff697131612a woof-2b-mlx
get_model Underdog-Woof-4B-1.1 5decb824793e03d33df8afb47bd30c4f3a917d87 woof-4b-mlx

# Zero-shot DFlash draft for the 4B (built for Qwen3.5-4B, the base Woof 4B was post-trained from).
mkdir -p drafts
draft_sum="a46bda8d229760ff330de40a02d948e4d1811c4419a774321588e4355707762b  drafts/Qwen3.5-4B-DFlash.gguf"
if ! echo "$draft_sum" | sha256sum -c --quiet 2>/dev/null; then
  curl -sSL --fail -C - --retry 5 -o drafts/Qwen3.5-4B-DFlash.gguf \
    https://huggingface.co/EntityDeletr/Qwen3.5-4B-DFlash-GGUF/resolve/c2e6879b70f5870f389f850934b144fbdb51c8e4/Qwen3.5-4B-DFlash.gguf
  echo "$draft_sum" | sha256sum -c
fi
echo DOWNLOAD_DONE
