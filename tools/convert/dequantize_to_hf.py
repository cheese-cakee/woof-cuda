"""Write an MLX 4-bit checkpoint as float32 Hugging Face safetensors, as input for convert_hf_to_gguf.py."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from mlx_weights import MlxQuantizedModel, dequantize
from mlx_to_hf_names import hf_tensors

SHARD_BYTES = 2 * 1024**3
COPIED_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")


def write_shards(named_arrays, out_dir):
    weight_map, shard, shard_bytes, index = {}, {}, 0, 0

    def flush():
        nonlocal shard, shard_bytes, index
        if shard:
            index += 1
            file_name = f"model-{index:05d}.safetensors"
            save_file(shard, str(out_dir / file_name))
            weight_map.update({name: file_name for name in shard})
            shard, shard_bytes = {}, 0

    for name, array in named_arrays:
        if shard_bytes + array.nbytes > SHARD_BYTES:
            flush()
        shard[name] = np.ascontiguousarray(array)
        shard_bytes += array.nbytes
    flush()
    (out_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mlx_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.mlx_dir / "config.json").read_text())
    group_size = config["quantization"]["group_size"]
    model = MlxQuantizedModel(args.mlx_dir / "model.safetensors")

    def mlx_f32_tensors():
        for base in model.quantized_bases:
            yield f"{base}.weight", dequantize(*model.quantized_parts(base), group_size)
        for name in model.plain_names:
            yield name, model.plain_f32(name)

    write_shards(hf_tensors(config["model_type"], mlx_f32_tensors()), args.out_dir)

    for key in ("quantization", "quantization_config"):
        config.pop(key, None)
    config["dtype"] = "float32"
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    for file_name in COPIED_FILES:
        if (args.mlx_dir / file_name).exists():
            shutil.copy(args.mlx_dir / file_name, args.out_dir / file_name)


if __name__ == "__main__":
    main()
