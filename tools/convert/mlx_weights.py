"""Read MLX affine-quantized safetensors with numpy only."""
import json
import struct

import numpy as np

SAFETENSORS_DTYPES = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32, "F16": np.float16}


def load_safetensors(path):
    """Return {name: (dtype_name, array)}; BF16 arrays are returned as raw uint16."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_len)
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = SAFETENSORS_DTYPES[info["dtype"]]
        tensors[name] = (info["dtype"], data[start:end].view(dtype).reshape(info["shape"]))
    return tensors


def bf16_to_f32(raw_u16):
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


def unpack_4bit(packed_u32):
    """[rows, cols/8] uint32 -> [rows, cols] uint8; element i sits in bits 4*(i%8) of word i//8."""
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    nibbles = (packed_u32[..., None] >> shifts) & 0xF
    return nibbles.reshape(packed_u32.shape[0], -1).astype(np.uint8)


def dequantize(q, scales_f32, biases_f32, group_size=64):
    """w = scale * q + bias per group, in float32 (multiply then add, matching ggml's Q4_1 dequantize)."""
    rows, cols = q.shape
    q_groups = q.reshape(rows, cols // group_size, group_size).astype(np.float32)
    w = q_groups * scales_f32[..., None] + biases_f32[..., None]
    return w.reshape(rows, cols)


class MlxQuantizedModel:
    """Groups `<base>.weight/.scales/.biases` triples; other tensors are plain."""

    def __init__(self, path):
        self.tensors = load_safetensors(path)
        self.quantized_bases = sorted(n[: -len(".scales")] for n in self.tensors if n.endswith(".scales"))
        quantized_names = {f"{b}.{s}" for b in self.quantized_bases for s in ("weight", "scales", "biases")}
        self.plain_names = sorted(n for n in self.tensors if n not in quantized_names)

    def quantized_parts(self, base):
        """Return (q uint8 [rows, cols], scales f32 [rows, groups], biases f32 [rows, groups])."""
        q = unpack_4bit(self.tensors[f"{base}.weight"][1])
        scales = bf16_to_f32(self.tensors[f"{base}.scales"][1])
        biases = bf16_to_f32(self.tensors[f"{base}.biases"][1])
        return q, scales, biases

    def plain_f32(self, name):
        dtype_name, array = self.tensors[name]
        return bf16_to_f32(array) if dtype_name == "BF16" else array.astype(np.float32)
