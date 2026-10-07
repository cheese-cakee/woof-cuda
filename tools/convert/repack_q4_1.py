"""Losslessly repack MLX 4-bit weights into a GGUF as Q4_1.

Input: the MLX checkpoint and the float32 GGUF that convert_hf_to_gguf.py made from its dequantized weights.
Each float32 tensor is matched to its MLX source by row contents, so any row reordering done by the converter
is carried over. A repacked tensor is kept only if its Q4_1 dequantization equals the float32 tensor bit for bit.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import gguf
from gguf.quants import dequantize as gguf_dequantize

from mlx_weights import MlxQuantizedModel, dequantize

Q4_1_BLOCK = 32


def row_hashes(matrix):
    return [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in matrix]


def fits_fp16_exactly(values_f32):
    return np.array_equal(values_f32.astype(np.float16).astype(np.float32), values_f32)


def zero_unused_scales(q, scales, group_size):
    """A group whose codes are all 0 equals its bias for any scale; MLX stores tiny placeholder scales there."""
    all_zero = (q.reshape(q.shape[0], -1, group_size) == 0).all(axis=-1)
    return np.where(all_zero, np.float32(0), scales)


def encode_q4_1(q, scales, biases, group_size):
    """q [rows, cols] uint8, scales/biases [rows, cols/group_size] f32 -> Q4_1 bytes [rows, cols/32*20]."""
    rows, cols = q.shape
    blocks_per_group = group_size // Q4_1_BLOCK
    blocks = q.reshape(rows, cols // Q4_1_BLOCK, Q4_1_BLOCK)
    packed = blocks[..., :16] | (blocks[..., 16:] << 4)
    d = np.repeat(scales, blocks_per_group, axis=1).astype(np.float16)
    m = np.repeat(biases, blocks_per_group, axis=1).astype(np.float16)
    out = np.empty((rows, cols // Q4_1_BLOCK, 20), dtype=np.uint8)
    out[..., 0:2] = d[..., None].view(np.uint8)
    out[..., 2:4] = m[..., None].view(np.uint8)
    out[..., 4:] = packed
    return out.reshape(rows, -1)


def group_column_hashes(matrix, group_size):
    """One hash per column block of group_size, over all rows."""
    blocks = matrix.reshape(matrix.shape[0], -1, group_size).transpose(1, 0, 2)
    return row_hashes(blocks.reshape(blocks.shape[0], -1))


def lookup_permutation(hashes, source_hashes):
    position = {h: i for i, h in enumerate(source_hashes)}
    if not all(h in position for h in hashes):
        return None
    return np.array([position[h] for h in hashes])


class SourceIndex:
    """Finds which MLX tensor produced a float32 matrix, allowing reordered rows or reordered column groups."""

    def __init__(self, model, group_size):
        self.model, self.group_size = model, group_size
        self.base_of_row, self.base_of_column_group = {}, {}
        for base in model.quantized_bases:
            weights = self.dequantized(base)
            for h in row_hashes(weights):
                self.base_of_row.setdefault(h, base)
            for h in group_column_hashes(weights, group_size):
                self.base_of_column_group.setdefault(h, base)

    def dequantized(self, base):
        return dequantize(*self.model.quantized_parts(base), self.group_size)

    def match(self, matrix):
        """Return (base, row_permutation, group_permutation) or None."""
        rows = np.arange(matrix.shape[0])
        groups = np.arange(matrix.shape[1] // self.group_size)

        hashes = row_hashes(matrix)
        base = collections.Counter(self.base_of_row.get(h) for h in hashes).most_common(1)[0][0]
        if base is not None:
            row_permutation = lookup_permutation(hashes, row_hashes(self.dequantized(base)))
            if row_permutation is not None:
                return base, row_permutation, groups

        hashes = group_column_hashes(matrix, self.group_size)
        base = collections.Counter(self.base_of_column_group.get(h) for h in hashes).most_common(1)[0][0]
        if base is not None:
            group_permutation = lookup_permutation(hashes, group_column_hashes(self.dequantized(base), self.group_size))
            if group_permutation is not None:
                return base, rows, group_permutation
        return None


def repack_tensor(model, index, tensor, group_size):
    """Return (Q4_1 bytes, reason) for a float32 tensor, or (None, reason) to keep it as is."""
    matrix = np.asarray(tensor.data)
    if tensor.tensor_type != gguf.GGMLQuantizationType.F32 or matrix.ndim != 2 or matrix.shape[1] % group_size:
        return None, "not a 2D f32 matrix"
    found = index.match(matrix)
    if found is None:
        return None, "no MLX source"
    base, row_permutation, group_permutation = found
    q, scales, biases = (part[row_permutation] for part in model.quantized_parts(base))
    rows = q.shape[0]
    q = q.reshape(rows, -1, group_size)[:, group_permutation].reshape(rows, -1)
    scales, biases = scales[:, group_permutation], biases[:, group_permutation]
    scales = zero_unused_scales(q, scales, group_size)
    if not (fits_fp16_exactly(scales) and fits_fp16_exactly(biases)):
        return None, f"{base}: scale/bias not exact in fp16"
    packed = encode_q4_1(q, scales, biases, group_size)
    roundtrip = gguf_dequantize(packed, gguf.GGMLQuantizationType.Q4_1)
    if not np.array_equal(roundtrip.view(np.uint32), matrix.view(np.uint32)):
        return None, f"{base}: Q4_1 roundtrip differs"
    notes = []
    if not np.array_equal(row_permutation, np.arange(len(row_permutation))):
        notes.append("rows reordered")
    if not np.array_equal(group_permutation, np.arange(len(group_permutation))):
        notes.append("column groups reordered")
    return packed, base + (f" ({', '.join(notes)})" if notes else "")


def blocks_with_tensors(reader):
    return {int(t.name.split(".")[1]) for t in reader.tensors if t.name.startswith("blk.")}


def copy_metadata(reader, writer, arch):
    """Copy all keys; drop an MTP (nextn) layer that the metadata declares but no tensor provides."""
    block_count_key, nextn_key = f"{arch}.block_count", f"{arch}.nextn_predict_layers"
    declared_block_count = reader.fields[block_count_key].contents()
    real_block_count = max(blocks_with_tensors(reader)) + 1
    phantom_nextn = nextn_key in reader.fields and declared_block_count > real_block_count
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name in (gguf.Keys.General.ARCHITECTURE, gguf.Keys.General.FILE_TYPE):
            continue
        if phantom_nextn and field.name == nextn_key:
            continue
        value = field.contents()
        if phantom_nextn and field.name == block_count_key:
            value = real_block_count
        elif phantom_nextn and field.name.startswith(arch) and isinstance(value, list) and len(value) == declared_block_count:
            value = value[:real_block_count]  # per-layer array
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, value, value_type, sub_type=sub_type)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_1)
    if phantom_nextn:
        print(f"dropped phantom MTP layer: {block_count_key} -> {real_block_count}, removed {nextn_key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mlx_dir", type=Path)
    parser.add_argument("f32_gguf", type=Path)
    parser.add_argument("out_gguf", type=Path)
    args = parser.parse_args()

    group_size = json.loads((args.mlx_dir / "config.json").read_text())["quantization"]["group_size"]
    model = MlxQuantizedModel(args.mlx_dir / "model.safetensors")
    index = SourceIndex(model, group_size)
    reader = gguf.GGUFReader(args.f32_gguf)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()

    outputs = []
    for tensor in reader.tensors:
        packed, reason = repack_tensor(model, index, tensor, group_size)
        print(f"{tensor.name:40s} {'Q4_1' if packed is not None else tensor.tensor_type.name:5s} {reason}")
        outputs.append((tensor, packed))
    repacked = sum(p is not None for _, p in outputs)
    print(f"repacked {repacked} of {len(outputs)} tensors; MLX quantized tensors: {len(model.quantized_bases)}")

    writer = gguf.GGUFWriter(args.out_gguf, arch=arch, endianess=reader.endianess)
    copy_metadata(reader, writer, arch)
    for tensor, packed in outputs:
        data = np.asarray(tensor.data) if packed is None else packed
        raw_dtype = tensor.tensor_type if packed is None else gguf.GGMLQuantizationType.Q4_1
        writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor, packed in outputs:
        writer.write_tensor_data(np.asarray(tensor.data) if packed is None else packed)
    writer.close()


if __name__ == "__main__":
    main()
