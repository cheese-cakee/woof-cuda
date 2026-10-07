"""Pack equal Q4_1 coefficient pairs into exact affine group-64 blocks.

Needs the patched gguf-py (it defines Q4_64): PYTHONPATH=$WOOF_LAB/llama.cpp-woof/gguf-py.
"""
import argparse
import json
from pathlib import Path

import gguf
import numpy as np
from gguf.quants import dequantize

from repack_q4_1 import copy_metadata


def pack(data):
    if data.shape[-1] % 40:
        raise ValueError("Row is not a multiple of 64 weights")
    pairs = data.reshape(data.shape[0], -1, 2, 20)
    if not np.array_equal(pairs[:, :, 0, :4], pairs[:, :, 1, :4]):
        raise ValueError("Adjacent blocks have different scale/bias")
    result = np.empty((*pairs.shape[:2], 36), dtype=np.uint8)
    result[:, :, :4] = pairs[:, :, 0, :4]
    result[:, :, 4:20] = pairs[:, :, 0, 4:]
    result[:, :, 20:] = pairs[:, :, 1, 4:]
    return result.reshape(data.shape[0], -1)


def unpack(data):
    groups = data.reshape(data.shape[0], -1, 36)
    pairs = np.empty((*groups.shape[:2], 2, 20), dtype=np.uint8)
    pairs[:, :, 0, :4] = groups[:, :, :4]
    pairs[:, :, 1, :4] = groups[:, :, :4]
    pairs[:, :, 0, 4:] = groups[:, :, 4:20]
    pairs[:, :, 1, 4:] = groups[:, :, 20:]
    return pairs.reshape(data.shape[0], -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise ValueError("Source and destination must differ")
    reader = gguf.GGUFReader(args.source)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(args.destination, arch=arch, endianess=reader.endianess)
    copy_metadata(reader, writer, arch)
    outputs = []
    evidence = []
    for tensor in reader.tensors:
        data = np.asarray(tensor.data)
        raw_type = tensor.tensor_type
        if raw_type == gguf.GGMLQuantizationType.Q4_1:
            data = pack(data)
            for start in range(0, data.shape[0], 1024):
                restored = unpack(data[start:start + 1024])
                original = tensor.data[start:start + 1024]
                if not np.array_equal(restored, original):
                    raise ValueError(f"Roundtrip differs: {tensor.name} rows {start}")
                if not np.array_equal(dequantize(restored, raw_type).view(np.uint32),
                                      dequantize(original, raw_type).view(np.uint32)):
                    raise ValueError(f"Weight values differ: {tensor.name} rows {start}")
            raw_type = gguf.GGMLQuantizationType.Q4_64
        writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, raw_type)
        outputs.append(data)
        evidence.append({"name": tensor.name, "type": raw_type.name, "source_bytes": int(tensor.data.nbytes),
                         "destination_bytes": int(data.nbytes), "exact": True})
        print(f"{tensor.name}: {raw_type.name} {tensor.data.nbytes} -> {data.nbytes} exact", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for data in outputs:
        writer.write_tensor_data(data)
    writer.close()
    reopened = gguf.GGUFReader(args.destination)
    if len(reopened.tensors) != len(reader.tensors):
        raise ValueError("Written tensor count differs")
    for original, written in zip(reader.tensors, reopened.tensors):
        if original.name != written.name or not np.array_equal(original.shape, written.shape):
            raise ValueError(f"Written name/shape differs: {original.name}")
        if written.tensor_type == gguf.GGMLQuantizationType.Q4_64:
            for start in range(0, written.data.shape[0], 1024):
                if not np.array_equal(unpack(written.data[start:start + 1024]), original.data[start:start + 1024]):
                    raise ValueError(f"Written roundtrip differs: {original.name}")
        elif written.tensor_type != original.tensor_type or not np.array_equal(written.data, original.data):
            raise ValueError(f"Written unquantized tensor differs: {original.name}")
    args.destination.with_suffix(".validation.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
