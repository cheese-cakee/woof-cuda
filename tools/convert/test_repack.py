"""Q4_1 repack must dequantize to exactly the MLX values, including after reordering column groups."""
import numpy as np
import gguf
from gguf.quants import dequantize as gguf_dequantize

from mlx_weights import dequantize, unpack_4bit
from repack_q4_1 import encode_q4_1, group_column_hashes, lookup_permutation, zero_unused_scales

GROUP = 64
rng = np.random.default_rng(0)


def random_mlx_tensor(rows=8, cols=256):
    q = rng.integers(0, 16, size=(rows, cols), dtype=np.uint8)
    scales = rng.normal(0, 0.01, size=(rows, cols // GROUP)).astype(np.float16).astype(np.float32)
    biases = rng.normal(0, 0.05, size=(rows, cols // GROUP)).astype(np.float16).astype(np.float32)
    return q, scales, biases


def test_unpack_order():
    word = np.array([[0x76543210]], dtype=np.uint32)
    assert unpack_4bit(word).tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_q4_1_roundtrip_is_bit_exact():
    q, scales, biases = random_mlx_tensor()
    packed = encode_q4_1(q, scales, biases, GROUP)
    roundtrip = gguf_dequantize(packed, gguf.GGMLQuantizationType.Q4_1)
    assert np.array_equal(roundtrip.view(np.uint32), dequantize(q, scales, biases, GROUP).view(np.uint32))


def test_zero_code_group_with_subnormal_scale_stays_exact():
    q, scales, biases = random_mlx_tensor()
    q[0, :GROUP] = 0
    scales[0, 0] = np.float32(1.001e-07)  # bf16-exact, not fp16-exact
    exact = dequantize(q, scales, biases, GROUP)
    packed = encode_q4_1(q, zero_unused_scales(q, scales, GROUP), biases, GROUP)
    roundtrip = gguf_dequantize(packed, gguf.GGMLQuantizationType.Q4_1)
    assert np.array_equal(roundtrip.view(np.uint32), exact.view(np.uint32))


def test_column_group_permutation_is_recovered():
    q, scales, biases = random_mlx_tensor()
    source = dequantize(q, scales, biases, GROUP)
    permutation = np.array([2, 0, 3, 1])
    shuffled = source.reshape(source.shape[0], -1, GROUP)[:, permutation].reshape(source.shape)
    found = lookup_permutation(group_column_hashes(shuffled, GROUP), group_column_hashes(source, GROUP))
    assert np.array_equal(found, permutation)


if __name__ == "__main__":
    for test in (test_unpack_order, test_q4_1_roundtrip_is_bit_exact,
                 test_zero_code_group_with_subnormal_scale_stays_exact, test_column_group_permutation_is_recovered):
        test()
        print("PASS", test.__name__)
