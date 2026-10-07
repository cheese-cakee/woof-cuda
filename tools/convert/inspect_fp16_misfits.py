"""Report MLX groups whose bf16 scale or bias is not exactly representable in fp16."""
import sys
import numpy as np
from mlx_weights import MlxQuantizedModel

GROUP = 64
model = MlxQuantizedModel(sys.argv[1] + "/model.safetensors")
for base in model.quantized_bases:
    q, scales, biases = model.quantized_parts(base)
    scale_misfit = scales.astype(np.float16).astype(np.float32) != scales
    bias_misfit = biases.astype(np.float16).astype(np.float32) != biases
    if not (scale_misfit.any() or bias_misfit.any()):
        continue
    all_zero_q = (q.reshape(q.shape[0], -1, GROUP) == 0).all(axis=-1)
    unresolved = (scale_misfit & ~all_zero_q) | bias_misfit
    rows = np.unique(np.nonzero(unresolved)[0])
    print(f"{base}: scale misfits {scale_misfit.sum()} (q all zero in {(scale_misfit & all_zero_q).sum()}), "
          f"bias misfits {bias_misfit.sum()}, unresolved groups {unresolved.sum()} in {len(rows)} rows")
    if len(rows):
        print(f"   rows min {rows.min()} max {rows.max()}; first rows {rows[:12].tolist()}")
