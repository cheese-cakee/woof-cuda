"""Undo mlx-lm's load-time weight changes so convert_hf_to_gguf.py sees Hugging Face layout."""
import numpy as np

# mlx-lm qwen3_5 sanitize adds 1.0 to these norms when it imports Hugging Face weights.
QWEN3_5_SHIFTED_NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def qwen3_5_to_hf(name, array):
    name = name.replace("language_model.model.", "model.language_model.", 1)
    if name.endswith("conv1d.weight"):
        array = np.moveaxis(array, 2, 1)
    if array.ndim == 1 and name.endswith(QWEN3_5_SHIFTED_NORMS):
        array = array - np.float32(1.0)
    return name, array


def hf_tensors(model_type, named_arrays):
    for name, array in named_arrays:
        if model_type == "qwen3_5":
            name, array = qwen3_5_to_hf(name, array)
        yield name, array
