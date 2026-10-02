"""Shared attention dispatch with optional Tokamax FlashAttention.

CPU, FP32/FP16, and installations without the ``tokamax`` extra retain Flax's
attention implementation. Tokamax selects a supported fused backend for GPU
BF16 inputs and falls back to XLA for unsupported shapes.
"""

from functools import cache

import jax
import jax.numpy as jnp
from flax import nnx


@cache
def _get_tokamax_attention():
    # Import only after device/distributed initialization, when a GPU call needs it.
    try:
        from tokamax import dot_product_attention
    except ModuleNotFoundError as error:
        if error.name != "tokamax":
            raise
        return None
    return dot_product_attention


def dot_product_attention(query, key, value, bias=None):
    """Attend to BTHD tensors, preserving Flax's scale and bias conventions."""
    if (
        query.dtype == jnp.bfloat16
        and query.dtype == key.dtype == value.dtype
        and jax.default_backend() == "gpu"
    ):
        attention = _get_tokamax_attention()
        if attention is not None:
            return attention(query, key, value, bias=bias)
    return nnx.dot_product_attention(query, key, value, bias=bias)
