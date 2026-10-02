"""Shared attention dispatch with autotuned Tokamax FlashAttention.

CPU and FP32/FP16 retain Flax's attention implementation. GPU BF16 uses Tokamax's
automatic backend selection and measured kernel configurations, with an XLA
fallback for unsupported shapes.
"""

from functools import cache

import jax
import jax.numpy as jnp
from flax import nnx


@cache
def _get_tokamax_attention():
    # Import only after device/distributed initialization, when a GPU call needs it.
    import tokamax
    from absl import flags

    # Set the native policy globally so backward kernels also autotune after the
    # forward call returns. Explicit Tokamax flag/context overrides still apply.
    flags.FLAGS.set_default("tokamax_autotuning_cache_miss_fallback", "autotune")
    return tokamax.dot_product_attention


def dot_product_attention(query, key, value, bias=None):
    """Attend to BTHD tensors, preserving Flax's scale and bias conventions."""
    if (
        query.dtype == jnp.bfloat16
        and query.dtype == key.dtype == value.dtype
        and jax.default_backend() == "gpu"
    ):
        return _get_tokamax_attention()(query, key, value, bias=bias, implementation=None)
    return nnx.dot_product_attention(query, key, value, bias=bias)
