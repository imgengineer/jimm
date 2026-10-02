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


def _attention_with_value_width(query, key, value, bias=None):
    """XLA attention whose value heads are wider or narrower than query/key heads."""
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, preferred_element_type=jnp.float32)
    logits = logits * query.shape[-1] ** -0.5
    if bias is not None:
        logits = logits + bias
    weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, value)


def dot_product_attention(query, key, value, bias=None):
    """Attend to BTHD tensors, preserving Flax's scale and bias conventions.

    Value heads may differ in width from query/key heads (as in LeViT); those
    calls use a float32-softmax XLA implementation.
    """
    if value.shape[-1] != query.shape[-1]:
        return _attention_with_value_width(query, key, value, bias)
    if (
        query.dtype == jnp.bfloat16
        and query.dtype == key.dtype == value.dtype
        and jax.default_backend() == "gpu"
    ):
        return _get_tokamax_attention()(query, key, value, bias=bias, implementation=None)
    return nnx.dot_product_attention(query, key, value, bias=bias)
