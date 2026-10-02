"""Shared attention dispatch with Tokamax FlashAttention.

CPU and FP32/FP16 retain Flax's attention implementation. GPU BF16 uses Tokamax's
automatic backend selection with heuristic kernel configurations, and an XLA
fallback for unsupported shapes. ``set_attention_autotuning(True)`` instead
benchmarks candidate kernels for every new shape: about 1-2% faster steps, but
minutes of extra compilation for models with many attention shapes.
"""

from functools import cache

import jax
import jax.numpy as jnp
from flax import nnx


@cache
def _get_tokamax_attention():
    # Import only after device/distributed initialization, when a GPU call needs it.
    import tokamax

    return tokamax.dot_product_attention


def set_attention_autotuning(enabled: bool) -> None:
    """Autotune Tokamax kernels for new attention shapes, or use heuristic configurations.

    The policy applies to forward and backward kernels compiled afterwards in this
    process; explicit Tokamax flag or context overrides still take precedence.
    """
    import tokamax  # noqa: F401  (defines the Tokamax flags)
    from absl import flags

    policy = "autotune" if enabled else "heuristics"
    flags.FLAGS.set_default("tokamax_autotuning_cache_miss_fallback", policy)


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
