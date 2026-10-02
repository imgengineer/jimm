"""Attention dispatch, numerical parity, and gradient regressions."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jimm import attention


def _inputs(query_length, key_length, heads, head_dim, dtype, with_bias):
    keys = jax.random.split(jax.random.key(0), 4)
    query = jax.random.normal(keys[0], (2, query_length, heads, head_dim), dtype)
    key = jax.random.normal(keys[1], (2, key_length, heads, head_dim), dtype)
    value = jax.random.normal(keys[2], key.shape, dtype)
    if with_bias:
        # FP32 relative bias is retained during mixed-precision model execution.
        bias = jax.random.normal(keys[3], (1, heads, query_length, key_length)) * 0.1
        return query, key, value, bias
    return query, key, value


@pytest.mark.parametrize(
    "query_length,key_length,with_bias", [(17, 17, False), (1, 29, False), (13, 13, True)]
)
def test_fp32_attention_preserves_flax_outputs_and_gradients(query_length, key_length, with_bias):
    args = _inputs(query_length, key_length, 3, 24, jnp.float32, with_bias)
    expected = jax.jit(nnx.dot_product_attention)(*args)
    actual = jax.jit(attention.dot_product_attention)(*args)
    np.testing.assert_array_equal(actual, expected)
    for fn in (nnx.dot_product_attention, attention.dot_product_attention):
        loss = lambda operands: jnp.square(fn(*operands)).sum()  # noqa: E731
        grads = jax.jit(jax.grad(loss))(args)
        if fn is nnx.dot_product_attention:
            expected_grads = grads
        else:
            for actual_grad, expected_grad in zip(grads, expected_grads):
                np.testing.assert_array_equal(actual_grad, expected_grad)


def test_gpu_half_precision_falls_back_without_tokamax(monkeypatch):
    monkeypatch.setattr(attention.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(attention, "_get_tokamax_attention", lambda: None)
    args = _inputs(17, 17, 2, 16, jnp.bfloat16, True)
    actual = jax.jit(attention.dot_product_attention)(*args)
    expected = jax.jit(nnx.dot_product_attention)(*args)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("tokamax") is None,
    reason="requires a GPU and the tokamax extra",
)
@pytest.mark.parametrize(
    "query_length,key_length,head_dim,with_bias",
    [(37, 37, 16, False), (1, 29, 32, False), (31, 31, 24, True), (49, 49, 32, True)],
)
def test_tokamax_outputs_and_all_input_gradients(query_length, key_length, head_dim, with_bias):
    dtype = jnp.bfloat16
    args = _inputs(query_length, key_length, 3, head_dim, dtype, with_bias)
    assert attention._get_tokamax_attention() is not None
    actual = jax.jit(attention.dot_product_attention)(*args)
    expected = jax.jit(nnx.dot_product_attention)(*args)
    tolerance = 0.05
    np.testing.assert_allclose(
        actual.astype(jnp.float32), expected.astype(jnp.float32), rtol=tolerance, atol=tolerance / 5
    )
    actual_grads = jax.jit(
        jax.grad(
            lambda operands: jnp.square(
                attention.dot_product_attention(*operands).astype(jnp.float32)
            ).sum()
        )
    )(args)
    expected_grads = jax.jit(
        jax.grad(
            lambda operands: jnp.square(
                nnx.dot_product_attention(*operands).astype(jnp.float32)
            ).sum()
        )
    )(args)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert bool(jnp.isfinite(actual_grad).all())
        np.testing.assert_allclose(
            actual_grad.astype(jnp.float32),
            expected_grad.astype(jnp.float32),
            rtol=tolerance,
            atol=tolerance,
        )
