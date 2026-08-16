"""Tests for train and eval mode behavior across special layers (Dropout, BatchNorm, DropPath)."""
import jax.numpy as jnp
from flax import nnx
import pytest
import jimm
from jimm.layers import DropPath


def test_droppath_train_and_eval_mode():
    dp = DropPath(rate=0.5, rngs=nnx.Rngs(0))
    x = jnp.ones((20, 8))

    # Eval mode: must be exact identity
    dp.eval()
    assert getattr(dp, "deterministic", False) is True
    y_eval = dp(x)
    assert jnp.allclose(y_eval, x)

    # Train mode: must randomly drop rows
    dp.train()
    assert getattr(dp, "deterministic", True) is False
    y_train = dp(x)
    assert not jnp.allclose(y_train, x)


def test_batchnorm_train_and_eval_mode():
    m = jimm.create_model("resnet18", num_classes=10, rngs=nnx.Rngs(0))
    x = jnp.ones((4, 224, 224, 3), jnp.float32)

    # Train mode: updates running stats and use_running_average is False
    m.train()
    assert m.bn1.use_running_average is False
    init_mean = m.bn1.mean[...]
    _ = m(x)
    after_train_mean = m.bn1.mean[...]
    assert not jnp.allclose(init_mean, after_train_mean)

    # Eval mode: does not update running stats and outputs are deterministic
    m.eval()
    assert m.bn1.use_running_average is True
    out_eval1 = m(x)
    out_eval2 = m(x)
    after_eval_mean = m.bn1.mean[...]
    assert jnp.allclose(after_train_mean, after_eval_mean)
    assert jnp.allclose(out_eval1, out_eval2)


def test_model_dropout_and_droppath_stochasticity():
    m_vit = jimm.create_model("vit_base_patch16_224", num_classes=10, drop_rate=0.2, drop_path_rate=0.2, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 224, 224, 3), jnp.float32)

    # Eval mode: deterministic
    m_vit.eval()
    out1 = m_vit(x)
    out2 = m_vit(x)
    assert jnp.allclose(out1, out2)

    # Train mode: stochastic
    m_vit.train()
    out_tr1 = m_vit(x)
    out_tr2 = m_vit(x)
    assert not jnp.allclose(out_tr1, out_tr2)
