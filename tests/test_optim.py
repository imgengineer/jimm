"""Tests for jimm.optim."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import jimm
from jimm.optim import create_optimizer, make_optimizer


def test_make_optimizer_basic():
    model = jimm.create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    opt = make_optimizer(model, lr=1e-3, weight_decay=1e-2, epochs=5, steps_per_epoch=10)
    assert opt is not None

    # Step optimizer
    grads = jax.tree.map(jnp.zeros_like, nnx.state(model, nnx.Param))
    opt.update(model, grads)


def test_create_optimizer_alias():
    model = jimm.create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    opt = create_optimizer(model, lr=1e-3, weight_decay=1e-2, epochs=1, steps_per_epoch=1)
    assert opt is not None


def test_optimizer_single_step():
    model = jimm.create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    opt = make_optimizer(model, lr=1e-3, weight_decay=1e-2, epochs=1, steps_per_epoch=1)
    assert opt is not None


def test_optimizer_clipping():
    model = jimm.create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    opt = make_optimizer(
        model, lr=1e-3, weight_decay=1e-2, epochs=2, steps_per_epoch=2, clip_grad=1.0
    )
    assert opt is not None


def test_optimizer_validation_errors():
    model = jimm.create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))

    with pytest.raises(ValueError, match="epochs and steps_per_epoch must be positive"):
        make_optimizer(model, lr=1e-3, weight_decay=1e-2, epochs=0, steps_per_epoch=10)
    with pytest.raises(ValueError, match="epochs and steps_per_epoch must be positive"):
        make_optimizer(model, lr=1e-3, weight_decay=1e-2, epochs=5, steps_per_epoch=0)

    with pytest.raises(ValueError, match="optimizer settings must be finite"):
        make_optimizer(model, lr=float("nan"), weight_decay=1e-2, epochs=5, steps_per_epoch=10)

    with pytest.raises(ValueError, match="must be non-negative"):
        make_optimizer(model, lr=-1e-3, weight_decay=1e-2, epochs=5, steps_per_epoch=10)
    with pytest.raises(ValueError, match="must be non-negative"):
        make_optimizer(model, lr=1e-3, weight_decay=-1e-2, epochs=5, steps_per_epoch=10)
    with pytest.raises(ValueError, match="must be non-negative"):
        make_optimizer(
            model, lr=1e-3, weight_decay=1e-2, epochs=5, steps_per_epoch=10, clip_grad=-1.0
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        make_optimizer(
            model, lr=1e-3, weight_decay=1e-2, epochs=5, steps_per_epoch=10, warmup_ratio=1.5
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        make_optimizer(
            model, lr=1e-3, weight_decay=1e-2, epochs=5, steps_per_epoch=10, min_lr_ratio=1.5
        )
