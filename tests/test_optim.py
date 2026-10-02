"""Tests for jimm.optim."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import jimm
from jimm.optim import create_optimizer, make_optimizer


def test_adamw_options_and_schedule_match_numeric_updates():
    model = nnx.Module()
    model.weight = nnx.Param(jnp.ones((1, 1)))
    optimizer = make_optimizer(
        model,
        lr=0.1,
        weight_decay=0.1,
        epochs=4,
        steps_per_epoch=1,
        warmup_epochs=2,
        warmup_lr=0.02,
        min_lr=0.001,
        eps=0.5,
        betas=(0.5, 0.5),
    )
    expected, first_moment, second_moment = 1.0, 0.0, 0.0
    for step, (lr, gradient) in enumerate(
        zip((0.02, 0.06, 0.1, 0.0505, 0.001), (2.0, 1.0, 0.0, 4.0, 1.0)), start=1
    ):
        first_moment = 0.5 * first_moment + 0.5 * gradient
        second_moment = 0.5 * second_moment + 0.5 * gradient**2
        correction = 1 - 0.5**step
        expected -= lr * (
            (first_moment / correction) / (np.sqrt(second_moment / correction) + 0.5)
            + 0.1 * expected
        )
        optimizer.update(
            model,
            jax.tree.map(lambda value: jnp.full_like(value, gradient), nnx.state(model, nnx.Param)),
        )
        np.testing.assert_allclose(model.weight[...], [[expected]], rtol=1e-6)


def test_zero_warmup_starts_at_peak_lr():
    model = nnx.Module()
    model.weight = nnx.Param(jnp.ones((1, 1)))
    optimizer = make_optimizer(
        model, lr=0.1, weight_decay=0.1, epochs=2, steps_per_epoch=2, warmup_epochs=0
    )
    optimizer.update(model, jax.tree.map(jnp.zeros_like, nnx.state(model, nnx.Param)))
    np.testing.assert_allclose(model.weight[...], [[0.99]])


@pytest.mark.parametrize(
    "options",
    [
        {"warmup_epochs": -1},
        {"warmup_epochs": 1.5},
        {"warmup_lr": float("nan")},
        {"eps": 0},
        {"min_lr": 0.1},
        {"betas": (0.9,)},
        {"betas": (0.9, 1.0)},
    ],
)
def test_new_optimizer_options_reject_invalid_values(options):
    with pytest.raises(ValueError):
        make_optimizer(
            nnx.Module(), lr=0.01, weight_decay=0.0, epochs=2, steps_per_epoch=2, **options
        )


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
