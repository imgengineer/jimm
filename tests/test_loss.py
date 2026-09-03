"""Tests for jimm.loss."""

import jax.numpy as jnp
import pytest

from jimm.loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    cross_entropy,
)


def test_cross_entropy_1d_labels():
    logits = jnp.array([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    loss = cross_entropy(logits, labels)
    assert loss.ndim == 0
    assert float(loss) > 0


def test_cross_entropy_label_smoothing():
    logits = jnp.array([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    loss_smooth = cross_entropy(logits, labels, smoothing=0.1)
    loss_plain = cross_entropy(logits, labels, smoothing=0.0)
    assert float(loss_smooth) != float(loss_plain)


def test_cross_entropy_2d_soft_labels():
    logits = jnp.array([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]], dtype=jnp.float32)
    targets = jnp.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=jnp.float32)
    loss = cross_entropy(logits, targets)
    assert loss.ndim == 0
    assert float(loss) > 0


def test_cross_entropy_invalid_inputs():
    logits = jnp.ones((2, 3), dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    # Logits not 2D
    with pytest.raises(ValueError, match="logits must be 2-D"):
        cross_entropy(jnp.ones((2, 3, 1)), labels)

    # Labels not 1D or 2D
    with pytest.raises(ValueError, match="labels must be 1-D or 2-D"):
        cross_entropy(logits, jnp.ones((2, 3, 1)))

    # Labels batch mismatch
    with pytest.raises(ValueError, match="labels shape"):
        cross_entropy(logits, jnp.array([0]))

    # 1D labels non-integer
    with pytest.raises(ValueError, match="1-D labels must contain integer"):
        cross_entropy(logits, jnp.array([0.0, 1.0]))

    # 2D labels non-floating
    with pytest.raises(ValueError, match="2-D labels must contain floating-point"):
        cross_entropy(logits, jnp.ones((2, 3), dtype=jnp.int32))

    # Invalid smoothing range
    with pytest.raises(ValueError, match="smoothing must be between"):
        cross_entropy(logits, labels, smoothing=-0.1)
    with pytest.raises(ValueError, match="smoothing must be between"):
        cross_entropy(logits, labels, smoothing=1.5)


def test_label_smoothing_cross_entropy_module():
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    assert loss_fn.smoothing == 0.1

    with pytest.raises(ValueError, match="smoothing must be between"):
        LabelSmoothingCrossEntropy(smoothing=1.5)

    logits = jnp.array([[2.0, 0.0], [0.0, 2.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    l1 = loss_fn(logits, labels)
    l2 = cross_entropy(logits, labels, smoothing=0.1)
    assert float(l1) == pytest.approx(float(l2))


def test_soft_target_cross_entropy_module():
    loss_fn = SoftTargetCrossEntropy()
    logits = jnp.array([[2.0, 0.0], [0.0, 2.0]], dtype=jnp.float32)
    targets = jnp.array([[0.9, 0.1], [0.1, 0.9]], dtype=jnp.float32)
    l1 = loss_fn(logits, targets)
    l2 = cross_entropy(logits, targets, smoothing=0.0)
    assert float(l1) == pytest.approx(float(l2))
