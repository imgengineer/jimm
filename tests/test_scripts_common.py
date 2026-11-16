"""Regression tests for standalone evaluation metrics and synchronization."""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jimm import scripts_common


@pytest.fixture
def eval_data():
    class Model(nnx.Module):
        def __init__(self):
            self.head = nnx.Linear(3, 8, rngs=nnx.Rngs(0))

        def __call__(self, images):
            return self.head(images.mean(axis=(1, 2)))

    model = Model()
    model.eval()
    images = jnp.asarray(np.random.default_rng(0).normal(size=(5, 2, 2, 3)), jnp.float32)
    labels = jnp.array([0, 2, 7, 1, 3], jnp.int32)
    batches = [
        {"image": images[start : start + 2], "label": labels[start : start + 2]}
        for start in range(0, 5, 2)
    ]
    return model, images, labels, batches


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("log_freq", [0, 1, 2])
def test_evaluate_metrics_without_implicit_host_sync(eval_data, monkeypatch, amp, log_freq):
    model, images, labels, batches = eval_data
    logits = np.asarray(scripts_common._forward_with_precision(model, images, amp))
    targets = np.asarray(labels)
    shifted = logits - logits.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(5), targets]
    predictions = np.argsort(-logits, axis=1)

    def forbid_scalar_transfer(value):
        raise AssertionError("evaluation must not read a device scalar each batch")

    monkeypatch.setattr(type(labels), "__int__", forbid_scalar_transfer)
    logger = Mock()
    metrics = scripts_common.evaluate(model, batches, amp=amp, log_freq=log_freq, logger=logger)

    assert metrics["samples"] == 5
    assert metrics["loss"] == pytest.approx(float(losses.mean()), rel=1e-6)
    assert metrics["top1"] == np.mean(predictions[:, 0] == targets)
    assert metrics["top5"] == np.mean(np.any(predictions[:, :5] == targets[:, None], axis=1))
    assert logger.info.call_count == (len(batches) // log_freq if log_freq else 0)


def test_evaluate_times_pending_device_work(eval_data, monkeypatch):
    model, _, _, batches = eval_data
    now = 0.0
    device_get = jax.device_get

    def wait_for_metrics(value):
        nonlocal now
        result = device_get(value)
        now += 1.0
        return result

    monkeypatch.setattr(scripts_common.jax, "device_get", wait_for_metrics)
    monkeypatch.setattr(scripts_common.time, "perf_counter", lambda: now)
    metrics = scripts_common.evaluate(model, batches, log_freq=0)

    assert metrics["seconds"] == 1.0
    assert metrics["samples_per_sec"] == 5.0


def test_evaluate_rejects_empty_loader(eval_data):
    model, _, _, _ = eval_data
    with pytest.raises(ValueError, match="no samples"):
        scripts_common.evaluate(model, [])
