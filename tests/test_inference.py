"""Numerical checks for cached, compiled top-k inference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from inference import _predict
from jimm.train import _forward_with_precision


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("output_type", ["prob", "logit"])
@pytest.mark.parametrize("topk", [1, 5])
def test_cached_predictions_match_separate_operations(amp, output_type, topk):
    class Model(nnx.Module):
        def __init__(self):
            rngs = nnx.Rngs(0)
            self.bn = nnx.BatchNorm(3, rngs=rngs)
            self.head = nnx.Linear(3, 8, rngs=rngs)
            self.drop = nnx.Dropout(0.5, rngs=rngs)

        def __call__(self, images):
            return self.head(self.drop(self.bn(images)).mean(axis=(1, 2)))

    model = Model()
    model.eval()
    predict = nnx.cached_partial(_predict, model, graph=True, graph_updates=True)
    images = jax.random.normal(jax.random.key(1), (5, 4, 4, 3))
    for start in range(0, len(images), 2):
        batch = images[start : start + 2]
        logits = _forward_with_precision(model, batch, amp)
        scores = jax.nn.softmax(logits, axis=-1) if output_type == "prob" else logits
        expected_scores, expected_indices = jax.lax.top_k(scores, topk)
        actual_scores, actual_indices = predict(batch, amp=amp, topk=topk, output_type=output_type)
        np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(actual_indices, expected_indices)
    assert model.head.dtype is None
    np.testing.assert_array_equal(model.bn.mean[...], jnp.zeros(3))
