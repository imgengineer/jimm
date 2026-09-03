"""Loss functions for jimm models (mirrors timm.loss).

Supports:
  - Integer class labels (1D) with optional label smoothing
  - Soft probability targets (2D) from Mixup / CutMix
"""

import jax
import jax.numpy as jnp  # pyright: ignore[reportMissingImports]
import optax  # pyright: ignore[reportMissingImports]
from flax import nnx  # pyright: ignore[reportMissingImports]

__all__ = [
    "cross_entropy",
    "LabelSmoothingCrossEntropy",
    "SoftTargetCrossEntropy",
]


def _cross_entropy_losses(
    logits: jax.Array, labels: jax.Array, smoothing: float = 0.0
) -> jax.Array:
    """Compute per-example cross entropy losses with optional label smoothing."""
    if logits.ndim != 2 or labels.ndim not in (1, 2):
        raise ValueError("logits must be 2-D and labels must be 1-D or 2-D")
    expected = logits.shape if labels.ndim == 2 else (logits.shape[0],)
    if labels.shape != expected:
        raise ValueError(f"labels shape {labels.shape} must be {expected}")
    if labels.ndim == 1 and not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError("1-D labels must contain integer class ids")
    if labels.ndim == 2 and not jnp.issubdtype(labels.dtype, jnp.floating):
        raise ValueError("2-D labels must contain floating-point targets")
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("smoothing must be between 0 and 1")
    one_hot = labels if labels.ndim == logits.ndim else nnx.one_hot(labels, logits.shape[-1])
    one_hot = one_hot.astype(logits.dtype)
    if labels.ndim != logits.ndim:
        one_hot = one_hot * (1 - smoothing) + smoothing / logits.shape[-1]
    return optax.softmax_cross_entropy(logits, one_hot)


def cross_entropy(logits: jax.Array, labels: jax.Array, smoothing: float = 0.0) -> jax.Array:
    """Compute mean cross entropy loss over a batch."""
    return _cross_entropy_losses(logits, labels, smoothing).mean()


class LabelSmoothingCrossEntropy(nnx.Module):
    """Cross entropy loss with label smoothing (mirrors timm.loss.LabelSmoothingCrossEntropy)."""

    def __init__(self, smoothing: float = 0.1):
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("smoothing must be between 0 and 1")
        self.smoothing = smoothing

    def __call__(self, logits: jax.Array, target: jax.Array) -> jax.Array:
        return cross_entropy(logits, target, smoothing=self.smoothing)


class SoftTargetCrossEntropy(nnx.Module):
    """Cross entropy loss for soft probability targets (mirrors timm.loss.SoftTargetCrossEntropy)."""

    def __call__(self, logits: jax.Array, target: jax.Array) -> jax.Array:
        return cross_entropy(logits, target, smoothing=0.0)
