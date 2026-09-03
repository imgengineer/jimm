"""Optimizer factory and weight-decay parameter grouping (mirrors timm.optim)."""

import jax  # pyright: ignore[reportMissingImports]
import numpy as np
import optax  # pyright: ignore[reportMissingImports]
from flax import nnx  # pyright: ignore[reportMissingImports]

__all__ = ["make_optimizer", "create_optimizer"]


def make_optimizer(
    model: nnx.Module,
    lr: float,
    weight_decay: float,
    epochs: int,
    steps_per_epoch: int,
    clip_grad: float = 0.0,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.01,
) -> nnx.Optimizer:
    """AdamW (warmup + cosine decay) with timm-style weight-decay grouping.

    Following timm's default (`param_groups_weight_decay`), weight decay only
    applies to parameters with ndim >= 2 (conv/linear kernels); 1-D parameters
    (biases, norm scales) are exempt.
    """
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if not np.all(np.isfinite((lr, weight_decay, clip_grad, warmup_ratio, min_lr_ratio))):
        raise ValueError("optimizer settings must be finite")
    if lr < 0 or weight_decay < 0 or clip_grad < 0:
        raise ValueError("lr, weight_decay, and clip_grad must be non-negative")
    if not 0 <= warmup_ratio <= 1 or not 0 <= min_lr_ratio <= 1:
        raise ValueError("warmup_ratio and min_lr_ratio must be between 0 and 1")
    total = epochs * steps_per_epoch
    if total == 1:
        schedule = optax.constant_schedule(lr)
    else:
        try:
            warmup_calc = max(int(total * warmup_ratio), 1)
        except (TypeError, ValueError):
            warmup_calc = 1
        warmup_steps = min(total - 1, 5 * steps_per_epoch, 10000, warmup_calc)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total,
            end_value=lr * min_lr_ratio,
        )
    tx = optax.clip_by_global_norm(clip_grad) if clip_grad > 0 else optax.identity()
    decay_mask = lambda params: jax.tree.map(lambda p: p.ndim >= 2, params)  # noqa: E731
    adamw = optax.adamw(schedule, weight_decay=weight_decay, mask=decay_mask)
    return nnx.Optimizer(model, optax.chain(tx, adamw), wrt=nnx.Param)


# timm-compatible alias
create_optimizer = make_optimizer
