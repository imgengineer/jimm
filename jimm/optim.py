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
    *,
    warmup_epochs: int | None = None,
    warmup_lr: float = 0.0,
    min_lr: float | None = None,
    eps: float = 1e-8,
    betas: tuple[float, float] = (0.9, 0.999),
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
    if warmup_epochs is not None and (not isinstance(warmup_epochs, int) or warmup_epochs < 0):
        raise ValueError("warmup_epochs must be a non-negative integer")
    if not np.all(np.isfinite((warmup_lr, eps))) or warmup_lr < 0 or eps <= 0:
        raise ValueError("warmup_lr must be non-negative and eps must be positive and finite")
    if min_lr is not None and (not np.isfinite(min_lr) or not 0 <= min_lr <= lr):
        raise ValueError("min_lr must be finite and satisfy 0 <= min_lr <= lr")
    if (
        len(betas) != 2
        or not np.all(np.isfinite(betas))
        or not all(0 <= beta < 1 for beta in betas)
    ):
        raise ValueError("betas must contain two finite values between 0 and 1 (exclusive)")
    total = epochs * steps_per_epoch
    if total == 1:
        schedule = optax.constant_schedule(lr)
    else:
        try:
            warmup_calc = max(int(total * warmup_ratio), 1)
        except (TypeError, ValueError):
            warmup_calc = 1
        warmup_steps = min(total - 1, 5 * steps_per_epoch, 10000, warmup_calc)
        if warmup_epochs is not None:
            warmup_steps = min(total - 1, warmup_epochs * steps_per_epoch)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=warmup_lr,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total,
            end_value=lr * min_lr_ratio if min_lr is None else min_lr,
        )
    tx = optax.clip_by_global_norm(clip_grad) if clip_grad > 0 else optax.identity()
    decay_mask = lambda params: jax.tree.map(lambda p: p.ndim >= 2, params)  # noqa: E731
    adamw = optax.adamw(
        schedule, weight_decay=weight_decay, mask=decay_mask, eps=eps, b1=betas[0], b2=betas[1]
    )
    return nnx.Optimizer(model, optax.chain(tx, adamw), wrt=nnx.Param)


# timm-compatible alias
create_optimizer = make_optimizer
