"""Device-side image preprocessing for compiled train and eval steps.

Loaders created with ``normalize=False`` keep images as uint8 through the worker,
IPC, and host-to-device stages, like timm's prefetcher. These helpers normalize
inside the compiled step and, for training, apply timm random erasing and batch
Mixup/CutMix on device without a NumPy round trip.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .data import IMAGENET_MEAN, IMAGENET_STD, MixupCutmix

__all__ = [
    "ImagePreprocess",
    "mixup_cutmix",
    "normalize_images",
    "preprocess_images",
    "random_erasing",
]


class ImagePreprocess(NamedTuple):
    """Device-side normalization and timm random erasing for uint8 NHWC batches.

    Like timm's prefetcher, loaders created with ``normalize=False`` keep images
    as uint8 through the worker, IPC, and host-to-device stages; train and eval
    steps normalize inside the compiled step, and training steps also erase.
    """

    mean: tuple[float, ...] = tuple(IMAGENET_MEAN.tolist())
    std: tuple[float, ...] = tuple(IMAGENET_STD.tolist())
    re_prob: float = 0.0
    re_mode: str = "const"
    re_count: int = 1


def normalize_images(images, preprocess):
    if images.dtype != jnp.uint8:
        raise ValueError("device preprocessing expects uint8 images from normalize=False loaders")
    # Same float32 constants as the host transform: x * (1 / (255 * std)) - mean / std.
    inv_std = (1.0 / np.asarray(preprocess.std, np.float32)).astype(np.float32)
    shift = (-np.asarray(preprocess.mean, np.float32) * inv_std).astype(np.float32)
    scale = (inv_std * np.float32(1.0 / 255.0)).astype(np.float32)
    return images.astype(jnp.float32) * scale + shift


def random_erasing(images, rng, preprocess, min_area=0.02, max_area=1 / 3, min_aspect=0.3):
    """timm ``RandomErasing`` applied independently to each normalized image."""
    batch, height, width, channels = images.shape
    count = max(1, int(preprocess.re_count))
    attempts = 10
    keys = jax.random.split(rng, 7)
    apply = jax.random.uniform(keys[0], (batch,)) < preprocess.re_prob
    boxes = (
        jax.random.randint(keys[1], (batch,), 1, count + 1)
        if count > 1
        else jnp.ones((batch,), jnp.int32)
    )
    # Draw every attempt up front and keep the first valid box, as timm's loop does.
    shape = (batch, count, attempts)
    area = jax.random.uniform(keys[2], shape, minval=min_area, maxval=max_area)
    area = area * (height * width) / boxes[:, None, None]
    log_aspect = math.log(min_aspect)
    aspect = jnp.exp(jax.random.uniform(keys[3], shape, minval=log_aspect, maxval=-log_aspect))
    box_h = jnp.round(jnp.sqrt(area * aspect)).astype(jnp.int32)
    box_w = jnp.round(jnp.sqrt(area / aspect)).astype(jnp.int32)
    valid = (box_h > 0) & (box_h < height) & (box_w > 0) & (box_w < width)
    first = jnp.argmax(valid, axis=-1)[..., None]
    box_h = jnp.take_along_axis(box_h, first, axis=-1)[..., 0]
    box_w = jnp.take_along_axis(box_w, first, axis=-1)[..., 0]
    top = jax.random.randint(keys[4], (batch, count), 0, jnp.maximum(height - box_h + 1, 1))
    left = jax.random.randint(keys[5], (batch, count), 0, jnp.maximum(width - box_w + 1, 1))
    active = apply[:, None] & valid.any(axis=-1) & (jnp.arange(count) < boxes[:, None])
    rows = jnp.arange(height)[:, None]
    cols = jnp.arange(width)[None, :]
    for index in range(count):
        top_i, left_i = top[:, index, None, None], left[:, index, None, None]
        mask = (
            active[:, index, None, None]
            & (rows >= top_i)
            & (rows < top_i + box_h[:, index, None, None])
            & (cols >= left_i)
            & (cols < left_i + box_w[:, index, None, None])
        )
        fill_key = jax.random.fold_in(keys[6], index)
        if preprocess.re_mode == "pixel":
            fill = jax.random.normal(fill_key, images.shape, images.dtype)
        elif preprocess.re_mode == "rand":
            fill = jax.random.normal(fill_key, (batch, 1, 1, channels), images.dtype)
        elif preprocess.re_mode == "mean":
            fill = images.mean(axis=(1, 2), keepdims=True)
        else:
            fill = jnp.zeros((), images.dtype)
        images = jnp.where(mask[..., None], fill, images)
    return images


def preprocess_images(images, preprocess, rng, training):
    if preprocess is None:
        return images
    images = normalize_images(images, preprocess)
    if training and preprocess.re_prob > 0:
        if rng is None:
            raise ValueError("rng is required when random erasing is enabled")
        # A folded key keeps Mixup/CutMix draws independent of erasing.
        images = random_erasing(images, jax.random.fold_in(rng, 1), preprocess)
    return images


def mixup_cutmix(images, labels, rng, config: MixupCutmix):
    """Apply batch Mixup/CutMix on device without a NumPy round trip."""
    if config.prob <= 0 or (config.mixup_alpha <= 0 and config.cutmix_alpha <= 0):
        return images, labels

    batch, height, width, _ = images.shape
    targets = labels if labels.ndim == 2 else jax.nn.one_hot(labels, config.num_classes)
    mixed_targets = targets
    if config.label_smoothing:
        mixed_targets = targets * (1.0 - config.label_smoothing)
        mixed_targets += config.label_smoothing / config.num_classes

    apply_key, switch_key, alpha_key, box_key, perm_key = jax.random.split(rng, 5)
    apply = True if config.prob >= 1 else jax.random.uniform(apply_key) < config.prob
    if config.cutmix_alpha > 0 and config.mixup_alpha > 0:
        use_cutmix = jax.random.uniform(switch_key) < config.switch_prob
    else:
        use_cutmix = config.cutmix_alpha > 0
    alpha = jnp.maximum(jnp.where(use_cutmix, config.cutmix_alpha, config.mixup_alpha), 1e-6)
    indices = (
        jnp.arange(batch - 1, -1, -1)
        if config.mode == "pair"
        else jax.random.permutation(perm_key, batch)
    )

    def mixup(_):
        if config.mode == "elem":
            lam = jax.random.beta(alpha_key, alpha, alpha, shape=(batch,))
            image_lam = lam[:, None, None, None]
            label_lam = lam[:, None]
        else:
            lam = jax.random.beta(alpha_key, alpha, alpha)
            image_lam = lam
            label_lam = lam
        mixed_images = image_lam * images + (1.0 - image_lam) * images[indices]
        mixed_labels = label_lam * mixed_targets + (1.0 - label_lam) * mixed_targets[indices]
        return mixed_images, mixed_labels

    def cutmix(_):
        box_keys = jax.random.split(box_key, 4)
        if config.mode == "elem":
            lam = jax.random.beta(alpha_key, alpha, alpha, shape=(batch,))
            if config.cutmix_minmax is None:
                ratio = jnp.sqrt(jnp.maximum(0.0, 1.0 - lam))
                box_h = jnp.rint(height * ratio).astype(jnp.int32)
                box_w = jnp.rint(width * ratio).astype(jnp.int32)
            else:
                low, high = config.cutmix_minmax
                box_h = jnp.rint(
                    height * jax.random.uniform(box_keys[0], (batch,), minval=low, maxval=high)
                ).astype(jnp.int32)
                box_w = jnp.rint(
                    width * jax.random.uniform(box_keys[1], (batch,), minval=low, maxval=high)
                ).astype(jnp.int32)
            center_y = jax.random.randint(box_keys[2], (batch,), 0, height)
            center_x = jax.random.randint(box_keys[3], (batch,), 0, width)
            top = jnp.maximum(0, center_y - box_h // 2)
            left = jnp.maximum(0, center_x - box_w // 2)
            bottom = jnp.minimum(height, center_y + box_h // 2)
            right = jnp.minimum(width, center_x + box_w // 2)
            yy = jnp.arange(height)[None, :, None]
            xx = jnp.arange(width)[None, None, :]
            mask = (
                (yy >= top[:, None, None])
                & (yy < bottom[:, None, None])
                & (xx >= left[:, None, None])
                & (xx < right[:, None, None])
            )
            actual_lam = 1.0 - (bottom - top) * (right - left) / (height * width)
            mixed_images = jnp.where(mask[..., None], images[indices], images)
            mixed_labels = (
                actual_lam[:, None] * mixed_targets
                + (1.0 - actual_lam[:, None]) * mixed_targets[indices]
            )
            return mixed_images, mixed_labels

        lam = jax.random.beta(alpha_key, alpha, alpha)
        if config.cutmix_minmax is None:
            ratio = jnp.sqrt(jnp.maximum(0.0, 1.0 - lam))
            box_h = jnp.rint(height * ratio).astype(jnp.int32)
            box_w = jnp.rint(width * ratio).astype(jnp.int32)
        else:
            low, high = config.cutmix_minmax
            box_h = jnp.rint(
                height * jax.random.uniform(box_keys[0], minval=low, maxval=high)
            ).astype(jnp.int32)
            box_w = jnp.rint(
                width * jax.random.uniform(box_keys[1], minval=low, maxval=high)
            ).astype(jnp.int32)
        center_y = jax.random.randint(box_keys[2], (), 0, height)
        center_x = jax.random.randint(box_keys[3], (), 0, width)
        top = jnp.maximum(0, center_y - box_h // 2)
        left = jnp.maximum(0, center_x - box_w // 2)
        bottom = jnp.minimum(height, center_y + box_h // 2)
        right = jnp.minimum(width, center_x + box_w // 2)
        yy = jnp.arange(height)[:, None]
        xx = jnp.arange(width)[None, :]
        mask = (yy >= top) & (yy < bottom) & (xx >= left) & (xx < right)
        actual_lam = 1.0 - (bottom - top) * (right - left) / (height * width)
        mixed_images = jnp.where(mask[None, ..., None], images[indices], images)
        mixed_labels = actual_lam * mixed_targets + (1.0 - actual_lam) * mixed_targets[indices]
        return mixed_images, mixed_labels

    def identity(_):
        # Return the smoothed targets so batches that skip mixing get the same
        # label smoothing as mixed ones (cross_entropy only smooths class ids).
        return images, mixed_targets

    return jax.lax.cond(
        apply,
        lambda _: jax.lax.cond(use_cutmix, cutmix, mixup, None),
        identity,
        None,
    )
