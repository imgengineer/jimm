"""Distributed training entry: Optax + Flax NNX + Grain + Orbax.

Supports:
  - Single-device training (1 GPU / CPU)
  - Single-node multi-GPU data-parallel training (DDP with SPMD Data Mesh)
  - Multi-node multi-GPU distributed data-parallel training (Multi-host JAX + Grain Sharding)
  - FSDP (Fully Sharded Data Parallel / ZeRO-3 parameter and optimizer state sharding)
  - MaxText / MaxDiffusion style async Host-to-Device prefetching (double buffering)
  - XLA compilation timing, steady-state throughput (img/s, step_time) & JAX profiler integration

Examples:
  # Standard DDP on all available GPUs on the node:
  python -m jimm.train --model resnet50 --data-dir /path/to/imagenet --epochs 90

  # Keep only the 3 most recent epoch checkpoints (plus the best-val-accuracy one):
  python -m jimm.train --model resnet50 --data-dir /path/to/imagenet --epochs 90 --max-to-keep 3

  # Resume an interrupted run from the latest checkpoint under --output:
  python -m jimm.train --model resnet50 --data-dir /path/to/imagenet --epochs 90 --resume

  # FSDP mode (ZeRO-3: shards weights and optimizer states across devices to save memory):
  python -m jimm.train --model eva02_large_patch14_224 --data-dir /path/to/imagenet --fsdp

  # Multi-node training (e.g. Node 0 of 2 nodes, 8 GPUs each):
  python -m jimm.train --model convnext_tiny --data-dir /path/to/imagenet \\
      --dist-coordinator-address 192.168.1.100:12345 --dist-num-processes 2 --dist-process-id 0

  # Profile execution with JAX profiler / Perfetto:
  python -m jimm.train --model resnet50 --data-dir /path/to/imagenet --profile-step 5 --profile-dir ./profiles
"""

import argparse
import ast
import collections
import functools
import gc
import math
import os
import time
from typing import NamedTuple

import jax  # pyright: ignore[reportMissingImports]
import jax.numpy as jnp  # pyright: ignore[reportMissingImports]
import numpy as np
import optax  # pyright: ignore[reportMissingImports]
import yaml
from flax import nnx  # pyright: ignore[reportMissingImports]

from .attention import set_attention_autotuning
from .checkpoint import CheckpointManager
from .data import IMAGENET_MEAN, IMAGENET_STD, MixupCutmix, create_loader
from .loss import _cross_entropy_losses, cross_entropy
from .models.nfnet import ScaledStdConv
from .optim import create_optimizer, make_optimizer
from .registry import create_model

__all__ = [
    "ImagePreprocess",
    "StepMetrics",
    "init_distributed",
    "fsdp_shard_model",
    "prefetch_to_device",
    "cross_entropy",
    "make_optimizer",
    "create_optimizer",
    "train_step",
    "train_step_with_metrics",
    "eval_step",
    "make_cached_train_step",
    "make_cached_eval_step",
    "main",
]


class StepMetrics(NamedTuple):
    """Structured metrics returned by training step."""

    loss: jax.Array
    accuracy: jax.Array
    grad_norm: jax.Array


def init_distributed(coordinator_address=None, num_processes=None, process_id=None):
    """Initializes multi-node JAX distributed cluster if configured."""
    if coordinator_address is not None:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )
    elif "JAX_COORDINATOR_ADDRESS" in os.environ or "SLURM_JOB_ID" in os.environ:
        try:
            jax.distributed.initialize()
        except Exception as e:
            if jax.process_index() == 0:
                print(f"[Warning] Auto jax.distributed.initialize() skipped: {e}")


def fsdp_shard_model(model_or_opt, mesh, mesh_axis="data"):
    """Shards parameters and optimizer states across the mesh axis (ZeRO-3 / FSDP)."""
    num_devices = len(mesh.devices)
    P = jax.sharding.PartitionSpec
    for path, node in nnx.graph.iter_graph(model_or_opt):
        if isinstance(node, nnx.Variable):
            val = node.get_value()
            if isinstance(val, (jax.Array, np.ndarray)) and val.ndim >= 1:
                if val.shape[0] % num_devices == 0:
                    spec = P(mesh_axis, *(None,) * (val.ndim - 1))
                elif val.ndim == 4 and val.shape[-1] % num_devices == 0:
                    # Shard 4D Conv kernels (H, W, In, Out) along the output channel axis
                    spec = P(None, None, None, mesh_axis)
                else:
                    spec = P()  # replicate if leading dimension is not evenly divisible
            else:
                continue
            sharding = jax.sharding.NamedSharding(mesh, spec)
            node.set_value(jax.device_put(val, sharding))


def prefetch_to_device(
    data_iter, data_sharding, label_sharding, prefetch_size=2, mask_sharding=None
):
    """Asynchronously prefetches and shards host data onto devices (double buffering).

    MaxText/MaxDiffusion pattern: overlaps host CPU data loading/decoding & Host-to-Device (H2D)
    transfer with on-device accelerator execution, preventing GPU/TPU idle starvation bubbles.
    """
    if (
        isinstance(prefetch_size, bool)
        or not isinstance(prefetch_size, (int, np.integer))
        or prefetch_size <= 0
    ):
        raise ValueError("prefetch_size must be a positive integer")
    queue = collections.deque()

    def _put(batch):
        images = jax.make_array_from_process_local_data(data_sharding, batch["image"])
        labels = jax.make_array_from_process_local_data(label_sharding, batch["label"])
        if mask_sharding is not None:
            valid = jax.make_array_from_process_local_data(mask_sharding, batch["valid"])
            return images, labels, valid
        return images, labels

    # Prime the queue
    for _ in range(prefetch_size):
        try:
            batch = next(data_iter)
            queue.append(_put(batch))
        except StopIteration:
            break

    # Yield and keep buffer populated
    while queue:
        item = queue.popleft()
        try:
            batch = next(data_iter)
            queue.append(_put(batch))
        except StopIteration:
            pass
        yield item


def _accuracy(logits, labels):
    target = jnp.argmax(labels, axis=-1) if labels.ndim == logits.ndim else labels
    return jnp.mean(jnp.argmax(logits, -1) == target)


def _validate_batch(images, labels):
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have NHWC shape with three channels")
    if labels.ndim not in (1, 2) or images.shape[0] != labels.shape[0]:
        raise ValueError("labels must be 1-D or 2-D and match the image batch size")


def _mean_metrics(losses, accuracies, counts=None):
    """Return finite mean loss/accuracy, optionally weighted by batch size."""
    if not losses or len(losses) != len(accuracies):
        raise ValueError("losses and accuracies must be non-empty and have equal length")
    metrics = jnp.stack((jnp.stack(losses), jnp.stack(accuracies)))
    if counts is None:
        means = metrics.mean(axis=1)
    else:
        weights = np.asarray(counts, dtype=np.float32)
        if (
            weights.shape != (len(losses),)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError("counts must contain one positive finite value per batch")
        device_weights = jnp.asarray(weights)
        means = (metrics * device_weights).sum(axis=1) / device_weights.sum()
    values = np.asarray(jax.device_get(means))
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite epoch metrics")
    try:
        val0 = float(values[0])
        val1 = float(values[1])
    except (TypeError, ValueError) as exc:
        raise FloatingPointError("non-finite epoch metrics") from exc
    return val0, val1


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


def _normalize_images(images, preprocess):
    if images.dtype != jnp.uint8:
        raise ValueError("device preprocessing expects uint8 images from normalize=False loaders")
    # Same float32 constants as the host transform: x * (1 / (255 * std)) - mean / std.
    inv_std = (1.0 / np.asarray(preprocess.std, np.float32)).astype(np.float32)
    shift = (-np.asarray(preprocess.mean, np.float32) * inv_std).astype(np.float32)
    scale = (inv_std * np.float32(1.0 / 255.0)).astype(np.float32)
    return images.astype(jnp.float32) * scale + shift


def _random_erasing_jax(images, rng, preprocess, min_area=0.02, max_area=1 / 3, min_aspect=0.3):
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


def _preprocess_images(images, preprocess, rng, training):
    if preprocess is None:
        return images
    images = _normalize_images(images, preprocess)
    if training and preprocess.re_prob > 0:
        if rng is None:
            raise ValueError("rng is required when random erasing is enabled")
        # A folded key keeps Mixup/CutMix draws independent of erasing.
        images = _random_erasing_jax(images, jax.random.fold_in(rng, 1), preprocess)
    return images


def _mixup_cutmix_jax(images, labels, rng, config):
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


def _forward_with_precision(model, images, amp):
    if not amp:
        return model(images)
    # Override compute dtypes only. Master parameters and normalization statistics
    # stay in FP32; restoring static attributes also keeps cached NNX graphs valid.
    layers = [
        (node, node.dtype)
        for _, node in nnx.graph.iter_graph(model)
        if isinstance(node, (nnx.Conv, nnx.ConvTranspose, nnx.Linear, nnx.Einsum, ScaledStdConv))
    ]
    try:
        for layer, _ in layers:
            layer.dtype = jnp.bfloat16
        return model(images.astype(jnp.bfloat16)).astype(jnp.float32)
    finally:
        for layer, dtype in layers:
            layer.dtype = dtype


def _train_step_impl(
    model,
    optimizer,
    images,
    labels,
    smoothing=0.0,
    amp=False,
    mixup=None,
    rng=None,
    preprocess=None,
):
    _validate_batch(images, labels)
    images = _preprocess_images(images, preprocess, rng, training=True)
    if mixup is not None:
        if rng is None:
            raise ValueError("rng is required when mixup or cutmix is enabled")
        images, labels = _mixup_cutmix_jax(images, labels, rng, mixup)

    def loss_fn(model):
        logits = _forward_with_precision(model, images, amp)
        return cross_entropy(logits, labels, smoothing), logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    acc = _accuracy(logits, labels)
    return loss, logits, grads, acc


@nnx.jit(static_argnames=("smoothing", "amp", "mixup", "preprocess"))
def train_step(
    model,
    optimizer,
    images,
    labels,
    smoothing=0.0,
    amp=False,
    mixup=None,
    rng=None,
    preprocess=None,
):
    loss, _, _, acc = _train_step_impl(
        model, optimizer, images, labels, smoothing, amp, mixup, rng, preprocess
    )
    return loss, acc


@nnx.jit(static_argnames=("smoothing", "amp", "mixup", "preprocess"))
def train_step_with_metrics(
    model,
    optimizer,
    images,
    labels,
    smoothing=0.0,
    amp=False,
    mixup=None,
    rng=None,
    preprocess=None,
):
    """Executes a training step, computing pre-clipping grad_norm for monitoring."""
    loss, _, grads, acc = _train_step_impl(
        model, optimizer, images, labels, smoothing, amp, mixup, rng, preprocess
    )
    grad_norm = (
        optax.tree.norm(grads)
        if hasattr(optax, "tree") and hasattr(optax.tree, "norm")
        else optax.global_norm(grads)
    )
    return StepMetrics(loss=loss, accuracy=acc, grad_norm=grad_norm)


@nnx.jit(static_argnames=("amp", "preprocess"))
def eval_step(model, images, labels, amp=False, valid=None, preprocess=None):
    """Return mean loss and accuracy; ``preprocess`` normalizes uint8 images only."""
    _validate_batch(images, labels)
    if valid is not None and (valid.ndim != 1 or valid.shape != (images.shape[0],)):
        raise ValueError("valid mask must be 1-D and match the image batch size")
    images = _preprocess_images(images, preprocess, None, training=False)
    logits = _forward_with_precision(model, images, amp)
    losses = _cross_entropy_losses(logits, labels)
    if valid is None:
        return losses.mean(), _accuracy(logits, labels)
    weights = valid.astype(logits.dtype)
    valid_count = weights.sum()
    target = jnp.argmax(labels, axis=-1) if labels.ndim == logits.ndim else labels
    correct = (jnp.argmax(logits, -1) == target).astype(logits.dtype)
    return ((losses * weights).sum() / valid_count, (correct * weights).sum() / valid_count)


def make_cached_train_step(model, optimizer, amp=False, mixup=None, preprocess=None):
    """Create one cached JIT train step with AMP, batch mixing, and preprocessing bound."""
    return nnx.jit_partial(
        functools.partial(train_step.__wrapped__, amp=amp, mixup=mixup, preprocess=preprocess),
        model,
        optimizer,
        # jit_partial packs bound arguments into one leading argument; positional
        # smoothing follows images and labels at index 3 of the compiled call.
        static_argnums=(3,),
        static_argnames=("smoothing", "amp", "mixup", "preprocess"),
        graph=True,
        graph_updates=False,
    )


def make_cached_eval_step(model, amp=False, preprocess=None):
    """Create one cached JIT eval step with AMP and preprocessing bound at construction."""
    return nnx.jit_partial(
        functools.partial(eval_step.__wrapped__, amp=amp, preprocess=preprocess),
        model,
        static_argnames=("amp", "preprocess"),
        graph=True,
        graph_updates=False,
    )


class _ParseModelKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = dict(getattr(namespace, self.dest))
        for item in values:
            key, separator, value = item.partition("=")
            if not separator or not key:
                raise argparse.ArgumentError(self, "model kwargs must use KEY=VALUE")
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            kwargs[key] = value
        setattr(namespace, self.dest, kwargs)


def _parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-c", "--config", default="", help="YAML configuration file")
    config, remaining = config_parser.parse_known_args(argv)
    p = argparse.ArgumentParser(
        prog="jimm.train",
        description="JAX image classification training with Flax NNX and Grain",
        parents=[config_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("data", nargs="?", metavar="DIR", help="dataset root; --data-dir also works")

    group = p.add_argument_group("Dataset parameters")
    group.add_argument("--data-dir", help="ImageFolder dataset root")
    group.add_argument("--train-split", default="train", help="training subdirectory")
    group.add_argument("--val-split", default="val", help="validation subdirectory")

    group = p.add_argument_group("Model and input parameters")
    group.add_argument("--model", default="resnet50", help="model architecture name")
    group.add_argument("--num-classes", type=int, default=1000)
    group.add_argument("--img-size", type=int, default=224)
    group.add_argument("--input-size", type=int, nargs=3, help="square RGB input: 3 H H")
    group.add_argument("--crop-pct", type=float, default=0.875)
    group.add_argument("--mean", type=float, nargs=3, default=IMAGENET_MEAN.tolist())
    group.add_argument("--std", type=float, nargs=3, default=IMAGENET_STD.tolist())
    group.add_argument(
        "--interpolation",
        default="bilinear",
        choices=("nearest", "bilinear", "bicubic", "lanczos", "area"),
    )
    group.add_argument("-b", "--batch-size", type=int, default=128, help="batch size per host")
    group.add_argument(
        "-vb", "--validation-batch-size", type=int, help="validation batch size per host"
    )
    group.add_argument(
        "--gp", "--global-pool", dest="global_pool", help="model global pooling override"
    )
    group.add_argument(
        "--model-kwargs", nargs="*", default={}, action=_ParseModelKwargs, metavar="KEY=VALUE"
    )
    group.add_argument(
        "--initial-checkpoint", default="", help="initial model weights in jimm NPZ format"
    )
    group.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Orbax manager directory; omit path to use the output directory",
    )

    group = p.add_argument_group("Optimizer parameters")
    group.add_argument("--opt", default="adamw", choices=("adamw",))
    group.add_argument("--opt-eps", type=float, default=1e-8)
    group.add_argument("--opt-betas", type=float, nargs=2, default=(0.9, 0.999))
    group.add_argument("--weight-decay", type=float, default=0.05)
    group.add_argument(
        "--clip-grad", type=float, default=1.0, help="global norm clipping; 0 disables"
    )

    group = p.add_argument_group("Learning rate schedule parameters")
    group.add_argument("--sched", default="cosine", choices=("cosine",))
    group.add_argument("--lr", type=float, default=5e-4)
    group.add_argument("--epochs", type=int, default=90)
    group.add_argument(
        "--warmup-epochs", type=int, help="override automatic warmup capped at five epochs"
    )
    group.add_argument("--warmup-lr", type=float, default=0.0)
    group.add_argument("--min-lr", type=float, help="minimum LR; default is 1%% of --lr")

    group = p.add_argument_group("Augmentation and regularization parameters")
    group.add_argument("--no-aug", action="store_true", help="disable training image augmentation")
    group.add_argument("--train-crop-mode", default="rrc", choices=("rrc", "rkrc", "rkrr"))
    group.add_argument("--scale", type=float, nargs=2, default=(0.08, 1.0))
    group.add_argument("--ratio", type=float, nargs=2, default=(3.0 / 4.0, 4.0 / 3.0))
    group.add_argument("--hflip", type=float, default=0.5)
    group.add_argument("--vflip", type=float, default=0.0)
    group.add_argument("--color-jitter", type=float, default=0.4)
    group.add_argument("--color-jitter-prob", type=float)
    group.add_argument("--grayscale-prob", type=float, default=0.0)
    group.add_argument("--gaussian-blur-prob", type=float, default=0.0)
    group.add_argument(
        "--aa", "--auto-augment", dest="auto_augment", help="timm AutoAugment policy"
    )
    group.add_argument("--reprob", type=float, default=0.2, help="random erasing probability")
    group.add_argument("--remode", default="const", choices=("const", "rand", "pixel"))
    group.add_argument("--recount", type=int, default=1)
    group.add_argument("--mixup", "--mixup-alpha", dest="mixup_alpha", type=float, default=0.0)
    group.add_argument("--cutmix", "--cutmix-alpha", dest="cutmix_alpha", type=float, default=0.0)
    group.add_argument("--cutmix-minmax", type=float, nargs=2)
    group.add_argument("--mixup-prob", type=float, default=1.0)
    group.add_argument("--mixup-switch-prob", type=float, default=0.5)
    group.add_argument("--mixup-mode", choices=("batch", "pair", "elem"), default="batch")
    group.add_argument("--smoothing", type=float, default=0.1)
    group.add_argument("--drop", type=float, default=0.0)
    group.add_argument("--drop-path", type=float, default=0.0)
    group.add_argument(
        "--train-interpolation",
        default="random",
        choices=("random", "nearest", "bilinear", "bicubic", "lanczos", "area"),
    )

    group = p.add_argument_group("Miscellaneous parameters")
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("-j", "--workers", type=int, default=4, help="Grain workers per host")
    group.add_argument("--output", default="./output", help="output root directory")
    group.add_argument(
        "--experiment", default="", help="output subdirectory; defaults to model name"
    )
    group.add_argument(
        "--checkpoint-hist",
        "--max-to-keep",
        dest="max_to_keep",
        type=int,
        help="checkpoint retention count",
    )
    group.add_argument("--log-interval", type=int, default=50)
    group.add_argument("--steps-per-epoch", type=int, help="cap training steps per epoch")

    group = p.add_argument_group("JAX device and distributed parameters")
    group.add_argument(
        "--prefetch", type=int, default=2, help="batches to prefetch to device (double buffering)"
    )
    group.add_argument(
        "--fsdp",
        action="store_true",
        default=False,
        help="enable FSDP (ZeRO-3 style parameter and optimizer state sharding)",
    )
    group.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable AMP bfloat16 compute on Tensor Cores",
    )
    group.add_argument(
        "--attn-autotune",
        action="store_true",
        help="autotune bfloat16 Tokamax attention kernels for new shapes (slower first compile)",
    )
    group.add_argument(
        "--profile-step", type=int, default=None, help="step index to trigger JAX profiler trace"
    )
    group.add_argument(
        "--profile-dir", type=str, default=None, help="directory to store JAX profile traces"
    )

    group.add_argument(
        "--dist-coordinator-address",
        type=str,
        default=None,
        help="IP:port of master coordinator for multi-node training (e.g. 192.168.1.1:12345)",
    )
    group.add_argument(
        "--dist-num-processes", type=int, default=None, help="total number of nodes/hosts"
    )
    group.add_argument(
        "--dist-process-id",
        type=int,
        default=None,
        help="rank/id of current node (0..num_processes-1)",
    )
    if config.config:
        try:
            with open(config.config) as file:
                defaults = yaml.safe_load(file)
        except (OSError, yaml.YAMLError) as error:
            p.error(f"unable to load config: {error}")
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            p.error("config must contain a mapping of argument names to values")
        actions = {action.dest: action for action in p._actions}
        actions.update(
            {
                option.lstrip("-").replace("-", "_"): action
                for action in p._actions
                for option in action.option_strings
                if not (
                    isinstance(action, argparse.BooleanOptionalAction)
                    and option.startswith("--no-")
                )
            }
        )
        converted = {}
        for key, value in defaults.items():
            action = actions.get(key.replace("-", "_")) if isinstance(key, str) else None
            if action is None or action.dest in ("help", "config"):
                p.error(f"unknown config argument: {key}")
            if action.dest in converted:
                p.error(f"duplicate config argument: {key}")
            try:
                if value is None and action.default is not None:
                    raise ValueError("must not be null")
                if action.dest == "model_kwargs":
                    if not isinstance(value, dict) or not all(
                        isinstance(name, str) for name in value
                    ):
                        raise ValueError("must be a mapping")
                elif isinstance(
                    action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction)
                ):
                    if not isinstance(value, bool):
                        raise ValueError("must be a boolean")
                elif action.dest == "resume":
                    if not isinstance(value, (bool, str)):
                        raise ValueError("must be a boolean or checkpoint directory")
                elif value is not None:
                    multiple = isinstance(action.nargs, int) or action.nargs in ("+", "*")
                    values = value if multiple else [value]
                    if not isinstance(values, (list, tuple)) or (
                        isinstance(action.nargs, int) and len(values) != action.nargs
                    ):
                        raise ValueError(f"must contain {action.nargs} values")
                    if action.type is not None:
                        if any(
                            isinstance(item, bool)
                            or (
                                action.type is int
                                and isinstance(item, float)
                                and not item.is_integer()
                            )
                            for item in values
                        ):
                            raise ValueError("has an invalid numeric value")
                        values = [action.type(item) for item in values]
                    elif any(not isinstance(item, str) for item in values):
                        raise ValueError("must be a string")
                    if action.choices is not None and any(
                        item not in action.choices for item in values
                    ):
                        raise ValueError(f"must be one of {action.choices}")
                    value = values if multiple else values[0]
            except (TypeError, ValueError, OverflowError) as error:
                p.error(f"invalid config argument {key}: {error}")
            converted[action.dest] = value
        p.set_defaults(**converted)
    p.set_defaults(config=config.config)
    args = p.parse_args(remaining)
    args.data_dir = args.data_dir or args.data
    if not args.data_dir:
        p.error("a dataset root is required: --data-dir DIR or positional DIR")
    if isinstance(args.resume, str) and args.resume and not os.path.isdir(args.resume):
        p.error(f"resume directory does not exist: {args.resume}")
    if args.input_size is not None:
        channels, height, width = args.input_size
        if channels != 3 or height != width or height <= 0:
            p.error("--input-size must specify square RGB input: 3 H H")
        args.img_size = height
    if args.validation_batch_size is None:
        args.validation_batch_size = args.batch_size

    for name in (
        "epochs",
        "batch_size",
        "validation_batch_size",
        "img_size",
        "num_classes",
        "prefetch",
        "log_interval",
        "recount",
    ):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0:
        p.error("--workers must be non-negative")
    if args.seed < 0:
        p.error("--seed must be non-negative")
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        p.error("--steps-per-epoch must be positive")
    if args.max_to_keep is not None and args.max_to_keep <= 0:
        p.error("--max-to-keep must be positive")
    for name in (
        "lr",
        "weight_decay",
        "clip_grad",
        "mixup_alpha",
        "cutmix_alpha",
        "color_jitter",
        "warmup_lr",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            p.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    for name in (
        "smoothing",
        "hflip",
        "vflip",
        "grayscale_prob",
        "gaussian_blur_prob",
        "mixup_prob",
        "mixup_switch_prob",
        "reprob",
        "drop",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            p.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if not np.isfinite(args.drop_path) or not 0 <= args.drop_path < 1:
        p.error("--drop-path must be between 0 (inclusive) and 1 (exclusive)")
    if args.color_jitter_prob is not None and (
        not np.isfinite(args.color_jitter_prob) or not 0 <= args.color_jitter_prob <= 1
    ):
        p.error("--color-jitter-prob must be between 0 and 1")
    if not np.isfinite(args.crop_pct) or not 0 < args.crop_pct <= 1:
        p.error("--crop-pct must be between 0 (exclusive) and 1 (inclusive)")
    for name in ("scale", "ratio"):
        low, high = getattr(args, name)
        if not np.all(np.isfinite((low, high))) or not 0 < low <= high:
            p.error(f"--{name} must contain two ordered positive values")
    if args.cutmix_minmax is not None:
        low, high = args.cutmix_minmax
        if not np.all(np.isfinite((low, high))) or not 0 <= low <= high <= 1:
            p.error("--cutmix-minmax must satisfy 0 <= min <= max <= 1")
    if (
        not np.all(np.isfinite(args.mean))
        or not np.all(np.isfinite(args.std))
        or np.any(np.asarray(args.std) <= 0)
    ):
        p.error("--mean and --std must be finite; std must be positive")
    if (
        not np.isfinite(args.opt_eps)
        or args.opt_eps <= 0
        or not np.all(np.isfinite(args.opt_betas))
        or not all(0 <= value < 1 for value in args.opt_betas)
    ):
        p.error("--opt-eps must be positive and --opt-betas must be between 0 and 1 (exclusive)")
    if args.warmup_epochs is not None and args.warmup_epochs < 0:
        p.error("--warmup-epochs must be non-negative")
    if args.min_lr is not None and (
        not np.isfinite(args.min_lr) or not 0 <= args.min_lr <= args.lr
    ):
        p.error("--min-lr must be finite and satisfy 0 <= min-lr <= lr")
    reserved = {"rngs", "num_classes", "drop_path_rate", "drop_rate", "global_pool"}.intersection(
        args.model_kwargs
    )
    if reserved:
        p.error(f"use dedicated CLI options for these model kwargs: {sorted(reserved)}")
    if (args.profile_step is None) != (args.profile_dir is None):
        p.error("--profile-step and --profile-dir must be used together")
    if args.profile_step is not None and args.profile_step < 0:
        p.error("--profile-step must be non-negative")
    dist_options = (args.dist_coordinator_address, args.dist_num_processes, args.dist_process_id)
    if any(value is not None for value in dist_options) and not all(
        value is not None for value in dist_options
    ):
        p.error("all three --dist-* options must be used together")
    if args.dist_num_processes is not None and (
        args.dist_num_processes <= 0 or not 0 <= args.dist_process_id < args.dist_num_processes
    ):
        p.error("distributed process count/id must satisfy 0 <= id < count")
    return args


def main(argv=None):
    args = _parse_args(argv)

    # 1. Initialize distributed cluster if needed
    init_distributed(args.dist_coordinator_address, args.dist_num_processes, args.dist_process_id)
    if args.attn_autotune:
        set_attention_autotuning(True)

    rank = jax.process_index()
    world_size = jax.process_count()
    local_devices = jax.local_devices()
    total_devices = jax.devices()
    global_batch_size = args.batch_size * world_size

    if rank == 0:
        print("=== JAX Distributed Training Setup (MaxText / MaxDiffusion Pipeline) ===")
        print(
            f"  Parallel mode:       {'FSDP (ZeRO-3 Sharded)' if args.fsdp else 'DDP (Replicated Weights)'}"
        )
        print(f"  Hosts (processes):   {world_size}")
        print(
            f"  Total devices:       {len(total_devices)} (devices: {[d.id for d in total_devices]})"
        )
        print(f"  Local devices/host:  {len(local_devices)}")
        print(f"  Process-local batch: {args.batch_size}")
        print(f"  Global batch size:   {global_batch_size}")
        print(f"  Architecture:        {args.model} (classes: {args.num_classes})")
        print(f"  AMP (bfloat16):      {args.amp}")
        print("=========================================================================")

    mixup = None
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0 or args.cutmix_minmax is not None:
        mixup = MixupCutmix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=1.0 if args.cutmix_minmax is not None else args.cutmix_alpha,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            cutmix_minmax=args.cutmix_minmax,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )

    # Loaders yield uint8 images; compiled steps normalize them and erase
    # training samples on device, like timm's prefetcher.
    preprocess = ImagePreprocess(
        mean=tuple(args.mean),
        std=tuple(args.std),
        re_prob=0.0 if args.no_aug else args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
    )

    # 2. Setup 1D Data-Parallel Mesh & SPMD NamedSharding
    mesh = jax.sharding.Mesh(total_devices, ("data",))
    P = jax.sharding.PartitionSpec
    data_sharding = jax.sharding.NamedSharding(mesh, P("data", None, None, None))
    train_label_sharding = jax.sharding.NamedSharding(
        mesh,
        P(
            "data",
        ),
    )
    eval_label_sharding = jax.sharding.NamedSharding(
        mesh,
        P(
            "data",
        ),
    )

    # 3. Instantiate model and data pipeline. Like timm, pass regularization and
    # pooling overrides only when requested; not every architecture accepts them.
    model_kwargs = dict(args.model_kwargs)
    if args.drop:
        model_kwargs["drop_rate"] = args.drop
    if args.drop_path:
        model_kwargs["drop_path_rate"] = args.drop_path
    if args.global_pool is not None:
        model_kwargs["global_pool"] = args.global_pool
    model = create_model(
        args.model,
        pretrained=args.initial_checkpoint or False,
        num_classes=args.num_classes,
        rngs=nnx.Rngs(args.seed),
        **model_kwargs,
    )
    model.train()

    if args.batch_size % len(local_devices) != 0:
        raise ValueError(
            f"batch_size {args.batch_size} must be divisible by local device count "
            f"{len(local_devices)} for SPMD data sharding (each device gets "
            f"batch_size / num_devices examples)"
        )

    train_loader = create_loader(
        os.path.join(args.data_dir, args.train_split),
        args.batch_size,
        img_size=args.img_size,
        is_training=True,
        auto_augment=args.auto_augment,
        no_aug=args.no_aug,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        interpolation=args.train_interpolation,
        crop_pct=args.crop_pct,
        mean=args.mean,
        std=args.std,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        num_workers=args.workers,
        seed=args.seed + rank,
        normalize=False,
    )
    steps_per_epoch = (
        args.steps_per_epoch if args.steps_per_epoch is not None else max(1, len(train_loader))
    )

    val_loader = None
    if os.path.isdir(os.path.join(args.data_dir, args.val_split)):
        if args.validation_batch_size % len(local_devices) != 0:
            raise ValueError("validation_batch_size must be divisible by local device count")
        val_loader = create_loader(
            os.path.join(args.data_dir, args.val_split),
            args.validation_batch_size,
            img_size=args.img_size,
            is_training=False,
            interpolation=args.interpolation,
            crop_pct=args.crop_pct,
            mean=args.mean,
            std=args.std,
            num_workers=args.workers,
            pad_remainder=True,
            normalize=False,
        )

    optimizer = make_optimizer(
        model,
        args.lr,
        args.weight_decay,
        args.epochs,
        steps_per_epoch,
        clip_grad=args.clip_grad,
        eps=args.opt_eps,
        betas=args.opt_betas,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
    )

    # Step-numbered checkpoint manager with retention; when validation runs,
    # additionally retain the epoch with the best val accuracy.
    output_dir = os.path.join(args.output, args.experiment or args.model)
    ckpt_manager = CheckpointManager(
        output_dir,
        max_to_keep=args.max_to_keep,
        best_fn=(lambda m: m["val_acc"]) if val_loader is not None else None,
        best_mode="max",
    )

    compiled_first_step = False
    profile_active = False

    try:
        start_epoch = 0
        if args.resume:
            if (
                isinstance(args.resume, str)
                and os.path.abspath(args.resume) != ckpt_manager.directory
            ):
                with CheckpointManager(args.resume) as source:
                    step, epoch = source.restore_latest(model, optimizer)
                if step is None:
                    raise ValueError(f"no checkpoints found in resume directory: {args.resume}")
            else:
                step, epoch = ckpt_manager.restore_latest(model, optimizer)
            if step is not None and isinstance(epoch, int):
                start_epoch = epoch + 1
                if rank == 0:
                    print(
                        f"  [Resume] Restored checkpoint at epoch {epoch}; "
                        f"resuming at epoch {start_epoch}"
                    )
            elif rank == 0:
                print("  [Resume] No checkpoint found; starting fresh")

        if rank == 0:
            with open(os.path.join(output_dir, "args.yaml"), "w") as file:
                yaml.safe_dump(
                    {
                        key: value
                        for key, value in vars(args).items()
                        if key not in ("config", "data")
                    },
                    file,
                    sort_keys=True,
                )

        global_step = start_epoch * steps_per_epoch
        train_loader.set_start_step(global_step)
        train_loader.start_prefetch()

        # Apply FSDP sharding if enabled (after any resume so restored arrays get
        # sharded too)
        if args.fsdp:
            fsdp_shard_model(model, mesh)
            fsdp_shard_model(optimizer, mesh)

        # Construct cached train & eval steps
        model.train()
        cached_train_step = make_cached_train_step(
            model, optimizer, amp=args.amp, mixup=mixup, preprocess=preprocess
        )
        train_rng = (
            jax.random.fold_in(jax.random.PRNGKey(args.seed), rank)
            if mixup is not None or preprocess.re_prob > 0
            else None
        )
        model.eval()
        cached_eval_step = (
            make_cached_eval_step(model, amp=args.amp, preprocess=preprocess)
            if val_loader is not None
            else None
        )
        model.train()

        # Async Host-to-Device prefetch pipeline (MaxText pattern)
        device_data_stream = prefetch_to_device(
            iter(train_loader),
            data_sharding,
            train_label_sharding,
            prefetch_size=args.prefetch,
        )

        for epoch in range(start_epoch, args.epochs):
            t_epoch_start = time.perf_counter()
            losses, accuracies = [], []
            t_steady_start = t_epoch_start if compiled_first_step else None
            steady_steps = 0

            for step in range(steps_per_epoch):
                # Start JAX profiler trace if requested
                if args.profile_dir and global_step == args.profile_step:
                    if rank == 0:
                        print(
                            f"  [Profiler] Starting JAX trace at global step {global_step} -> {args.profile_dir}"
                        )
                    jax.profiler.start_trace(args.profile_dir)
                    profile_active = True

                t_step_start = time.perf_counter()
                images, labels = next(device_data_stream)

                if train_rng is not None:
                    step_rng = jax.random.fold_in(train_rng, global_step)
                    loss, acc = cached_train_step(images, labels, args.smoothing, rng=step_rng)
                else:
                    loss, acc = cached_train_step(images, labels, args.smoothing)

                if not compiled_first_step:
                    # Block on first step to accurately measure XLA compilation time
                    loss.block_until_ready()
                    compile_time = time.perf_counter() - t_step_start
                    if rank == 0:
                        print(
                            f"  [XLA] Step 0 compiled and executed in {compile_time:.2f}s",
                            flush=True,
                        )
                    compiled_first_step = True
                    # Objects alive now (modules, compiled steps, data pipeline) last the
                    # whole run. Freezing them stops the cyclic GC, which allocations in
                    # every step dispatch trigger, from rescanning them: with Tokamax
                    # imported before the first compile, RepViT-M0.9 steps took 19.8 ms
                    # instead of 13.0 ms.
                    gc.collect()
                    gc.freeze()
                    t_steady_start = time.perf_counter()
                else:
                    steady_steps += 1

                losses.append(loss)
                accuracies.append(acc)
                del images, labels

                # Periodic step logging (MaxText style)
                if (
                    (step + 1) % args.log_interval == 0
                    and rank == 0
                    and t_steady_start is not None
                    and steady_steps > 0
                ):
                    # Materializing metrics waits for asynchronous device work
                    # before measuring completed training throughput.
                    step_loss = float(loss)
                    step_acc = float(acc)
                    elapsed_steady = time.perf_counter() - t_steady_start
                    step_time_ms = (elapsed_steady / steady_steps) * 1000.0
                    img_per_sec = (global_batch_size * steady_steps) / max(elapsed_steady, 1e-6)
                    print(
                        f"epoch {epoch:>3} [{step + 1:>4}/{steps_per_epoch}]: "
                        f"loss {step_loss:.4f} acc {step_acc:.4f} | "
                        f"{img_per_sec:7.1f} img/s ({step_time_ms:.1f}ms/step)",
                        flush=True,
                    )

                # Capture exactly the requested step, including its input transfer.
                if profile_active:
                    loss.block_until_ready()
                    profile_active = False
                    jax.profiler.stop_trace()
                    if rank == 0:
                        print(f"  [Profiler] JAX trace completed and written to {args.profile_dir}")

                global_step += 1

            loss_avg, acc_avg = _mean_metrics(losses, accuracies)

            epoch_time = time.perf_counter() - t_epoch_start
            epoch_img_per_sec = (global_batch_size * steps_per_epoch) / max(epoch_time, 1e-6)
            msg = (
                f"epoch {epoch:>3} summary: loss {loss_avg:.4f} acc {acc_avg:.4f} "
                f"| {epoch_img_per_sec:7.1f} img/s ({epoch_time:.2f}s)"
            )

            v_acc_avg = None
            if val_loader is not None and cached_eval_step is not None:
                # Every host must execute the SPMD validation step; only rank 0 reports it.
                # Reuse the async H2D prefetch pipeline so eval batches overlap
                # transfer with compute, like the training loop.
                v_losses, v_accuracies, v_counts = [], [], []
                val_stream = prefetch_to_device(
                    iter(val_loader),
                    data_sharding,
                    eval_label_sharding,
                    prefetch_size=args.prefetch,
                    mask_sharding=eval_label_sharding,
                )
                for v_images, v_labels, v_valid in val_stream:
                    val_loss, val_acc = cached_eval_step(v_images, v_labels, valid=v_valid)
                    v_losses.append(val_loss)
                    v_accuracies.append(val_acc)
                    v_counts.append(v_valid.sum())
                if not v_losses:
                    raise ValueError("validation loader produced no batches")
                v_loss_avg, v_acc_avg = _mean_metrics(v_losses, v_accuracies, v_counts)
                if rank == 0:
                    msg += f" | val loss {v_loss_avg:.4f} val acc {v_acc_avg:.4f}"

            if rank == 0:
                print(msg, flush=True)
            # Orbax synchronizes all hosts and writes their addressable shards.
            metrics = {"val_acc": v_acc_avg} if v_acc_avg is not None else None
            ckpt_manager.save(epoch, model, optimizer, metrics=metrics)

    finally:
        try:
            if profile_active:
                profile_active = False
                jax.profiler.stop_trace()
        finally:
            train_loader.close()
            if val_loader is not None:
                val_loader.close()
            ckpt_manager.close()


if __name__ == "__main__":
    main()
