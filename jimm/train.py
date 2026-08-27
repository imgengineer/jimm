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
  python -m jimm.train --model eva_large_patch16_224 --data-dir /path/to/imagenet --fsdp

  # Multi-node training (e.g. Node 0 of 2 nodes, 8 GPUs each):
  python -m jimm.train --model convnext_tiny --data-dir /path/to/imagenet \\
      --dist-coordinator-address 192.168.1.100:12345 --dist-num-processes 2 --dist-process-id 0

  # Profile execution with JAX profiler / Perfetto:
  python -m jimm.train --model resnet50 --data-dir /path/to/imagenet --profile-step 5 --profile-dir ./profiles
"""
import argparse
import collections
import functools
import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .checkpoint import CheckpointManager
from .data import MixupCutmix, create_loader
from .registry import create_model


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
            if isinstance(val, (jax.Array, np.ndarray)) and val.ndim >= 1 and val.shape[0] % num_devices == 0:
                spec = P(mesh_axis, *(None,) * (val.ndim - 1))
            elif isinstance(val, (jax.Array, np.ndarray)):
                spec = P()  # replicate if leading dimension is not evenly divisible
            else:
                continue
            sharding = jax.sharding.NamedSharding(mesh, spec)
            node.set_value(jax.device_put(val, sharding))


def prefetch_to_device(data_iter, data_sharding, label_sharding, prefetch_size=2):
    """Asynchronously prefetches and shards host data onto devices (double buffering).

    MaxText/MaxDiffusion pattern: overlaps host CPU data loading/decoding & Host-to-Device (H2D)
    transfer with on-device accelerator execution, preventing GPU/TPU idle starvation bubbles.
    """
    if isinstance(prefetch_size, bool) or not isinstance(prefetch_size, (int, np.integer)) or prefetch_size <= 0:
        raise ValueError("prefetch_size must be a positive integer")
    queue = collections.deque()

    def _put(batch):
        images = jax.make_array_from_process_local_data(data_sharding, batch["image"])
        labels = jax.make_array_from_process_local_data(label_sharding, batch["label"])
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


def cross_entropy(logits, labels, smoothing=0.0):
    # Mixup/CutMix supplies soft one-hot targets; ordinary batches use class ids.
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
    return optax.softmax_cross_entropy(logits, one_hot).mean()


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
        if (weights.shape != (len(losses),) or not np.all(np.isfinite(weights))
                or np.any(weights <= 0)):
            raise ValueError("counts must contain one positive finite value per batch")
        device_weights = jnp.asarray(weights)
        means = (metrics * device_weights).sum(axis=1) / device_weights.sum()
    values = np.asarray(jax.device_get(means))
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite epoch metrics")
    return float(values[0]), float(values[1])


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
    alpha = jnp.maximum(
        jnp.where(use_cutmix, config.cutmix_alpha, config.mixup_alpha), 1e-6)
    indices = (
        jnp.arange(batch - 1, -1, -1)
        if config.mode == "pair" else jax.random.permutation(perm_key, batch)
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
                (yy >= top[:, None, None]) & (yy < bottom[:, None, None])
                & (xx >= left[:, None, None]) & (xx < right[:, None, None])
            )
            actual_lam = 1.0 - (bottom - top) * (right - left) / (height * width)
            mixed_images = jnp.where(mask[..., None], images[indices], images)
            mixed_labels = actual_lam[:, None] * mixed_targets + (1.0 - actual_lam[:, None]) * mixed_targets[indices]
            return mixed_images, mixed_labels

        lam = jax.random.beta(alpha_key, alpha, alpha)
        if config.cutmix_minmax is None:
            ratio = jnp.sqrt(jnp.maximum(0.0, 1.0 - lam))
            box_h = jnp.rint(height * ratio).astype(jnp.int32)
            box_w = jnp.rint(width * ratio).astype(jnp.int32)
        else:
            low, high = config.cutmix_minmax
            box_h = jnp.rint(height * jax.random.uniform(box_keys[0], minval=low, maxval=high)).astype(jnp.int32)
            box_w = jnp.rint(width * jax.random.uniform(box_keys[1], minval=low, maxval=high)).astype(jnp.int32)
        center_y = jax.random.randint(box_keys[2], (), 0, height)
        center_x = jax.random.randint(box_keys[3], (), 0, width)
        top = jnp.maximum(0, center_y - box_h // 2)
        left = jnp.maximum(0, center_x - box_w // 2)
        bottom = jnp.minimum(height, center_y + box_h // 2)
        right = jnp.minimum(width, center_x + box_w // 2)
        yy = jnp.arange(height)[:, None]
        xx = jnp.arange(width)[None, :]
        mask = (
            (yy >= top) & (yy < bottom) & (xx >= left) & (xx < right)
        )
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


def make_optimizer(model, lr, weight_decay, epochs, steps_per_epoch, clip_grad=0.0,
                   warmup_ratio=0.1, min_lr_ratio=0.01):
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
        warmup_steps = min(
            total - 1, 5 * steps_per_epoch, 10000,
            max(int(total * warmup_ratio), 1))
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total, end_value=lr * min_lr_ratio)
    tx = optax.clip_by_global_norm(clip_grad) if clip_grad > 0 else optax.identity()
    decay_mask = lambda params: jax.tree.map(lambda p: p.ndim >= 2, params)  # noqa: E731
    adamw = optax.adamw(schedule, weight_decay=weight_decay, mask=decay_mask)
    return nnx.Optimizer(model, optax.chain(tx, adamw),
                         wrt=nnx.Param)


def _train_step(model, optimizer, images, labels, smoothing=0.0, amp=False,
                mixup=None, rng=None, with_metrics=False):
    _validate_batch(images, labels)
    if mixup is not None:
        if rng is None:
            raise ValueError("rng is required when mixup or cutmix is enabled")
        images, labels = _mixup_cutmix_jax(images, labels, rng, mixup)

    def loss_fn(model):
        x = images.astype(jnp.bfloat16) if amp else images
        logits = model(x)
        if amp:
            logits = logits.astype(jnp.float32)
        return cross_entropy(logits, labels, smoothing), logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    if with_metrics:
        grad_norm = optax.tree.norm(grads) if hasattr(optax, "tree") and hasattr(optax.tree, "norm") else optax.global_norm(grads)
    optimizer.update(model, grads)
    acc = _accuracy(logits, labels)
    return (StepMetrics(loss=loss, accuracy=acc, grad_norm=grad_norm)
            if with_metrics else (loss, acc))


@nnx.jit(static_argnames=("smoothing", "amp", "mixup"))
def train_step(model, optimizer, images, labels, smoothing=0.0, amp=False,
               mixup=None, rng=None):
    return _train_step(model, optimizer, images, labels, smoothing, amp, mixup, rng)


@nnx.jit(static_argnames=("smoothing", "amp", "mixup"))
def train_step_with_metrics(model, optimizer, images, labels, smoothing=0.0, amp=False,
                            mixup=None, rng=None):
    """Executes a training step, computing pre-clipping grad_norm for monitoring."""
    return _train_step(model, optimizer, images, labels, smoothing, amp, mixup, rng, True)


@nnx.jit(static_argnames=("amp",))
def eval_step(model, images, labels, amp=False):
    _validate_batch(images, labels)
    x = images.astype(jnp.bfloat16) if amp else images
    logits = model(x)
    if amp:
        logits = logits.astype(jnp.float32)
    return cross_entropy(logits, labels), _accuracy(logits, labels)


def make_cached_train_step(model, optimizer, amp=False, mixup=None):
    """Create one cached JIT train step with AMP and batch mixing bound."""
    return nnx.cached_partial(
        functools.partial(train_step, amp=amp, mixup=mixup), model, optimizer)


def make_cached_eval_step(model, amp=False):
    """Create one cached JIT eval step with AMP bound at construction time."""
    return nnx.cached_partial(functools.partial(eval_step, amp=amp), model)


def main(argv=None):
    p = argparse.ArgumentParser(prog="jimm.train")
    p.add_argument("--model", default="resnet50", help="model architecture name")
    p.add_argument("--data-dir", required=True, help="dataset root containing train/ and val/ directories")
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=128,
                   help="process-local batch size (each host processes this batch size)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4, help="data loader worker count per host")
    p.add_argument("--prefetch", type=int, default=2, help="batches to prefetch to device (double buffering)")
    p.add_argument("--clip-grad", type=float, default=1.0,
                   help="global-norm gradient clipping (0 = disabled)")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="cap train steps per epoch (default: full epoch)")
    p.add_argument("--log-interval", type=int, default=50, help="steps interval for logging throughput metrics")
    p.add_argument("--output", default="./output", help="output directory for checkpoints")
    p.add_argument("--max-to-keep", type=int, default=None,
                   help="retain only the N most recent epoch checkpoints (default: keep all)")
    p.add_argument("--resume", action="store_true", default=False,
                   help="resume training from the latest checkpoint under --output (starts fresh if none)")
    p.add_argument("--fsdp", action="store_true", default=False,
                   help="enable FSDP (ZeRO-3 style parameter and optimizer state sharding)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="enable AMP bfloat16 compute on Tensor Cores (default: true)")
    p.add_argument("--auto-augment", default=None,
                   help="timm policy: v0, original, rand-m9-n2, augmix-m3-w3-d-1, or trivialaugment")
    p.add_argument("--vflip", type=float, default=0.0)
    p.add_argument("--grayscale-prob", type=float, default=0.0)
    p.add_argument("--gaussian-blur-prob", type=float, default=0.0)
    p.add_argument("--mixup-alpha", type=float, default=0.0)
    p.add_argument("--cutmix-alpha", type=float, default=0.0)
    p.add_argument("--mixup-prob", type=float, default=1.0)
    p.add_argument("--mixup-mode", choices=("batch", "pair", "elem"), default="batch")

    # Profiler & Diagnostics
    p.add_argument("--profile-step", type=int, default=None, help="step index to trigger JAX profiler trace")
    p.add_argument("--profile-dir", type=str, default=None, help="directory to store JAX profile traces")

    # Multi-node / distributed options
    p.add_argument("--dist-coordinator-address", type=str, default=None,
                   help="IP:port of master coordinator for multi-node training (e.g. 192.168.1.1:12345)")
    p.add_argument("--dist-num-processes", type=int, default=None, help="total number of nodes/hosts")
    p.add_argument("--dist-process-id", type=int, default=None, help="rank/id of current node (0..num_processes-1)")
    args = p.parse_args(argv)

    for name in ("epochs", "batch_size", "img_size", "num_classes", "prefetch", "log_interval"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0:
        p.error("--workers must be non-negative")
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        p.error("--steps-per-epoch must be positive")
    if args.max_to_keep is not None and args.max_to_keep <= 0:
        p.error("--max-to-keep must be positive")
    for name in ("lr", "weight_decay", "clip_grad", "mixup_alpha", "cutmix_alpha"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            p.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    for name in ("smoothing", "vflip", "grayscale_prob", "gaussian_blur_prob", "mixup_prob"):
        value = getattr(args, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            p.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if not np.isfinite(args.drop_path) or not 0 <= args.drop_path < 1:
        p.error("--drop-path must be between 0 (inclusive) and 1 (exclusive)")
    if (args.profile_step is None) != (args.profile_dir is None):
        p.error("--profile-step and --profile-dir must be used together")
    if args.profile_step is not None and args.profile_step < 0:
        p.error("--profile-step must be non-negative")
    dist_options = (
        args.dist_coordinator_address, args.dist_num_processes, args.dist_process_id)
    if any(value is not None for value in dist_options) and not all(
            value is not None for value in dist_options):
        p.error("all three --dist-* options must be used together")
    if args.dist_num_processes is not None and (
            args.dist_num_processes <= 0
            or not 0 <= args.dist_process_id < args.dist_num_processes):
        p.error("distributed process count/id must satisfy 0 <= id < count")

    # 1. Initialize distributed cluster if needed
    init_distributed(args.dist_coordinator_address, args.dist_num_processes, args.dist_process_id)

    rank = jax.process_index()
    world_size = jax.process_count()
    local_devices = jax.local_devices()
    total_devices = jax.devices()
    global_batch_size = args.batch_size * world_size

    if rank == 0:
        print(f"=== JAX Distributed Training Setup (MaxText / MaxDiffusion Pipeline) ===")
        print(f"  Parallel mode:       {'FSDP (ZeRO-3 Sharded)' if args.fsdp else 'DDP (Replicated Weights)'}")
        print(f"  Hosts (processes):   {world_size}")
        print(f"  Total devices:       {len(total_devices)} (devices: {[d.id for d in total_devices]})")
        print(f"  Local devices/host:  {len(local_devices)}")
        print(f"  Process-local batch: {args.batch_size}")
        print(f"  Global batch size:   {global_batch_size}")
        print(f"  Architecture:        {args.model} (classes: {args.num_classes})")
        print(f"  AMP (bfloat16):      {args.amp}")
        print(f"=========================================================================")

    mixup = None
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        mixup = MixupCutmix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )

    # 2. Setup 1D Data-Parallel Mesh & SPMD NamedSharding
    mesh = jax.sharding.Mesh(total_devices, ('data',))
    P = jax.sharding.PartitionSpec
    data_sharding = jax.sharding.NamedSharding(mesh, P('data', None, None, None))
    train_label_sharding = jax.sharding.NamedSharding(mesh, P('data',))
    eval_label_sharding = jax.sharding.NamedSharding(mesh, P('data',))

    # 3. Instantiate model and data pipeline
    model = create_model(args.model, num_classes=args.num_classes,
                         drop_path_rate=args.drop_path, rngs=nnx.Rngs(0))
    model.train()

    if args.batch_size % len(local_devices) != 0:
        raise ValueError(
            f"batch_size {args.batch_size} must be divisible by local device count "
            f"{len(local_devices)} for SPMD data sharding (each device gets "
            f"batch_size / num_devices examples)")

    train_loader = create_loader(
        f"{args.data_dir}/train", args.batch_size,
        img_size=args.img_size, is_training=True,
        auto_augment=args.auto_augment,
        vflip=args.vflip,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        num_workers=args.workers, seed=rank,
    )
    train_loader.start_prefetch()
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else max(1, len(train_loader))

    val_loader = None
    if os.path.isdir(f"{args.data_dir}/val"):
        # SPMD evaluation requires batch sizes divisible by the device count and
        # an identical batch count on every host, so drop the tail batch when
        # running distributed; single-device keeps full val-set coverage.
        val_loader = create_loader(
            f"{args.data_dir}/val", args.batch_size,
            img_size=args.img_size, is_training=False,
            num_workers=args.workers,
            drop_remainder=len(total_devices) > 1)

    optimizer = make_optimizer(model, args.lr, args.weight_decay, args.epochs,
                               steps_per_epoch, clip_grad=args.clip_grad)

    # Step-numbered checkpoint manager with retention; when validation runs,
    # additionally retain the epoch with the best val accuracy.
    ckpt_manager = CheckpointManager(
        f"{args.output}/{args.model}",
        max_to_keep=args.max_to_keep,
        best_fn=(lambda m: m["val_acc"]) if val_loader is not None else None,
        best_mode="max",
    )

    start_epoch = 0
    if args.resume:
        step, epoch = ckpt_manager.restore_latest(model, optimizer)
        if step is not None:
            start_epoch = epoch + 1
            if rank == 0:
                print(f"  [Resume] Restored checkpoint at epoch {epoch}; "
                      f"resuming at epoch {start_epoch}")
        elif rank == 0:
            print("  [Resume] No checkpoint found; starting fresh")

    # Apply FSDP sharding if enabled (after any resume so restored arrays get
    # sharded too)
    if args.fsdp:
        fsdp_shard_model(model, mesh)
        fsdp_shard_model(optimizer, mesh)

    # Construct cached train & eval steps
    model.train()
    cached_train_step = make_cached_train_step(
        model, optimizer, amp=args.amp, mixup=mixup)
    train_rng = jax.random.fold_in(jax.random.PRNGKey(0), rank) if mixup is not None else None
    model.eval()
    cached_eval_step = make_cached_eval_step(model, amp=args.amp) if val_loader is not None else None
    model.train()

    # Async Host-to-Device prefetch pipeline (MaxText pattern)
    device_data_stream = prefetch_to_device(
        iter(train_loader), data_sharding, train_label_sharding,
        prefetch_size=args.prefetch,
    )

    global_step = start_epoch * steps_per_epoch
    compiled_first_step = False
    profile_active = False

    try:
        for epoch in range(start_epoch, args.epochs):
            t_epoch_start = time.time()
            losses, accuracies = [], []
            t_steady_start = None
            steady_steps = 0

            for step in range(steps_per_epoch):
                # Start JAX profiler trace if requested
                if args.profile_dir and global_step == args.profile_step:
                    if rank == 0:
                        print(f"  [Profiler] Starting JAX trace at global step {global_step} -> {args.profile_dir}")
                    jax.profiler.start_trace(args.profile_dir)
                    profile_active = True

                t_step_start = time.time()
                images, labels = next(device_data_stream)

                if train_rng is not None:
                    step_rng = jax.random.fold_in(train_rng, global_step)
                    loss, acc = cached_train_step(
                        images, labels, args.smoothing, rng=step_rng)
                else:
                    loss, acc = cached_train_step(images, labels, args.smoothing)

                if not compiled_first_step:
                    # Block on first step to accurately measure XLA compilation time
                    loss.block_until_ready()
                    compile_time = time.time() - t_step_start
                    if rank == 0:
                        print(f"  [XLA] Step 0 compiled and executed in {compile_time:.2f}s", flush=True)
                    compiled_first_step = True
                    t_steady_start = time.time()
                else:
                    steady_steps += 1

                losses.append(loss)
                accuracies.append(acc)
                del images, labels

                # Periodic step logging (MaxText style)
                if (step + 1) % args.log_interval == 0 and rank == 0 and t_steady_start is not None and steady_steps > 0:
                    elapsed_steady = time.time() - t_steady_start
                    step_time_ms = (elapsed_steady / steady_steps) * 1000.0
                    img_per_sec = (global_batch_size * steady_steps) / max(elapsed_steady, 1e-6)
                    step_loss = float(loss)
                    step_acc = float(acc)
                    print(
                        f"epoch {epoch:>3} [{step+1:>4}/{steps_per_epoch}]: "
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

            epoch_time = time.time() - t_epoch_start
            epoch_img_per_sec = (global_batch_size * steps_per_epoch) / max(epoch_time, 1e-6)
            msg = (
                f"epoch {epoch:>3} summary: loss {loss_avg:.4f} acc {acc_avg:.4f} "
                f"| {epoch_img_per_sec:7.1f} img/s ({epoch_time:.2f}s)"
            )

            if val_loader is not None and cached_eval_step is not None:
                # Every host must execute the SPMD validation step; only rank 0 reports it.
                # Reuse the async H2D prefetch pipeline so eval batches overlap
                # transfer with compute, like the training loop.
                v_losses, v_accuracies, v_counts = [], [], []
                val_stream = prefetch_to_device(
                    iter(val_loader), data_sharding, eval_label_sharding,
                    prefetch_size=args.prefetch)
                for v_images, v_labels in val_stream:
                    l, a = cached_eval_step(v_images, v_labels)
                    v_losses.append(l)
                    v_accuracies.append(a)
                    v_counts.append(v_labels.shape[0])
                if not v_losses:
                    raise ValueError("validation loader produced no batches")
                if rank == 0:
                    v_loss_avg, v_acc_avg = _mean_metrics(
                        v_losses, v_accuracies, v_counts)
                    msg += f" | val loss {v_loss_avg:.4f} val acc {v_acc_avg:.4f}"

            if rank == 0:
                print(msg, flush=True)
                # Checkpoint only from the primary host (rank 0), async; the
                # manager prunes old epochs per --max-to-keep when it finishes.
                metrics = (
                    {"val_acc": v_acc_avg}
                    if val_loader is not None and cached_eval_step is not None
                    else None)
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
