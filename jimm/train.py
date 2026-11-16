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

Structure: argument parsing lives in :mod:`jimm.train_cli`, device-side batch
preprocessing in :mod:`jimm.preprocess`, and the host-to-device prefetch loop in
:mod:`jimm.prefetch`; this module owns the distributed setup and the compiled
train/eval step functions.
"""

import functools
import gc
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
from .compilation import configure_compilation_cache
from .data import MixupCutmix, create_loader
from .loss import _cross_entropy_losses, cross_entropy
from .models.nfnet import ScaledStdConv
from .optim import create_optimizer, make_optimizer
from .prefetch import prefetch_stream
from .preprocess import ImagePreprocess, mixup_cutmix, preprocess_images
from .registry import create_model
from .train_cli import parse_args

__all__ = [
    "ImagePreprocess",
    "StepMetrics",
    "create_optimizer",
    "cross_entropy",
    "eval_step",
    "fsdp_shard_model",
    "init_distributed",
    "main",
    "make_cached_eval_step",
    "make_cached_train_step",
    "make_optimizer",
    "prefetch_to_device",
    "train_step",
    "train_step_with_metrics",
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
    for _path, node in nnx.graph.iter_graph(model_or_opt):
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

    def _put(batch):
        images = jax.make_array_from_process_local_data(data_sharding, batch["image"])
        labels = jax.make_array_from_process_local_data(label_sharding, batch["label"])
        if mask_sharding is not None:
            valid = jax.make_array_from_process_local_data(mask_sharding, batch["valid"])
            return images, labels, valid
        return images, labels

    yield from prefetch_stream(data_iter, _put, prefetch_size)


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
        return float(values[0]), float(values[1])
    except (TypeError, ValueError) as exc:
        raise FloatingPointError("non-finite epoch metrics") from exc


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
    images = preprocess_images(images, preprocess, rng, training=True)
    if mixup is not None:
        if rng is None:
            raise ValueError("rng is required when mixup or cutmix is enabled")
        images, labels = mixup_cutmix(images, labels, rng, mixup)

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
    return StepMetrics(loss=loss, accuracy=acc, grad_norm=optax.tree.norm(grads))


@nnx.jit(static_argnames=("amp", "preprocess"))
def eval_step(model, images, labels, amp=False, valid=None, preprocess=None):
    """Return mean loss and accuracy; ``preprocess`` normalizes uint8 images only."""
    _validate_batch(images, labels)
    if valid is not None and (valid.ndim != 1 or valid.shape != (images.shape[0],)):
        raise ValueError("valid mask must be 1-D and match the image batch size")
    images = preprocess_images(images, preprocess, None, training=False)
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


def _init_runtime(args):
    """Configure compilation caching and the cluster; return the device facts."""
    configure_compilation_cache(args.compilation_cache)
    init_distributed(args.dist_coordinator_address, args.dist_num_processes, args.dist_process_id)
    if args.attn_autotune:
        set_attention_autotuning(True)
    return jax.process_index(), jax.process_count(), jax.local_devices(), jax.devices()


def _print_setup(args, rank, world_size, local_devices, total_devices, global_batch_size):
    """Print the MaxText-style run banner on the coordinator host."""
    if rank != 0:
        return
    print("=== JAX Distributed Training Setup (MaxText / MaxDiffusion Pipeline) ===")
    print(
        f"  Parallel mode:       {'FSDP (ZeRO-3 Sharded)' if args.fsdp else 'DDP (Replicated Weights)'}"
    )
    print(f"  Hosts (processes):   {world_size}")
    print(f"  Total devices:       {len(total_devices)} (devices: {[d.id for d in total_devices]})")
    print(f"  Local devices/host:  {len(local_devices)}")
    print(f"  Process-local batch: {args.batch_size}")
    print(f"  Global batch size:   {global_batch_size}")
    print(f"  Architecture:        {args.model} (classes: {args.num_classes})")
    print(f"  AMP (bfloat16):      {args.amp}")
    print("=========================================================================")


def _create_mixup(args):
    """Return the batch Mixup/CutMix config, or None when mixing is disabled."""
    if args.mixup_alpha <= 0 and args.cutmix_alpha <= 0 and args.cutmix_minmax is None:
        return None
    return MixupCutmix(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=1.0 if args.cutmix_minmax is not None else args.cutmix_alpha,
        prob=args.mixup_prob,
        switch_prob=args.mixup_switch_prob,
        mode=args.mixup_mode,
        cutmix_minmax=args.cutmix_minmax,
        label_smoothing=args.smoothing,
        num_classes=args.num_classes,
    )


def _create_preprocess(args):
    """Device-side preprocessing matching the ``normalize=False`` loaders."""
    return ImagePreprocess(
        mean=tuple(args.mean),
        std=tuple(args.std),
        re_prob=0.0 if args.no_aug else args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
    )


def _create_shardings(total_devices):
    """Build the 1-D data-parallel mesh and the SPMD shardings used by the run."""
    mesh = jax.sharding.Mesh(total_devices, ("data",))
    P = jax.sharding.PartitionSpec
    data_sharding = jax.sharding.NamedSharding(mesh, P("data", None, None, None))
    train_label_sharding = jax.sharding.NamedSharding(mesh, P("data"))
    eval_label_sharding = jax.sharding.NamedSharding(mesh, P("data"))
    return mesh, data_sharding, train_label_sharding, eval_label_sharding


def _create_model(args):
    """Instantiate the model, passing regularization/pooling overrides only if requested.

    Like timm, the optional overrides are omitted rather than set to defaults because
    not every architecture accepts them.
    """
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
    return model


def _create_train_loader(args, rank, local_devices):
    """Build the training loader and return it with the number of steps per epoch."""
    if args.batch_size % len(local_devices) != 0:
        raise ValueError(
            f"batch_size {args.batch_size} must be divisible by local device count "
            f"{len(local_devices)} for SPMD data sharding (each device gets "
            f"batch_size / num_devices examples)"
        )
    loader = create_loader(
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
        args.steps_per_epoch if args.steps_per_epoch is not None else max(1, len(loader))
    )
    return loader, steps_per_epoch


def _create_val_loader(args, local_devices):
    """Build the validation loader, or return None when the split is missing."""
    if not os.path.isdir(os.path.join(args.data_dir, args.val_split)):
        return None
    if args.validation_batch_size % len(local_devices) != 0:
        raise ValueError("validation_batch_size must be divisible by local device count")
    # pad_remainder keeps the SPMD batch shape static on the last partial batch.
    return create_loader(
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


def _restore_run(args, model, optimizer, ckpt_manager, rank):
    """Restore weights for ``--resume`` and return the first epoch to run."""
    if not args.resume:
        return 0
    if isinstance(args.resume, str) and os.path.abspath(args.resume) != ckpt_manager.directory:
        with CheckpointManager(args.resume) as source:
            step, epoch = source.restore_latest(model, optimizer)
        if step is None:
            raise ValueError(f"no checkpoints found in resume directory: {args.resume}")
    else:
        step, epoch = ckpt_manager.restore_latest(model, optimizer)
    if step is not None and isinstance(epoch, int):
        if rank == 0:
            print(f"  [Resume] Restored checkpoint at epoch {epoch}; resuming at epoch {epoch + 1}")
        return epoch + 1
    if rank == 0:
        print("  [Resume] No checkpoint found; starting fresh")
    return 0


def _write_run_config(args, output_dir):
    """Persist the resolved arguments next to the checkpoints."""
    with open(os.path.join(output_dir, "args.yaml"), "w") as file:
        yaml.safe_dump(
            {key: value for key, value in vars(args).items() if key not in ("config", "data")},
            file,
            sort_keys=True,
        )


def _validate(args, val_loader, cached_eval_step, data_sharding, eval_label_sharding):
    """Run one validation pass and return its finite mean loss and accuracy.

    Every host must execute the SPMD validation step; the caller reports it on rank 0.
    Validation batches reuse the async host-to-device prefetch pipeline so transfer
    overlaps compute, like the training loop.
    """
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
    return _mean_metrics(v_losses, v_accuracies, v_counts)


def main(argv=None):
    """Build the run, then train (and validate) for ``--epochs`` epochs."""
    args = parse_args(argv)

    rank, world_size, local_devices, total_devices = _init_runtime(args)
    global_batch_size = args.batch_size * world_size
    _print_setup(args, rank, world_size, local_devices, total_devices, global_batch_size)

    mixup = _create_mixup(args)
    # Loaders yield uint8 images; compiled steps normalize them and erase training
    # samples on device, like timm's prefetcher.
    preprocess = _create_preprocess(args)
    mesh, data_sharding, train_label_sharding, eval_label_sharding = _create_shardings(
        total_devices
    )

    model = _create_model(args)
    train_loader, steps_per_epoch = _create_train_loader(args, rank, local_devices)
    val_loader = _create_val_loader(args, local_devices)
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
        start_epoch = _restore_run(args, model, optimizer, ckpt_manager, rank)
        if rank == 0:
            _write_run_config(args, output_dir)

        global_step = start_epoch * steps_per_epoch
        train_loader.set_start_step(global_step)
        train_loader.start_prefetch()

        # Apply FSDP sharding after any resume so restored arrays get sharded too.
        if args.fsdp:
            fsdp_shard_model(model, mesh)
            fsdp_shard_model(optimizer, mesh)

        # One cached JIT train step and (optionally) one cached JIT eval step.
        model.train()
        cached_train_step = make_cached_train_step(
            model, optimizer, amp=args.amp, mixup=mixup, preprocess=preprocess
        )
        train_rng = (
            jax.random.fold_in(jax.random.key(args.seed), rank)
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

        # Async host-to-device prefetch pipeline (MaxText pattern).
        device_data_stream = prefetch_to_device(
            iter(train_loader), data_sharding, train_label_sharding, prefetch_size=args.prefetch
        )

        for epoch in range(start_epoch, args.epochs):
            t_epoch_start = time.perf_counter()
            losses, accuracies = [], []
            t_steady_start = t_epoch_start if compiled_first_step else None
            steady_steps = 0

            for step in range(steps_per_epoch):
                # Start the JAX profiler trace at the requested global step.
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
                    # Block on the first step to measure XLA compilation time accurately.
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

                # Periodic step logging (MaxText style).
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
                v_loss_avg, v_acc_avg = _validate(
                    args, val_loader, cached_eval_step, data_sharding, eval_label_sharding
                )
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
