"""Distributed training entry: Optax + Flax NNX + Grain + Orbax.

Supports:
  - Single-device training (1 GPU / CPU)
  - Single-node multi-GPU data-parallel training (DDP with SPMD Data Mesh)
  - Multi-node multi-GPU distributed data-parallel training (Multi-host JAX + Grain Sharding)
  - FSDP (Fully Sharded Data Parallel / ZeRO-3 parameter and optimizer state sharding)

Examples:
  # Standard DDP on all available GPUs on the node:
  python -m jimm.train --model resnet50 --data-dir /path/to/imagenet --epochs 90

  # FSDP mode (ZeRO-3: shards weights and optimizer states across devices to save memory):
  python -m jimm.train --model eva_large_patch16_224 --data-dir /path/to/imagenet --fsdp

  # Multi-node training (e.g. Node 0 of 2 nodes, 8 GPUs each):
  python -m jimm.train --model convnext_tiny --data-dir /path/to/imagenet \\
      --dist-coordinator-address 192.168.1.100:12345 --dist-num-processes 2 --dist-process-id 0
"""
import argparse
import functools
import os
import time

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .checkpoint import save_checkpoint, wait_for_checkpoints
from .data import MixupCutmix, create_loader
from .registry import create_model


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


def cross_entropy(logits, labels, smoothing=0.0):
    # Mixup/CutMix supplies soft one-hot targets; ordinary batches use class ids.
    chex.assert_shape(logits, (None, None))
    chex.assert_rank(labels, {1, 2})
    chex.assert_equal(logits.shape[0], labels.shape[0])
    if labels.ndim == logits.ndim:
        chex.assert_shape(labels, (None, logits.shape[-1]))
    else:
        chex.assert_shape(labels, (None,))
    one_hot = labels if labels.ndim == logits.ndim else nnx.one_hot(labels, logits.shape[-1])
    one_hot = one_hot.astype(logits.dtype)
    if labels.ndim != logits.ndim:
        one_hot = one_hot * (1 - smoothing) + smoothing / logits.shape[-1]
    return optax.softmax_cross_entropy(logits, one_hot).mean()


def _accuracy(logits, labels):
    target = jnp.argmax(labels, axis=-1) if labels.ndim == logits.ndim else labels
    return jnp.mean(jnp.argmax(logits, -1) == target)


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
        return images, targets

    return jax.lax.cond(
        apply,
        lambda _: jax.lax.cond(use_cutmix, cutmix, mixup, None),
        identity,
        None,
    )


def make_optimizer(model, lr, weight_decay, epochs, steps_per_epoch, clip_grad=0.0):
    """AdamW (warmup + cosine decay) with timm-style weight-decay grouping.

    Following timm's default (`param_groups_weight_decay`), weight decay only
    applies to parameters with ndim >= 2 (conv/linear kernels); 1-D parameters
    (biases, norm scales) are exempt.
    """
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if lr < 0 or weight_decay < 0 or clip_grad < 0:
        raise ValueError("lr, weight_decay, and clip_grad must be non-negative")
    total = epochs * steps_per_epoch
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr,
        warmup_steps=min(5 * steps_per_epoch, 10000, max(total // 10, 1)),
        decay_steps=total, end_value=lr * 1e-2)
    tx = optax.clip_by_global_norm(clip_grad) if clip_grad > 0 else optax.identity()
    decay_mask = lambda params: jax.tree.map(lambda p: p.ndim >= 2, params)  # noqa: E731
    adamw = optax.adamw(schedule, weight_decay=weight_decay, mask=decay_mask)
    return nnx.Optimizer(model, optax.chain(tx, adamw),
                         wrt=nnx.Param)


@nnx.jit(static_argnames=("smoothing", "amp", "mixup"))
def train_step(model, optimizer, images, labels, smoothing=0.0, amp=False,
               mixup=None, rng=None):
    chex.assert_shape(images, (None, None, None, 3))
    chex.assert_rank(labels, {1, 2})
    chex.assert_equal(images.shape[0], labels.shape[0])
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
    optimizer.update(model, grads)
    acc = _accuracy(logits, labels)
    return loss, acc


@nnx.jit(static_argnames=("amp",))
def eval_step(model, images, labels, amp=False):
    chex.assert_shape(images, (None, None, None, 3))
    chex.assert_rank(labels, {1, 2})
    chex.assert_equal(images.shape[0], labels.shape[0])
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
    p.add_argument("--clip-grad", type=float, default=1.0,
                   help="global-norm gradient clipping (0 = disabled)")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="cap train steps per epoch (default: full epoch)")
    p.add_argument("--output", default="./output", help="output directory for checkpoints")
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
    
    # Multi-node / distributed options
    p.add_argument("--dist-coordinator-address", type=str, default=None,
                   help="IP:port of master coordinator for multi-node training (e.g. 192.168.1.1:12345)")
    p.add_argument("--dist-num-processes", type=int, default=None, help="total number of nodes/hosts")
    p.add_argument("--dist-process-id", type=int, default=None, help="rank/id of current node (0..num_processes-1)")
    args = p.parse_args(argv)

    # 1. Initialize distributed cluster if needed
    init_distributed(args.dist_coordinator_address, args.dist_num_processes, args.dist_process_id)

    rank = jax.process_index()
    world_size = jax.process_count()
    local_devices = jax.local_devices()
    total_devices = jax.devices()

    if rank == 0:
        print(f"=== JAX Distributed Training Setup ===")
        print(f"  Parallel mode:       {'FSDP (ZeRO-3 Sharded)' if args.fsdp else 'DDP (Replicated Weights)'}")
        print(f"  Hosts (processes):   {world_size}")
        print(f"  Total devices:       {len(total_devices)} (devices: {[d.id for d in total_devices]})")
        print(f"  Local devices/host:  {len(local_devices)}")
        print(f"  Process-local batch: {args.batch_size}")
        print(f"  Global batch size:   {args.batch_size * world_size}")
        print(f"  Architecture:        {args.model} (classes: {args.num_classes})")
        print(f"=======================================")

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

    # 3. Instantiate model and data pipeline (loader first: step count drives the LR schedule)
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
    it = iter(train_loader)
    steps_per_epoch = args.steps_per_epoch or max(1, len(train_loader))

    optimizer = make_optimizer(model, args.lr, args.weight_decay, args.epochs,
                               steps_per_epoch, clip_grad=args.clip_grad)

    # Apply FSDP sharding if enabled
    if args.fsdp:
        fsdp_shard_model(model, mesh)
        fsdp_shard_model(optimizer, mesh)

    val_loader = None
    if os.path.isdir(f"{args.data_dir}/val"):
        val_loader = create_loader(f"{args.data_dir}/val", args.batch_size,
                                   img_size=args.img_size, is_training=False,
                                   num_workers=args.workers)

    # Construct cached train & eval steps (freezes the respective mode graph traversal)
    model.train()
    cached_train_step = make_cached_train_step(
        model, optimizer, amp=args.amp, mixup=mixup)
    train_rng = jax.random.PRNGKey(rank) if mixup is not None else None
    model.eval()
    cached_eval_step = make_cached_eval_step(model, amp=args.amp) if val_loader is not None else None
    model.train()

    for epoch in range(args.epochs):
        t0 = time.time()
        losses, accuracies = [], []
        for _ in range(steps_per_epoch):
            batch = next(it)
            # Distribute process-local batch across devices before batch mixing.
            images = jax.make_array_from_process_local_data(data_sharding, batch["image"])
            labels = jax.make_array_from_process_local_data(train_label_sharding, batch["label"])
            if train_rng is not None:
                train_rng, step_rng = jax.random.split(train_rng)
                loss, acc = cached_train_step(
                    images, labels, args.smoothing, rng=step_rng)
            else:
                loss, acc = cached_train_step(images, labels, args.smoothing)
            losses.append(loss)
            accuracies.append(acc)
            del batch, images, labels

        try:
            loss_avg, acc_avg = (
                float(value)
                for value in jax.device_get((
                    jnp.mean(jnp.stack(losses)),
                    jnp.mean(jnp.stack(accuracies)),
                ))
            )
        except Exception:
            loss_avg, acc_avg = 0.0, 0.0
        msg = f"epoch {epoch:>3}: loss {loss_avg:.4f} acc {acc_avg:.4f} ({time.time()-t0:.1f}s)"
        if val_loader is not None and cached_eval_step is not None:
            # Every host must execute the SPMD validation step; only rank 0 reports it.
            v_losses, v_accuracies = [], []
            for batch in val_loader:
                v_images = jax.make_array_from_process_local_data(data_sharding, batch["image"])
                v_labels = jax.make_array_from_process_local_data(eval_label_sharding, batch["label"])
                l, a = cached_eval_step(v_images, v_labels)
                v_losses.append(l)
                v_accuracies.append(a)
                del batch, v_images, v_labels
            if rank == 0:
                try:
                    v_loss_avg, v_acc_avg = (
                        float(value)
                        for value in jax.device_get((
                            jnp.mean(jnp.stack(v_losses)),
                            jnp.mean(jnp.stack(v_accuracies)),
                        ))
                    )
                except Exception:
                    v_loss_avg, v_acc_avg = 0.0, 0.0
                msg += f" | val loss {v_loss_avg:.4f} val acc {v_acc_avg:.4f}"
        if rank == 0:
            print(msg, flush=True)
            # Checkpoint only from primary host (rank 0).
            save_checkpoint(
                f"{args.output}/{args.model}/epoch_{epoch}", model, optimizer,
                epoch=epoch, wait=False)

    train_loader.close()
    if val_loader is not None:
        val_loader.close()
    wait_for_checkpoints()


if __name__ == "__main__":
    main()
