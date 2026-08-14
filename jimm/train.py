"""Distributed training entry: Optax + Flax NNX + Grain + Orbax.

Supports:
  - Single-device training (1 GPU / CPU)
  - Single-node multi-GPU data-parallel training (SPMD Data Mesh)
  - Multi-node multi-GPU distributed data-parallel training (Multi-host JAX + Grain Sharding)

Examples:
  # Single-device / Single-node Multi-GPU (automatic device mesh):
  python -m jimm.train --model resnet50 --data-dir /path/to/imagenet --epochs 90

  # Multi-node training (e.g. Node 0 of 2 nodes, 8 GPUs each):
  python -m jimm.train --model convnext_tiny --data-dir /path/to/imagenet \\
      --dist-coordinator-address 192.168.1.100:12345 --dist-num-processes 2 --dist-process-id 0
"""
import argparse
import os
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .checkpoint import save_checkpoint
from .data import create_loader
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


def cross_entropy(logits, labels, smoothing=0.0):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    one_hot = one_hot * (1 - smoothing) + smoothing / logits.shape[-1]
    return optax.softmax_cross_entropy(logits, one_hot).mean()


def make_optimizer(model, lr, weight_decay, epochs, steps_per_epoch):
    total = epochs * steps_per_epoch
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr,
        warmup_steps=min(5 * steps_per_epoch, 10000, max(total // 10, 1)),
        decay_steps=total, end_value=lr * 1e-2)
    tx = optax.adamw(schedule, weight_decay=weight_decay)
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


@nnx.jit
def train_step(model, optimizer, images, labels, smoothing):
    def loss_fn(model):
        logits = model(images)
        return cross_entropy(logits, labels, smoothing), logits
    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, acc


@nnx.jit
def eval_step(model, images, labels):
    logits = model(images)
    return cross_entropy(logits, labels), jnp.mean(jnp.argmax(logits, -1) == labels)


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
    p.add_argument("--workers", type=int, default=8, help="data loader worker count per host")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="cap train steps per epoch (default: full epoch)")
    p.add_argument("--output", default="./output", help="output directory for checkpoints")
    
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
        print(f"  Hosts (processes):   {world_size}")
        print(f"  Total devices:       {len(total_devices)} (devices: {[d.id for d in total_devices]})")
        print(f"  Local devices/host:  {len(local_devices)}")
        print(f"  Process-local batch: {args.batch_size}")
        print(f"  Global batch size:   {args.batch_size * world_size}")
        print(f"  Architecture:        {args.model} (classes: {args.num_classes})")
        print(f"=======================================")

    # 2. Setup 1D Data-Parallel Mesh & SPMD NamedSharding
    mesh = jax.sharding.Mesh(total_devices, ('data',))
    P = jax.sharding.PartitionSpec
    data_sharding = jax.sharding.NamedSharding(mesh, P('data', None, None, None))
    label_sharding = jax.sharding.NamedSharding(mesh, P('data',))

    # 3. Instantiate model and optimizer
    model = create_model(args.model, num_classes=args.num_classes,
                         drop_path_rate=args.drop_path, rngs=nnx.Rngs(0))
    model.train()

    # 4. Data pipeline with automatic process sharding
    train_loader = create_loader(f"{args.data_dir}/train", args.batch_size,
                                 img_size=args.img_size, is_training=True,
                                 num_workers=args.workers, seed=rank)

    steps_per_epoch = args.steps_per_epoch or max(1, len(train_loader))
    optimizer = make_optimizer(model, args.lr, args.weight_decay, args.epochs, steps_per_epoch)

    val_loader = None
    if os.path.isdir(f"{args.data_dir}/val"):
        val_loader = create_loader(f"{args.data_dir}/val", args.batch_size,
                                   img_size=args.img_size, is_training=False,
                                   num_workers=args.workers)

    it = iter(train_loader)
    for epoch in range(args.epochs):
        t0, loss_sum, acc_sum = time.time(), 0.0, 0.0
        for _ in range(steps_per_epoch):
            batch = next(it)
            # Distribute process-local batch across devices using SPMD data sharding
            images = jax.make_array_from_process_local_data(data_sharding, batch["image"])
            labels = jax.make_array_from_process_local_data(label_sharding, batch["label"])
            
            loss, acc = train_step(model, optimizer, images, labels, args.smoothing)
            loss_sum += float(loss)
            acc_sum += float(acc)

        if rank == 0:
            msg = f"epoch {epoch:>3}: loss {loss_sum/steps_per_epoch:.4f} acc {acc_sum/steps_per_epoch:.4f} ({time.time()-t0:.1f}s)"
            if val_loader is not None:
                model.eval()
                v_loss, v_acc, n = 0.0, 0.0, 0
                for batch in val_loader:
                    v_images = jax.make_array_from_process_local_data(data_sharding, batch["image"])
                    v_labels = jax.make_array_from_process_local_data(label_sharding, batch["label"])
                    l, a = eval_step(model, v_images, v_labels)
                    v_loss += float(l)
                    v_acc += float(a)
                    n += 1
                model.train()
                msg += f" | val loss {v_loss/max(n, 1):.4f} val acc {v_acc/max(n, 1):.4f}"
            print(msg, flush=True)
            
            # Checkpoint only from primary host (rank 0)
            save_checkpoint(f"{args.output}/{args.model}/epoch_{epoch}", model, optimizer, epoch=epoch)


if __name__ == "__main__":
    main()
