"""Training entry: optax + nnx. Mirrors the spirit of timm's train.py (much smaller).

Example:
  python -m jimm.train --model resnet50 --data-dir /path/to/imagenet --epochs 90
"""
import argparse
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .data import create_loader
from .registry import create_model
from .checkpoint import save_checkpoint


def cross_entropy(logits, labels, smoothing=0.0):
    # smoothing=0 reduces to plain one-hot, so no branch needed (jit-safe)
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
    p.add_argument("--model", default="resnet50")
    p.add_argument("--data-dir", required=True, help="root with train/ and val/ splits")
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="cap train steps per epoch (default: full epoch)")
    p.add_argument("--output", default="./output")
    args = p.parse_args(argv)

    model = create_model(args.model, num_classes=args.num_classes,
                         drop_path_rate=args.drop_path, rngs=nnx.Rngs(0))
    model.train()
    train_loader = create_loader(f"{args.data_dir}/train", args.batch_size,
                                 img_size=args.img_size, is_training=True,
                                 num_workers=args.workers, seed=0)
    import os
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
            loss, acc = train_step(model, optimizer, batch["image"], batch["label"], args.smoothing)
            loss_sum, acc_sum = loss_sum + float(loss), acc_sum + float(acc)
        msg = f"epoch {epoch}: loss {loss_sum/steps_per_epoch:.4f} acc {acc_sum/steps_per_epoch:.4f} ({time.time()-t0:.0f}s)"
        if val_loader is not None:
            model.eval()
            v_loss, v_acc, n = 0.0, 0.0, 0
            for batch in val_loader:
                l, a = eval_step(model, batch["image"], batch["label"])
                v_loss, v_acc, n = v_loss + float(l), v_acc + float(a), n + 1
            model.train()
            msg += f" | val loss {v_loss/max(n,1):.4f} val acc {v_acc/max(n,1):.4f}"
        print(msg, flush=True)
        save_checkpoint(f"{args.output}/{args.model}/epoch_{epoch}", model, optimizer, epoch=epoch)


if __name__ == "__main__":
    main()
