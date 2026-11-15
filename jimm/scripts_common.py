"""Shared helpers for the timm-style top-level train/validate/inference/benchmark scripts."""

import argparse
import ast
import collections
import csv
import json
import logging
import os
import time

import jax
import jax.numpy as jnp
from flax import nnx

from .checkpoint import load_checkpoint
from .compilation import configure_compilation_cache
from .data import IMAGENET_MEAN, IMAGENET_STD, create_loader
from .loss import _cross_entropy_losses
from .registry import create_model, get_default_cfg
from .train import _forward_with_precision
from .weights import load_pretrained

__all__ = [
    "ParseKwargs",
    "add_data_args",
    "add_model_args",
    "add_preprocess_args",
    "add_runtime_args",
    "build_eval_loader",
    "check_device",
    "configure_compilation_cache",
    "create_eval_model",
    "evaluate",
    "model_param_count",
    "prefetch_batches",
    "print_result",
    "resolve_data_config",
    "resolve_data_root",
    "setup_logging",
    "write_results_file",
]


class ParseKwargs(argparse.Action):
    """Parse ``KEY=VALUE`` model constructor arguments."""

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


def check_device(device):
    """Validate a timm-style ``--device`` value against the active JAX backend."""
    if not device or device == "auto":
        return
    backend = jax.default_backend()
    want = "cpu" if "cpu" in device else "gpu"
    if backend != want:
        raise SystemExit(
            f"--device {device} requested, but JAX is running on {backend}; "
            "set JAX_PLATFORMS=cpu or JAX_PLATFORMS=cuda before launching"
        )


def setup_logging(name, level=logging.INFO):
    logging.basicConfig(format="%(levelname)s - %(message)s", level=level)
    return logging.getLogger(name)


def add_data_args(parser, split="validation", workers=4, batch_size=256):
    parser.add_argument("data", nargs="?", metavar="DIR", help="dataset root")
    parser.add_argument("--data-dir", metavar="DIR", help="dataset root (takes precedence)")
    parser.add_argument("--split", default=split, help="dataset split subdirectory")
    parser.add_argument("-j", "--workers", type=int, default=workers, help="data loader workers")
    parser.add_argument("-b", "--batch-size", type=int, default=batch_size, help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="random seed")


def add_model_args(parser, default_model="resnet50"):
    parser.add_argument("--model", "-m", default=default_model, help="model architecture name")
    parser.add_argument(
        "--pretrained", action="store_true", help="use registered pretrained weights"
    )
    parser.add_argument(
        "--checkpoint", default="", help="jimm Orbax checkpoint directory or local .npz file"
    )
    parser.add_argument("--num-classes", type=int, default=None, help="classifier width")
    parser.add_argument("--gp", "--global-pool", dest="global_pool", default=None)
    parser.add_argument(
        "--model-kwargs", nargs="*", default={}, action=ParseKwargs, metavar="KEY=VALUE"
    )


def add_preprocess_args(parser):
    parser.add_argument("--img-size", type=int, default=None, help="square input size")
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=3,
        metavar=("C", "H", "W"),
        help="input channels, height, width",
    )
    parser.add_argument(
        "--in-chans", type=int, default=None, help="input channels (only 3 supported)"
    )
    parser.add_argument(
        "--crop-pct", type=float, default=None, help="evaluation center-crop percent"
    )
    parser.add_argument("--mean", type=float, nargs=3, default=None, help="normalization mean")
    parser.add_argument("--std", type=float, nargs=3, default=None, help="normalization std")
    parser.add_argument("--interpolation", default=None, help="resize interpolation")


def add_runtime_args(parser, log_freq=10, results=True):
    parser.add_argument(
        "--device",
        default="",
        help="cpu/gpu hint; JAX selects the backend, override with JAX_PLATFORMS",
    )
    parser.add_argument("--amp", action="store_true", help="bfloat16 compute")
    parser.add_argument(
        "--prefetch", type=int, default=2, help="batches to queue on device ahead of compute"
    )
    parser.add_argument(
        "--compilation-cache",
        default="",
        metavar="DIR",
        help="JAX persistent compilation cache directory",
    )
    parser.add_argument("--log-freq", type=int, default=log_freq, help="batch logging interval")
    if results:
        parser.add_argument("--results-file", default="", help="results output path")
        parser.add_argument("--results-format", default="csv", choices=("csv", "json"))


def resolve_data_root(args):
    root = args.data_dir or args.data
    if not root:
        raise SystemExit("a dataset root is required: --data-dir DIR or positional DIR")
    split_root = os.path.join(root, args.split)
    if os.path.isdir(split_root):
        return split_root
    if args.split == "validation" and os.path.isdir(os.path.join(root, "val")):
        return os.path.join(root, "val")
    if not os.path.isdir(root):
        raise SystemExit(f"dataset directory does not exist: {root}")
    return root


def resolve_data_config(args):
    cfg = get_default_cfg(args.model)
    if args.input_size is not None:
        channels, height, width = args.input_size
        if channels != 3 or height != width or height <= 0:
            raise SystemExit("--input-size must specify square RGB input: 3 H H")
        img_size = height
        in_chans = 3
    else:
        in_chans = args.in_chans or cfg.get("input_size", (3, 224, 224))[0]
        if in_chans != 3:
            raise SystemExit("jimm data loaders only support RGB inputs (--in-chans 3)")
        img_size = args.img_size or cfg.get("input_size", (3, 224, 224))[-1]
    mean = tuple(args.mean) if args.mean is not None else tuple(cfg.get("mean", IMAGENET_MEAN))
    std = tuple(args.std) if args.std is not None else tuple(cfg.get("std", IMAGENET_STD))
    num_classes = args.num_classes
    if num_classes is None:
        num_classes = int(cfg.get("num_classes") or 1000)
    return dict(
        img_size=int(img_size),
        in_chans=in_chans,
        crop_pct=float(args.crop_pct if args.crop_pct is not None else cfg.get("crop_pct", 0.875)),
        interpolation=args.interpolation or cfg.get("interpolation", "bilinear"),
        mean=mean,
        std=std,
        num_classes=num_classes,
    )


def create_eval_model(args, data_config):
    if args.pretrained:
        raise SystemExit(
            "no pretrained weights are registered in jimm; pass --checkpoint PATH instead"
        )
    model_kwargs = dict(args.model_kwargs)
    if args.global_pool is not None:
        model_kwargs["global_pool"] = args.global_pool
    model = create_model(
        args.model,
        num_classes=data_config["num_classes"],
        rngs=nnx.Rngs(args.seed),
        **model_kwargs,
    )
    if args.checkpoint:
        if os.path.isdir(args.checkpoint):
            load_checkpoint(args.checkpoint, model)
        else:
            load_pretrained(model, args.checkpoint)
    model.eval()
    return model


def build_eval_loader(args, data_config, root=None):
    return create_loader(
        root or resolve_data_root(args),
        args.batch_size,
        img_size=data_config["img_size"],
        is_training=False,
        crop_pct=data_config["crop_pct"],
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        seed=args.seed,
        normalize=True,
    )


def model_param_count(model):
    return sum(leaf.size for leaf in jax.tree.leaves(nnx.state(model, nnx.Param)))


@nnx.jit(static_argnames=("amp",))
def _eval_step(model, images, labels, amp=False):
    logits = _forward_with_precision(model, images, amp)
    losses = _cross_entropy_losses(logits, labels)
    topk = min(5, logits.shape[-1])
    _, predictions = jax.lax.top_k(logits, topk)
    correct = jnp.cumsum(predictions == labels[:, None], axis=1)
    top5 = correct[:, topk - 1].sum() if topk >= 5 else correct[:, 0].sum()
    return losses.sum(), correct[:, 0].sum(), top5, images.shape[0]


def prefetch_batches(loader, prefetch_size=2):
    """Queue batches on device ahead of compute so H2D transfers overlap evaluation."""
    iterator = iter(loader)
    if prefetch_size <= 0:
        for batch in iterator:
            yield jax.device_put(batch["image"]), jax.device_put(batch["label"])
        return
    queue = collections.deque()

    def _put(batch):
        return jax.device_put(batch["image"]), jax.device_put(batch["label"])

    for _ in range(prefetch_size):
        try:
            queue.append(_put(next(iterator)))
        except StopIteration:
            break
    while queue:
        item = queue.popleft()
        try:
            queue.append(_put(next(iterator)))
        except StopIteration:
            pass
        yield item


def evaluate(model, loader, amp=False, log_freq=10, logger=None, prefetch=2):
    """Run top-1/top-5 evaluation; device metrics sync only for logs and the return."""
    total_loss = jnp.zeros((), jnp.float32)
    top1 = jnp.zeros((), jnp.int32)
    top5 = jnp.zeros((), jnp.int32)
    samples = 0
    start = time.perf_counter()
    for batch_index, (images, labels) in enumerate(prefetch_batches(loader, prefetch)):
        loss_sum, correct1, correct5, count = _eval_step(model, images, labels, amp)
        total_loss = total_loss + loss_sum
        top1 = top1 + correct1
        top5 = top5 + correct5
        samples += int(count)
        if logger is not None and log_freq > 0 and (batch_index + 1) % log_freq == 0:
            done_loss, done_top1 = jax.device_get((total_loss, top1))
            elapsed = max(time.perf_counter() - start, 1e-6)
            logger.info(
                "Eval: [%d/%d] loss %.4f acc %.4f %.1f samples/s",
                batch_index + 1,
                len(loader),
                float(done_loss) / samples,
                float(done_top1) / samples,
                samples / elapsed,
            )
    elapsed = max(time.perf_counter() - start, 1e-6)
    if samples == 0:
        raise ValueError("evaluation loader produced no samples")
    loss, correct1, correct5 = jax.device_get((total_loss, top1, top5))
    return dict(
        loss=float(loss) / samples,
        top1=float(correct1) / samples,
        top5=float(correct5) / samples,
        samples=samples,
        seconds=elapsed,
        samples_per_sec=samples / elapsed,
    )


def write_results_file(path, rows, fmt="csv"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fmt == "json":
        with open(path, "w") as file:
            json.dump(rows, file, indent=4)
        return
    rows = rows if isinstance(rows, (list, tuple)) else [rows]
    if not rows:
        return
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_result(result):
    """Print a JSON summary after timm's ``--result`` delimiter for bulk_runner."""
    print("--result")
    print(json.dumps(result, indent=4))
