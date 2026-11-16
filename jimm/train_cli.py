"""Command-line parsing for :mod:`jimm.train`."""

import argparse
import ast
import contextlib
import os

import numpy as np
import yaml

from .data import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["ParseKwargs", "parse_args"]


class ParseKwargs(argparse.Action):
    """Parse ``KEY=VALUE`` model constructor arguments."""

    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = dict(getattr(namespace, self.dest))
        for item in values:
            key, separator, value = item.partition("=")
            if not separator or not key:
                raise argparse.ArgumentError(self, "model kwargs must use KEY=VALUE")
            with contextlib.suppress(ValueError, SyntaxError):
                value = ast.literal_eval(value)
            kwargs[key] = value
        setattr(namespace, self.dest, kwargs)


def parse_args(argv=None):
    """Parse training options, applying a ``-c/--config`` YAML file first."""
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
        "--model-kwargs", nargs="*", default={}, action=ParseKwargs, metavar="KEY=VALUE"
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
        "--compilation-cache",
        default="",
        metavar="DIR",
        help="JAX persistent compilation cache directory (shared by all hosts)",
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
