#!/usr/bin/env python3
"""Strip optimizer state from a jimm checkpoint and write a model-only checkpoint.

The jimm equivalent of timm's ``clean_checkpoint.py``: the result is much
smaller and can still be restored with ``jimm.checkpoint.load_checkpoint``.

Example:
    python clean_checkpoint.py --checkpoint ./output/resnet50/10 \
        --model resnet50 --output ./output/resnet50-clean
"""

import argparse
import os

from flax import nnx

from jimm.checkpoint import load_checkpoint, save_checkpoint
from jimm.registry import create_model
from jimm.scripts_common import ParseKwargs, print_result, setup_logging


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Remove optimizer state from a jimm checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory to clean")
    parser.add_argument("--model", required=True, help="model architecture name")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)
    parser.add_argument("--output", required=True, help="output checkpoint directory")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logger = setup_logging("clean_checkpoint")
    if not os.path.isdir(args.checkpoint):
        raise SystemExit(f"checkpoint directory does not exist: {args.checkpoint}")
    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"output already exists: {args.output} (use --force to overwrite)")

    model = create_model(
        args.model, num_classes=args.num_classes, rngs=nnx.Rngs(0), **args.model_kwargs
    )
    epoch = load_checkpoint(args.checkpoint, model)
    save_checkpoint(args.output, model, epoch=epoch)

    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "output": args.output,
    }
    logger.info("Saved model-only checkpoint to %s (epoch %d)", args.output, epoch)
    print_result(result)
    return result


if __name__ == "__main__":
    main()
