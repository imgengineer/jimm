#!/usr/bin/env python3
"""Average model weights across checkpoints from a jimm training run.

The jimm equivalent of timm's ``avg_checkpoints.py``. Checkpoints must come
from the same architecture (same ``--model`` and ``--num-classes``).

Example:
    python avg_checkpoints.py --input ./output/resnet50 --model resnet50 \
        --output ./output/resnet50-avg -n 5
"""

import argparse
import os

import jax
import jax.numpy as jnp
from flax import nnx

from jimm.checkpoint import CheckpointManager, save_checkpoint
from jimm.registry import create_model
from jimm.scripts_common import ParseKwargs, print_result, setup_logging


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Average jimm checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="jimm CheckpointManager directory")
    parser.add_argument("--model", required=True, help="model architecture name")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)
    parser.add_argument("--output", required=True, help="output checkpoint directory")
    parser.add_argument("-n", type=int, default=10, help="number of checkpoints to average")
    parser.add_argument(
        "--metric-key", default="val_acc", help="checkpoint metric used to rank checkpoints"
    )
    parser.add_argument("--no-sort", action="store_true", help="average the latest checkpoints")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logger = setup_logging("avg_checkpoints")
    if args.n <= 0:
        raise SystemExit("-n must be positive")
    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"output already exists: {args.output} (use --force to overwrite)")

    manager = CheckpointManager(args.input)
    try:
        steps = manager.all_steps()
        if not steps:
            raise SystemExit(f"no checkpoints found in {args.input}")
        if args.no_sort:
            selected = sorted(steps)[-args.n :]
        else:
            ranked = []
            for step in steps:
                metrics = manager.metrics(step) or {}
                if args.metric_key in metrics:
                    ranked.append((float(metrics[args.metric_key]), step))
            if ranked:
                selected = [step for _, step in sorted(ranked)[-args.n :]]
            else:
                logger.warning(
                    "no %r metric in checkpoints; averaging the latest %d steps",
                    args.metric_key,
                    args.n,
                )
                selected = sorted(steps)[-args.n :]

        model = create_model(
            args.model, num_classes=args.num_classes, rngs=nnx.Rngs(0), **args.model_kwargs
        )
        states = []
        epochs = []
        for step in selected:
            epochs.append(manager.restore(step, model))
            states.append(nnx.to_pure_dict(nnx.state(model, (nnx.Param, nnx.BatchStat))))
        averaged = jax.tree.map(
            lambda *leaves: jnp.mean(jnp.stack([jnp.asarray(x) for x in leaves]), axis=0),
            *states,
        )
        nnx.update(model, averaged)
        save_checkpoint(args.output, model, epoch=max(epochs))
    finally:
        manager.close()

    result = {
        "model": args.model,
        "input": args.input,
        "steps": selected,
        "count": len(selected),
        "output": args.output,
    }
    logger.info("Averaged %d checkpoints into %s", len(selected), args.output)
    print_result(result)
    return result


if __name__ == "__main__":
    main()
