#!/usr/bin/env python3
"""Run inference over an image folder and write top-k predictions to CSV/JSON.

Mirrors timm's ``inference.py`` for jimm's ImageFolder pipeline.

Example:
    python inference.py --data-dir /data/imagenet --split validation \
        --model resnet50 --checkpoint ./output/resnet50 --topk 5 \
        --results-dir ./results --results-file resnet50-val
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
from flax import nnx

from jimm.scripts_common import (
    add_data_args,
    add_model_args,
    add_preprocess_args,
    add_runtime_args,
    build_eval_loader,
    check_device,
    configure_compilation_cache,
    create_eval_model,
    prefetch_batches,
    print_result,
    resolve_data_config,
    resolve_data_root,
    setup_logging,
    write_results_file,
)
from jimm.train import _forward_with_precision


@nnx.jit(static_argnames=("amp", "topk", "output_type"), graph=True, graph_updates=True)
def _predict(model, images, amp=False, topk=1, output_type="prob"):
    logits = _forward_with_precision(model, images, amp)
    scores = jax.nn.softmax(logits, axis=-1) if output_type == "prob" else logits
    return jax.lax.top_k(scores, topk)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run jimm image classification inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_args(parser)
    add_model_args(parser)
    add_preprocess_args(parser)
    add_runtime_args(parser, results=False)
    parser.add_argument("--topk", type=int, default=1, help="number of predictions per image")
    parser.add_argument(
        "--output-type", choices=("prob", "logit"), default="prob", help="score column values"
    )
    parser.add_argument("--output-col", default=None, help="score column name override")
    parser.add_argument("--results-file", default="", help="results filename (without extension)")
    parser.add_argument("--results-dir", default=None, help="directory for results files")
    parser.add_argument(
        "--results-format",
        nargs="+",
        choices=("csv", "json"),
        default=("csv",),
        help="one or more output formats",
    )
    parser.add_argument("--fullname", action="store_true", help="keep the path, not just basename")
    parser.add_argument(
        "--no-console-results", action="store_true", help="do not print per-image predictions"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logger = setup_logging("inference")
    check_device(args.device)
    configure_compilation_cache(args.compilation_cache)
    if args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("--batch-size must be positive and --workers non-negative")
    if args.topk <= 0:
        raise SystemExit("--topk must be positive")

    root = resolve_data_root(args)
    data_config = resolve_data_config(args)
    model = create_eval_model(args, data_config)
    topk = min(args.topk, data_config["num_classes"])

    loader = build_eval_loader(args, data_config, root)
    dataset = loader.dataset
    if dataset is None:
        raise RuntimeError("loader did not expose its image folder metadata")
    class_names = {index: name for name, index in dataset.class_to_idx.items()}
    batch_scores, batch_indices = [], []
    try:
        predict = nnx.cached_partial(_predict, model, graph=True, graph_updates=True)
        for batch_index, (images, _) in enumerate(prefetch_batches(loader, args.prefetch)):
            scores, indices = predict(images, amp=args.amp, topk=topk, output_type=args.output_type)
            batch_scores.append(scores)
            batch_indices.append(indices)
            if args.log_freq > 0 and (batch_index + 1) % args.log_freq == 0:
                logger.info("Predict: [%d/%d]", batch_index + 1, len(loader))
    finally:
        loader.close()

    if not batch_indices:
        raise RuntimeError("inference loader produced no batches")
    scores, indices = jax.device_get(
        (jnp.concatenate(batch_scores), jnp.concatenate(batch_indices))
    )
    if indices.shape[0] != len(dataset.samples):
        raise RuntimeError(
            f"loader produced {indices.shape[0]} predictions for {len(dataset.samples)} samples"
        )

    predictions = []
    for row in range(indices.shape[0]):
        path = dataset.samples[row][0]
        name = str(path) if args.fullname else path.name
        record = {"filename": name}
        for rank in range(topk):
            suffix = "" if topk == 1 else f"_{rank}"
            record[f"pred{suffix}"] = int(indices[row, rank])
            record[f"label{suffix}"] = class_names.get(int(indices[row, rank]), "<unknown>")
            record[f"{args.output_col or args.output_type}{suffix}"] = float(scores[row, rank])
        predictions.append(record)

    img_size = data_config["img_size"]
    results_file = args.results_file or f"{args.model}-{img_size}"
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        results_file = os.path.join(args.results_dir, results_file)
    if os.path.splitext(results_file)[1] in (".csv", ".json"):
        results_file = os.path.splitext(results_file)[0]
    for fmt in args.results_format:
        write_results_file(f"{results_file}.{fmt}", predictions, fmt)

    if not args.no_console_results:
        print(json.dumps(predictions, indent=4))

    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "img_size": img_size,
        "samples": len(predictions),
        "topk": topk,
        "results_file": results_file,
    }
    print_result(result)
    return result


if __name__ == "__main__":
    main()
