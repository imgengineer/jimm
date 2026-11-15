#!/usr/bin/env python3
"""Validate a jimm checkpoint (or randomly initialized model) on an image folder.

Mirrors the behaviour of timm's ``validate.py`` for the parts jimm supports:
ImageFolder splits, timm-style data configuration, top-1/top-5 accuracy, and a
``--result`` JSON summary for ``bulk_runner.py``.

Example:
    python validate.py --data-dir /data/imagenet --split validation \
        --model resnet50 --checkpoint ./output/resnet50 -b 256 --amp
"""

import argparse

from jimm.scripts_common import (
    add_data_args,
    add_model_args,
    add_preprocess_args,
    add_runtime_args,
    build_eval_loader,
    check_device,
    configure_compilation_cache,
    create_eval_model,
    evaluate,
    print_result,
    resolve_data_config,
    setup_logging,
    write_results_file,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate a jimm model on an ImageFolder split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_args(parser)
    add_model_args(parser)
    add_preprocess_args(parser)
    add_runtime_args(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logger = setup_logging("validate")
    check_device(args.device)
    configure_compilation_cache(args.compilation_cache)
    if args.batch_size <= 0 or args.workers < 0 or args.prefetch < 0:
        raise SystemExit("--batch-size must be positive and --workers/--prefetch non-negative")

    data_config = resolve_data_config(args)
    model = create_eval_model(args, data_config)
    loader = build_eval_loader(args, data_config)
    try:
        logger.info(
            "Validating %s (%d classes, %dx%d, crop %.3f, %s) on %s",
            args.model,
            data_config["num_classes"],
            data_config["img_size"],
            data_config["img_size"],
            data_config["crop_pct"],
            data_config["interpolation"],
            args.split,
        )
        metrics = evaluate(
            model,
            loader,
            amp=args.amp,
            log_freq=args.log_freq,
            logger=logger,
            prefetch=args.prefetch,
        )
    finally:
        loader.close()

    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "img_size": data_config["img_size"],
        **metrics,
    }
    if args.results_file:
        write_results_file(args.results_file, result, args.results_format)
    logger.info(
        "Top-1 %.4f  Top-5 %.4f  loss %.4f  (%d samples, %.1f samples/s)",
        result["top1"],
        result["top5"],
        result["loss"],
        result["samples"],
        result["samples_per_sec"],
    )
    print_result(result)
    return result


if __name__ == "__main__":
    main()
