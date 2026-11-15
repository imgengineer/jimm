#!/usr/bin/env python3
"""Benchmark jimm model inference and training throughput.

Mirrors timm's ``benchmark.py`` for jimm: forward-only and compiled train-step
throughput on the active JAX device. Use ``bulk_runner.py`` to sweep models.

Example:
    python benchmark.py --model resnet50 -b 128 --img-size 224 --bench both --amp
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
from flax import nnx

from jimm.checkpoint import load_checkpoint
from jimm.optim import make_optimizer
from jimm.registry import create_model, list_models
from jimm.scripts_common import (
    add_model_args,
    add_preprocess_args,
    add_runtime_args,
    check_device,
    configure_compilation_cache,
    model_param_count,
    print_result,
    resolve_data_config,
    setup_logging,
    write_results_file,
)
from jimm.train import _forward_with_precision, make_cached_train_step
from jimm.weights import load_pretrained


@nnx.jit(static_argnames=("amp",))
def _infer(model, images, amp=False):
    return _forward_with_precision(model, images, amp)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark jimm model throughput",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(parser)
    add_preprocess_args(parser)
    add_runtime_args(parser, log_freq=0)
    parser.add_argument("-b", "--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--model-list", default="", help="model name filter or file of names")
    parser.add_argument("--bench", choices=("infer", "train", "both"), default="both")
    parser.add_argument("--num-warm-iter", type=int, default=5)
    parser.add_argument("--num-bench-iter", type=int, default=20)
    return parser.parse_args(argv)


def _resolve_models(args):
    if not args.model_list:
        return [args.model]
    if args.model_list == "all":
        return list_models()
    if os.path.isfile(args.model_list):
        with open(args.model_list) as file:
            names = [line.strip() for line in file if line.strip()]
        if not names:
            raise SystemExit(f"model list is empty: {args.model_list}")
        return names
    names = list_models(args.model_list)
    if not names:
        raise SystemExit(f"no models match {args.model_list!r}")
    return names


def _make_model(name, args, data_config):
    kwargs = dict(args.model_kwargs)
    if args.global_pool is not None:
        kwargs["global_pool"] = args.global_pool
    model = create_model(
        name,
        num_classes=data_config["num_classes"],
        rngs=nnx.Rngs(args.seed),
        **kwargs,
    )
    if args.checkpoint:
        if os.path.isdir(args.checkpoint):
            load_checkpoint(args.checkpoint, model)
        else:
            load_pretrained(model, args.checkpoint)
    return model


def _samples_per_sec(iterations, batch_size, elapsed):
    return iterations * batch_size / max(elapsed, 1e-9)


def _block(result):
    if result is not None:
        jax.block_until_ready(result)


def _bench_infer(model, images, args):
    model.eval()
    result = None
    for _ in range(args.num_warm_iter):
        result = _infer(model, images, args.amp)
    _block(result)
    start = time.perf_counter()
    for _ in range(args.num_bench_iter):
        result = _infer(model, images, args.amp)
    _block(result)
    elapsed = time.perf_counter() - start
    return _samples_per_sec(args.num_bench_iter, images.shape[0], elapsed)


def _bench_train(model, images, labels, args):
    model.train()
    optimizer = make_optimizer(model, 1e-3, 0.01, 1, 1)
    step = make_cached_train_step(model, optimizer, amp=args.amp)
    result = None
    for _ in range(args.num_warm_iter):
        result = step(images, labels, 0.0)
    _block(result)
    start = time.perf_counter()
    for _ in range(args.num_bench_iter):
        result = step(images, labels, 0.0)
    _block(result)
    elapsed = time.perf_counter() - start
    return _samples_per_sec(args.num_bench_iter, images.shape[0], elapsed)


def _bench_one(name, args):
    args.model = name
    data_config = resolve_data_config(args)
    size = data_config["img_size"]
    model = _make_model(name, args, data_config)
    images = jax.random.normal(
        jax.random.PRNGKey(args.seed), (args.batch_size, size, size, 3), jnp.float32
    )
    labels = jax.random.randint(
        jax.random.PRNGKey(args.seed + 1), (args.batch_size,), 0, data_config["num_classes"]
    ).astype(jnp.int32)
    result = {
        "model": name,
        "param_count": model_param_count(model),
        "img_size": size,
        "batch_size": args.batch_size,
        "amp": args.amp,
    }
    if args.bench in ("infer", "both"):
        result["infer_samples_per_sec"] = _bench_infer(model, images, args)
    if args.bench in ("train", "both"):
        result["train_samples_per_sec"] = _bench_train(model, images, labels, args)
    return result


def main(argv=None):
    args = _parse_args(argv)
    logger = setup_logging("benchmark")
    check_device(args.device)
    configure_compilation_cache(args.compilation_cache)
    if args.batch_size <= 0 or args.num_bench_iter <= 0 or args.num_warm_iter < 0:
        raise SystemExit("invalid batch size or iteration counts")
    if args.pretrained:
        raise SystemExit(
            "no pretrained weights are registered in jimm; pass --checkpoint PATH instead"
        )
    if args.checkpoint and args.model_list:
        raise SystemExit("--checkpoint can only be used with a single --model")

    results = []
    for name in _resolve_models(args):
        try:
            result = _bench_one(name, args)
        except Exception as error:  # continue a sweep when one model does not fit
            logger.error("%s: %s", name, error)
            results.append({"model": name, "error": str(error)})
            continue
        metrics = " ".join(
            f"{key} {value:.1f}" for key, value in result.items() if key.endswith("samples_per_sec")
        )
        logger.info("%s: params %d | %s", name, result["param_count"], metrics)
        results.append(result)

    successful = [result for result in results if "error" not in result]
    if args.results_file and successful:
        write_results_file(args.results_file, successful, args.results_format)
    print_result(results[0] if len(results) == 1 else results)
    return results


if __name__ == "__main__":
    main()
