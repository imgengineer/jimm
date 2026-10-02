"""Compare shared attention with Flax using Tokamax's device timing utilities.

Example: uv run python scripts/benchmark_attention.py --seq-len 2304 --autotune
"""

import argparse
import functools
import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import tokamax
from flax import nnx

from jimm.attention import dot_product_attention, set_attention_autotuning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2304)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--autotune", action="store_true", help="autotune Tokamax kernels instead of heuristics"
    )
    args = parser.parse_args()
    set_attention_autotuning(args.autotune)
    if min(args.batch_size, args.seq_len, args.heads, args.head_dim, args.iterations) <= 0:
        parser.error("shape dimensions and iterations must be positive")
    dtype = getattr(jnp, args.dtype)
    shape = (args.batch_size, args.seq_len, args.heads, args.head_dim)
    keys = jax.random.split(jax.random.key(0), 4)
    inputs = tuple(jax.random.normal(key, shape, dtype) for key in keys[:3])
    if args.bias:
        bias_shape = (1, args.heads, args.seq_len, args.seq_len)
        inputs += (jax.random.normal(keys[3], bias_shape) * 0.1,)
    report = {
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "flax": flax.__version__,
        "tokamax": tokamax.__version__,
        "shape": shape,
        "dtype": args.dtype,
        "bias": args.bias,
        "autotune": args.autotune,
        "results": [],
    }
    reference = jax.jit(nnx.dot_product_attention)(*inputs)
    for name, attention in (("flax", nnx.dot_product_attention), ("jimm", dot_product_attention)):
        forward = functools.partial(_forward, attention)
        result = jax.jit(forward)(inputs)
        max_error = float(jnp.abs(result.astype(jnp.float32) - reference.astype(jnp.float32)).max())
        for mode, fn in (
            ("forward", forward),
            ("forward_backward", jax.grad(functools.partial(_loss, forward))),
        ):
            timing = tokamax.benchmark(fn, inputs, iterations=args.iterations)
            memory = jax.jit(fn).lower(inputs).compile().memory_analysis()
            record = {
                "backend": name,
                "mode": mode,
                "median_ms": timing.median_evaluation_time_ms,
                "temporary_mib": memory.temp_size_in_bytes / 2**20,
                "max_absolute_error": max_error,
            }
            report["results"].append(record)
            print(json.dumps(record), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def _forward(attention, inputs):
    return attention(*inputs)


def _loss(forward, inputs):
    return forward(inputs).astype(jnp.float32).sum()


if __name__ == "__main__":
    main()
