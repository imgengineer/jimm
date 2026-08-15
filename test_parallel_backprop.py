"""Parallel backward pass verification on CPU across all 94 architecture module families."""
import multiprocessing as mp
import time

import jax
import jax.numpy as jnp
from flax import nnx

import jimm
from jimm.train import cross_entropy


def check_one(name):
    try:
        model = jimm.create_model(name, num_classes=5, rngs=nnx.Rngs(0))
        size = model.default_cfg.get("input_size", (3, 224, 224))[1]
        model.train()
        x = jnp.ones((2, size, size, 3), jnp.float32)
        y = jnp.array([0, 1], jnp.int32)

        def loss_fn(m):
            out = m(x)
            if m.get_classifier() is None:
                out = jnp.mean(out, axis=(1, 2))
            return cross_entropy(out, y)

        grads = nnx.grad(loss_fn)(model)
        leaves = jax.tree.leaves(jax.tree.map(lambda v: getattr(v, "value", v),
                                              nnx.state(grads).to_pure_dict()))
        n_bad = sum(1 for g in leaves if not bool(jnp.isfinite(g).all()))
        n_nz = sum(1 for g in leaves if float(jnp.abs(g).sum()) > 0.0)
        assert n_bad == 0, f"{name}: {n_bad} non-finite gradients"
        assert n_nz > len(leaves) * 0.5, f"{name}: dead gradients {n_nz}/{len(leaves)}"
        return name, True, f"{n_nz}/{len(leaves)} grads"
    except Exception as e:
        return name, False, f"{type(e).__name__}: {str(e)[:70]}"


def main():
    names = [jimm.list_models(module=m)[0] for m in sorted(jimm.list_modules())]
    print(f"Running parallel backward verification across all {len(names)} architecture modules with 4 workers...")
    t0 = time.time()
    with mp.Pool(4) as pool:
        results = pool.map(check_one, names)
    dt = time.time() - t0

    failed = [r for r in results if not r[1]]
    print(f"\nCompleted in {dt:.1f}s!")
    print(f"Result: {len(results) - len(failed)}/{len(results)} architecture families BACKWARD PASS OK!")
    if failed:
        for name, ok, msg in failed:
            print("  FAIL:", name, msg)
        return 1
    else:
        print("  >>> ALL 94 ARCHITECTURE MODULE FAMILIES BACKWARD PASS 100% SUCCESSFUL & FINITE! <<<")
        return 0


if __name__ == "__main__":
    mp.set_start_method("spawn")
    raise SystemExit(main())
