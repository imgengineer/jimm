"""Fast full-registry backward sweep: jit-compiled backward step per model."""
import gc
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import jimm
from jimm.train import cross_entropy


@nnx.jit
def _backward_step(model, x, y):
    def loss_fn(m):
        out = m(x)
        if m.get_classifier() is None:  # encoder-only models (e.g. vit_sam)
            out = jnp.mean(out, axis=(1, 2))
        return cross_entropy(out, y)
    return nnx.grad(loss_fn)(model)


def backward_check(name):
    model = jimm.create_model(name, num_classes=5, rngs=nnx.Rngs(0))
    size = model.default_cfg.get("input_size", (3, 224, 224))[1]
    model.train()
    x = jnp.ones((2, size, size, 3), jnp.float32)
    y = jnp.array([0, 1], jnp.int32)
    grads = _backward_step(model, x, y)
    leaves = jax.tree.leaves(jax.tree.map(lambda v: getattr(v, "value", v),
                                          nnx.state(grads).to_pure_dict()))
    assert leaves, f"{name}: no gradient leaves"
    n_bad = sum(1 for g in leaves if not bool(jnp.isfinite(g).all()))
    n_nonzero = sum(1 for g in leaves if float(jnp.abs(g).sum()) > 0.0)
    assert n_bad == 0, f"{name}: {n_bad} non-finite gradients"
    assert n_nonzero > len(leaves) * 0.5, f"{name}: too many dead gradients ({n_nonzero}/{len(leaves)})"
    return len(leaves), n_nonzero


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--modules"
    if mode == "--all":
        names = jimm.list_models()
    else:
        names = [jimm.list_models(module=m)[0] for m in sorted(jimm.list_modules())]
    failed = []
    for i, name in enumerate(names):
        try:
            n, nz = backward_check(name)
            print(f"[{i + 1:>3}/{len(names)}] OK  {name}", flush=True)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {str(e)[:80]}"))
            print(f"[{i + 1:>3}/{len(names)}] FAIL {name}: {str(e)[:80]}", flush=True)
        finally:
            if (i + 1) % 10 == 0:
                gc.collect()
    print(f"\n{len(names) - len(failed)}/{len(names)} backward OK")
    for n, e in failed:
        print("  FAIL:", n, e)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
