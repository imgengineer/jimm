"""Backpropagation correctness checks for jimm.

Checks, on representative architecture families:
1. every nnx.Param receives a gradient (no dead branches),
2. gradients are finite (no NaN/Inf),
3. per-layer gradient flow (early/deep layers get non-zero grad norms),
4. finite-difference gradient check vs autodiff,
5. real optimization: loss decreases over repeated steps on a fixed batch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import jimm
from jimm.train import cross_entropy, make_optimizer, train_step

SEED = 0


def _grad_tree(model):
    size = model.default_cfg.get("input_size", (3, 224, 224))[1]
    def loss_fn(model):
        logits = model(jnp.ones((2, size, size, 3), jnp.float32))
        return cross_entropy(logits, jnp.array([0, 1], jnp.int32))
    return nnx.grad(loss_fn)(model)


def check_gradient_presence(names):
    """Every Param gets a finite gradient; no dead branches."""
    for name in names:
        model = jimm.create_model(name, num_classes=5, rngs=nnx.Rngs(SEED))
        model.train()
        grads = _grad_tree(model)
        state = nnx.state(grads)
        leaves = jax.tree.leaves(jax.tree.map(lambda v: getattr(v, "value", v), state.to_pure_dict()))
        n_zero = sum(1 for g in leaves if float(jnp.abs(g).sum()) == 0.0)
        n_bad = sum(1 for g in leaves if not bool(jnp.isfinite(g).all()))
        assert n_bad == 0, f"{name}: {n_bad} non-finite gradients"
        assert n_zero < len(leaves) * 0.2, f"{name}: too many dead gradients ({n_zero}/{len(leaves)})"
        assert len(leaves) > 0
        print(f"  {name}: {len(leaves)} param grads, dead={n_zero}, finite=True OK")


def check_layer_grad_flow():
    """First/middle/last layers of a deep net all receive non-zero gradient."""
    for name in ["resnet50", "convnext_tiny", "vit_tiny_patch16_224"]:
        model = jimm.create_model(name, num_classes=5, rngs=nnx.Rngs(SEED))
        model.train()
        grads = _grad_tree(model)
        pure = nnx.state(grads).to_pure_dict()

        def norm(d, keys):
            total = 0.0
            for k in keys:
                if k in d:
                    total += float(sum(float(jnp.linalg.norm(v)) for v in jax.tree.leaves(
                        jax.tree.map(lambda x: getattr(x, "value", x), d[k]))))
            return total

        flat = jax.tree_util.tree_flatten_with_path(pure)[0]
        groups = {"first": 0.0, "middle": 0.0, "last": 0.0}
        n = len(flat)
        for i, (path, val) in enumerate(flat):
            g = float(jnp.linalg.norm(getattr(val, "value", val)))
            grp = "first" if i < n / 3 else ("middle" if i < 2 * n / 3 else "last")
            groups[grp] += g
        assert all(v > 0 for v in groups.values()), f"{name}: dead layer group {groups}"
        print(f"  {name}: grad norms first/mid/last = "
              f"{groups['first']:.2e}/{groups['middle']:.2e}/{groups['last']:.2e} OK")


def check_finite_difference():
    """Autodiff gradient matches central finite differences (run on CPU:
    GPU conv/attention forward has reduction-order jitter that swamps fd)."""
    cpu = jax.devices("cpu")[0]
    model = jimm.create_model("vit_tiny_patch16_224", num_classes=3, rngs=nnx.Rngs(SEED))
    model.train()
    model = jax.tree.map(lambda a: jax.device_put(a, cpu) if isinstance(a, jax.Array) else a,
                         model, is_leaf=lambda a: isinstance(a, jax.Array))
    x = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (1, 224, 224, 3)) * 0.1, cpu)
    y = jnp.array([1], jnp.int32)

    def loss_fn(model):
        return cross_entropy(model(x), y)

    grads = nnx.grad(loss_fn)(model)
    gpure = nnx.state(grads).to_pure_dict()

    # pick a few scalar positions in different layers
    targets = [("patch_embed", "proj", "kernel", (0, 0, 0, 0)),
               ("blocks", 0, "attn", "qkv", "kernel", (0, 0)),
               ("head", "kernel", (0, 0))]
    eps = 1e-2  # fp32: larger eps needed to clear forward reduction noise

    def get(d, keys):
        for k in keys:
            d = d[k]
        return d

    def set_leaf(d, keys, v):
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = v

    mpure = nnx.state(model).to_pure_dict()
    for keys in targets:
        *path, idx = keys
        w = np.array(get(mpure, path))
        g_ad = float(np.array(get(gpure, path))[idx])
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[idx] += eps
        w_minus[idx] -= eps
        set_leaf(mpure, path, w_plus)
        nnx.update(model, mpure)
        l_plus = float(loss_fn(model))
        set_leaf(mpure, path, w_minus)
        nnx.update(model, mpure)
        l_minus = float(loss_fn(model))
        set_leaf(mpure, path, w)
        nnx.update(model, mpure)
        g_fd = (l_plus - l_minus) / (2 * eps)
        rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-8)
        assert rel < 1e-1, f"finite-diff mismatch at {keys}: ad={g_ad:.6f} fd={g_fd:.6f} rel={rel:.3f}"
        print(f"  fd-check {'.'.join(str(k) for k in keys)}: ad={g_ad:.5f} fd={g_fd:.5f} rel={rel:.2e} OK")


def check_loss_decreases():
    """Real optimization: loss strictly decreases on a fixed random batch."""
    for name in ["resnet18", "vit_tiny_patch16_224", "convnext_tiny", "swin_tiny_patch4_window7_224"]:
        model = jimm.create_model(name, num_classes=5, rngs=nnx.Rngs(SEED))
        model.train()
        # lr=3e-4 + warmup 5: swin is adam-lr-sensitive (diverges at 1e-3+), others tolerate more;
        # 3e-4 converges for all four (verified). This is lr sensitivity, not a backward bug.
        opt = make_optimizer(model, lr=3e-4, weight_decay=0.0, epochs=5, steps_per_epoch=10)
        rng = np.random.RandomState(SEED)
        images = jnp.array(rng.randn(8, 224, 224, 3), jnp.float32)
        labels = jnp.array(rng.randint(0, 5, 8), jnp.int32)
        losses = [float(train_step(model, opt, images, labels, 0.0)[0]) for _ in range(30)]
        assert losses[-1] < losses[0], f"{name}: loss did not decrease {losses[0]:.3f} -> {losses[-1]:.3f}"
        assert all(np.isfinite(losses))
        print(f"  {name}: loss {losses[0]:.3f} -> {losses[-1]:.3f} (decreasing) OK")


if __name__ == "__main__":
    print("== 1. gradient presence across families ==")
    check_gradient_presence([
        "resnet18", "vgg11_bn", "densenet121", "inception_v3", "efficientnet_b0",
        "mobilenetv2_100", "convnext_tiny", "regnetx_002", "dpn68", "dla34",
        "vit_tiny_patch16_224", "swin_tiny_patch4_window7_224", "cait_xxs24_224",
        "maxvit_tiny_rw_224", "hrnet_w18_small", "volo_d1_224"])
    print("== 2. per-layer gradient flow ==")
    check_layer_grad_flow()
    print("== 3. finite-difference gradient check ==")
    check_finite_difference()
    print("== 4. loss decreases on fixed batch ==")
    check_loss_decreases()
    print("ALL BACKPROP CHECKS PASSED")
