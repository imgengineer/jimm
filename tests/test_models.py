"""Comprehensive parameterized tests for registered jimm model families and variants."""
import gc
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import jimm
from jimm.registry import create_model, list_models, list_modules
from jimm.train import cross_entropy

MODULES = sorted(list_modules())
ALL_MODELS = sorted(list_models())


def test_all_registered_model_entrypoints_instantiation():
    """Verify that every registered model entrypoint instantiates cleanly."""
    for i, name in enumerate(ALL_MODELS):
        m = create_model(name, num_classes=5, rngs=nnx.Rngs(0))
        assert isinstance(m, nnx.Module)
        assert hasattr(m, "num_features")
        assert getattr(m, "num_features") > 0
        assert hasattr(m, "default_cfg")
        assert "input_size" in getattr(m, "default_cfg")
        del m
        if (i + 1) % 20 == 0:
            gc.collect()


@pytest.mark.parametrize("module_name", MODULES)
def test_architecture_family_forward_and_backward(module_name):
    """Verify forward, classifier reset, and backward paths across model families."""
    models_in_module = jimm.list_models(module=module_name)
    name = models_in_module[0]  # representative architecture model

    # 1. Instantiate model & forward pass in training mode
    m = create_model(name, num_classes=5, rngs=nnx.Rngs(0))
    size = m.default_cfg.get("input_size", (3, 224, 224))[1]
    m.train()

    x = jnp.ones((1, size, size, 3), dtype=jnp.float32)
    y = jnp.array([0], dtype=jnp.int32)

    # 2. Backward pass & gradient verification
    def loss_fn(model):
        out = model(x)
        if model.get_classifier() is None:
            out = jnp.mean(out, axis=(1, 2))
        return cross_entropy(out, y)

    grads = nnx.grad(loss_fn)(m)
    leaves = jax.tree.leaves(jax.tree.map(lambda v: getattr(v, "value", v),
                                          nnx.state(grads).to_pure_dict()))
    assert len(leaves) > 0, f"{name}: no parameter gradient leaves"
    n_bad = sum(1 for g in leaves if not bool(jnp.isfinite(g).all()))
    assert n_bad == 0, f"{name}: {n_bad} non-finite gradients in backward pass"

    # 3. Eval mode & feature extractor check
    m.eval()
    logits = m(x)
    assert bool(jnp.isfinite(logits).all()), f"{name} forward produced NaN/Inf"
    if m.get_classifier() is not None:
        assert logits.shape == (1, 5), f"{name} expected logits shape (1, 5), got {logits.shape}"
        m.reset_classifier(0)
        feats = m(x)
        assert feats.shape[-1] == m.num_features, f"{name} feature shape mismatch: {feats.shape} vs {m.num_features}"
    else:
        assert logits.shape[-1] == m.num_features

    # Clean up memory
    del m, x, logits, grads
    gc.collect()


def test_variant_constructor_smoke():
    for name in ("convnextv2_tiny", "dm_nfnet_f0", "eca_vovnet39b", "regnety_008_tv"):
        model = create_model(name, num_classes=5, rngs=nnx.Rngs(0))
        assert model.num_features > 0
        assert "input_size" in model.default_cfg


def test_extra_multi_architecture_modules():
    """Tests co-located distinct architectures within the same file (e.g. ResMLP in mlp_mixer, SwinV2 in swin_transformer)."""
    for name in ["resmlp_12_224", "swinv2_tiny_window8_256", "darknet53"]:
        m = create_model(name, num_classes=5, rngs=nnx.Rngs(0))
        m.eval()
        size = m.default_cfg.get("input_size", (3, 224, 224))[1]
        x = jnp.ones((1, size, size, 3), dtype=jnp.float32)
        out = m(x)
        assert bool(jnp.isfinite(out).all())
        m.reset_classifier(0)
        assert m(x).shape[-1] == m.num_features
        del m, x
        gc.collect()
