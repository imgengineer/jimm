"""Forward-pass smoke checks for registered jimm models."""

import jax.numpy as jnp

import jimm
from jimm import create_model, list_models


def _require(condition, message=None):
    if not condition:
        raise AssertionError(message)


REPRESENTATIVE_MODELS = [
    "resnet18",
    "vit_tiny_patch16_224",
    "swin_tiny_patch4_window7_224",
    "convnext_tiny",
    "efficientnet_b0",
]


def check_all_models_forward(mode="representative"):
    """Test forward pass + feature mode across models."""
    if mode == "all":
        models_to_test = list_models()
        print(f"Exhaustive test across all {len(models_to_test)} registered models...")
    elif mode == "modules":
        modules = sorted(jimm.list_modules())
        models_to_test = [jimm.list_models(module=m)[0] for m in modules]
        print(f"Testing {len(models_to_test)} models (1 per architecture module)...")
    else:
        models_to_test = REPRESENTATIVE_MODELS
        print(f"Testing {len(models_to_test)} core representative models...")

    for i, name in enumerate(models_to_test):
        m = create_model(name, num_classes=7)
        size = m.default_cfg.get("input_size", (3, 224, 224))[1]
        x = jnp.zeros((1, size, size, 3), jnp.float32)
        m.eval()
        logits = m(x)
        _require(bool(jnp.isfinite(logits).all()), f"NaN in {name}")
        if (
            m.get_classifier() is None
        ):  # encoder-only models (e.g. vit_sam): feature maps in, features out
            _require(logits.shape[-1] == m.num_features, (name, logits.shape))
            continue
        _require(logits.shape == (1, 7), (name, logits.shape))
        m.reset_classifier(0)
        feats = m(x)
        _require(feats.shape[-1] == m.num_features, (name, feats.shape, m.num_features))
        print(f"  [{i + 1:>2}/{len(models_to_test)}] {name:<30} OK")


if __name__ == "__main__":
    import sys

    mode = (
        "all"
        if "--all" in sys.argv
        else ("modules" if "--modules" in sys.argv else "representative")
    )
    check_all_models_forward(mode=mode)
    print("ALL FORWARD CHECKS PASSED")
