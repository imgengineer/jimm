"""Tests for features_only and intermediate feature extraction in jimm."""

import jax.numpy as jnp
import pytest
from flax import nnx

import jimm


def test_features_only_resnet():
    m = jimm.create_model(
        "resnet50", features_only=True, out_indices=(1, 2, 3, 4), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((2, 224, 224, 3), jnp.float32)
    feats = m(x)
    assert len(feats) == 4
    assert feats[0].shape == (2, 56, 56, 256)
    assert feats[1].shape == (2, 28, 28, 512)
    assert feats[2].shape == (2, 14, 14, 1024)
    assert feats[3].shape == (2, 7, 7, 2048)


def test_features_only_convnext():
    m = jimm.create_model(
        "convnext_tiny", features_only=True, out_indices=(1, 2, 3, 4), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((2, 224, 224, 3), jnp.float32)
    feats = m(x)
    assert len(feats) == 4
    assert feats[0].shape == (2, 56, 56, 96)
    assert feats[1].shape == (2, 28, 28, 192)
    assert feats[2].shape == (2, 14, 14, 384)
    assert feats[3].shape == (2, 7, 7, 768)


def test_features_only_vit():
    m = jimm.create_model(
        "vit_base_patch16_224", features_only=True, out_indices=(-2, -1), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((2, 224, 224, 3), jnp.float32)
    feats = m(x)
    assert len(feats) == 2
    assert feats[0].shape == (2, 197, 768)
    assert feats[1].shape == (2, 197, 768)


def test_features_out_of_range_raises():
    m = jimm.create_model("resnet50", features_only=True, out_indices=(0, 99), rngs=nnx.Rngs(0))
    x = jnp.ones((2, 224, 224, 3), jnp.float32)
    with pytest.raises(ValueError, match="out of range"):
        m(x)


def test_feature_info_get_timm_semantics():
    from jimm.features import FeatureInfo

    info = [
        {"num_chs": 64, "reduction": 2},
        {"num_chs": 256, "reduction": 4},
        {"num_chs": 512, "reduction": 8},
    ]
    fi = FeatureInfo(info, out_indices=(0, 2))
    # idx=None lists selected stages; an idx addresses the full metadata list.
    assert fi.get("num_chs") == [64, 512]
    assert fi.get("num_chs", 0) == 64
    assert fi.get("num_chs", 1) == 256
    assert fi.channels() == [64, 512]
    assert fi.reduction() == [2, 8]
    assert len(fi) == 2
    assert "FeatureInfo" in repr(fi)


def test_feature_extractor_generic_model():
    from jimm.features import create_feature_extractor

    class SimpleGenericModel(nnx.Module):
        def __init__(self):
            self.fc = nnx.Linear(10, 5, rngs=nnx.Rngs(0))

        def forward_features(self, x):
            return x * 2.0

        def __call__(self, x):
            return self.fc(self.forward_features(x))

    model = SimpleGenericModel()
    fe = create_feature_extractor(model)
    x = jnp.ones((2, 10), dtype=jnp.float32)
    feats = fe(x)
    assert len(feats) == 1
    assert jnp.allclose(feats[0], x * 2.0)
