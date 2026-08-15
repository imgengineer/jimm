"""Tests for features_only and intermediate feature extraction in jimm."""
import jax.numpy as jnp
from flax import nnx
import pytest
import jimm


def test_features_only_resnet():
    m = jimm.create_model("resnet50", features_only=True, out_indices=(1, 2, 3, 4), rngs=nnx.Rngs(0))
    x = jnp.ones((2, 224, 224, 3), jnp.float32)
    feats = m(x)
    assert len(feats) == 4
    assert feats[0].shape == (2, 56, 56, 256)
    assert feats[1].shape == (2, 28, 28, 512)
    assert feats[2].shape == (2, 14, 14, 1024)
    assert feats[3].shape == (2, 7, 7, 2048)


def test_features_only_convnext():
    m = jimm.create_model("convnext_tiny", features_only=True, out_indices=(1, 2, 3, 4), rngs=nnx.Rngs(0))
    x = jnp.ones((2, 224, 224, 3), jnp.float32)
    feats = m(x)
    assert len(feats) == 4
    assert feats[0].shape == (2, 56, 56, 96)
    assert feats[1].shape == (2, 28, 28, 192)
    assert feats[2].shape == (2, 14, 14, 384)
    assert feats[3].shape == (2, 7, 7, 768)


def test_features_only_vit():
    m = jimm.create_model("vit_base_patch16_224", features_only=True, out_indices=(-2, -1), rngs=nnx.Rngs(0))
    x = jnp.ones((2, 224, 224, 3), jnp.float32)
    feats = m(x)
    assert len(feats) == 2
    assert feats[0].shape == (2, 197, 768)
    assert feats[1].shape == (2, 197, 768)
