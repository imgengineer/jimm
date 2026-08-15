"""Unit tests for jimm.layers."""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jimm.layers import (
    ClassifierMixin,
    ConvBNAct,
    DropPath,
    Mlp,
    PatchEmbed,
    SqueezeExcite,
    global_pool_nhwc,
    hswish,
    relu6,
)


def test_global_pool_nhwc():
    x = jnp.arange(2 * 4 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 4, 8)
    avg_p = global_pool_nhwc(x, "avg")
    assert avg_p.shape == (2, 8)
    assert jnp.allclose(avg_p, jnp.mean(x, axis=(1, 2)))

    max_p = global_pool_nhwc(x, "max")
    assert max_p.shape == (2, 8)
    assert jnp.allclose(max_p, jnp.max(x, axis=(1, 2)))

    with pytest.raises(ValueError, match="unsupported pool"):
        global_pool_nhwc(x, "invalid_pool")


def test_activations():
    x = jnp.array([-10.0, -3.0, 0.0, 3.0, 10.0], dtype=jnp.float32)
    r6 = relu6(x)
    assert jnp.allclose(r6, jnp.array([0.0, 0.0, 0.0, 3.0, 6.0]))

    hs = hswish(x)
    assert hs.shape == x.shape
    assert float(hs[2]) == 0.0  # hswish(0) = 0


def test_drop_path():
    rngs = nnx.Rngs(0)
    # rate = 0 is a no-op
    dp0 = DropPath(0.0, rngs=rngs)
    x = jnp.ones((4, 8, 8, 16), dtype=jnp.float32)
    assert jnp.allclose(dp0(x), x)

    # rate > 0 in eval mode (deterministic=True) is a no-op
    dp = DropPath(0.2, rngs=rngs)
    dp.deterministic = True
    assert jnp.allclose(dp(x), x)

    # rate > 0 in train mode drops some samples
    dp.deterministic = False
    x_train = dp(x)
    assert x_train.shape == x.shape
    assert bool(jnp.isfinite(x_train).all())

    # Works on 3D token sequences (B, N, C) without IndexError
    tokens = jnp.ones((4, 16, 32), dtype=jnp.float32)
    t_out = dp(tokens)
    assert t_out.shape == tokens.shape


def test_patch_embed():
    rngs = nnx.Rngs(0)
    pe = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=192, rngs=rngs)
    assert pe.num_patches == 196
    assert pe.grid_size == (14, 14)

    x = jnp.ones((2, 224, 224, 3), dtype=jnp.float32)
    out = pe(x)
    assert out.shape == (2, 14, 14, 192)


def test_mlp():
    rngs = nnx.Rngs(0)
    mlp = Mlp(dim=64, hidden_dim=256, drop=0.1, rngs=rngs)
    x = jnp.ones((2, 16, 64), dtype=jnp.float32)
    out = mlp(x)
    assert out.shape == (2, 16, 64)

    # default hidden_dim = dim
    mlp2 = Mlp(dim=64, rngs=rngs)
    assert mlp2(x).shape == (2, 16, 64)


def test_squeeze_excite():
    rngs = nnx.Rngs(0)
    se = SqueezeExcite(chs=64, rd_ratio=0.25, rngs=rngs)
    x = jnp.ones((2, 8, 8, 64), dtype=jnp.float32)
    out = se(x)
    assert out.shape == (2, 8, 8, 64)
    assert bool(jnp.isfinite(out).all())


def test_conv_bn_act():
    rngs = nnx.Rngs(0)
    # 1. 2D int kernel with standard SAME padding
    c1 = ConvBNAct(in_chs=16, out_chs=32, kernel=3, stride=2, act="relu", rngs=rngs)
    x1 = jnp.ones((2, 16, 16, 16), dtype=jnp.float32)
    assert c1(x1).shape == (2, 8, 8, 32)

    # 2. Tuple asymmetric kernel (e.g. 1x7 in Inception)
    c2 = ConvBNAct(in_chs=32, out_chs=32, kernel=(1, 7), stride=1, act="hswish", rngs=rngs)
    x2 = jnp.ones((2, 8, 8, 32), dtype=jnp.float32)
    assert c2(x2).shape == (2, 8, 8, 32)

    # 3. VALID padding
    c3 = ConvBNAct(in_chs=32, out_chs=64, kernel=3, stride=2, padding="VALID", act="silu", rngs=rngs)
    assert c3(x2).shape == (2, 3, 3, 64)

    # 4. No BN (use_bn=False) and identity act
    c4 = ConvBNAct(in_chs=32, out_chs=32, kernel=1, stride=1, use_bn=False, act="identity", rngs=rngs)
    assert c4(x2).shape == (2, 8, 8, 32)

    # 5. Grouped convolution
    c5 = ConvBNAct(in_chs=32, out_chs=32, kernel=3, stride=1, groups=32, act="gelu", rngs=rngs)
    assert c5(x2).shape == (2, 8, 8, 32)


def test_classifier_mixin():
    class DummyConvNet(ClassifierMixin, nnx.Module):
        def __init__(self, num_classes=10):
            self.num_classes = num_classes
            self.global_pool = "avg"
            self.num_features = 64
            self.head_drop = nnx.Dropout(0.0, rngs=nnx.Rngs(0))
            self.fc = nnx.Linear(64, num_classes, rngs=nnx.Rngs(0)) if num_classes > 0 else None

    # Test get_classifier & forward_head
    m = DummyConvNet(num_classes=10)
    assert m.get_classifier() is m.fc
    x = jnp.ones((2, 4, 4, 64), dtype=jnp.float32)
    out = m.forward_head(x)
    assert out.shape == (2, 10)

    # Test reset_classifier to new num_classes
    m.reset_classifier(5, global_pool="max")
    assert m.num_classes == 5
    assert m.global_pool == "max"
    assert m.forward_head(x).shape == (2, 5)

    # Test reset_classifier to 0 (feature mode)
    m.reset_classifier(0)
    assert m.get_classifier() is None
    assert m.num_classes == 0
    assert m.forward_head(x).shape == (2, 64)

    # Test cannot re-add classifier to num_classes=0 model
    m_zero = DummyConvNet(num_classes=0)
    with pytest.raises(RuntimeError, match="cannot re-add classifier"):
        m_zero.reset_classifier(10)

    # Custom _classifier_attr = "head"
    class DummyViT(ClassifierMixin, nnx.Module):
        _classifier_attr = "head"
        _default_global_pool = ""

        def __init__(self, num_classes=10):
            self.num_classes = num_classes
            self.global_pool = ""
            self.num_features = 64
            self.head = nnx.Linear(64, num_classes, rngs=nnx.Rngs(0))

    m_vit = DummyViT(num_classes=10)
    assert m_vit.get_classifier() is m_vit.head
    m_vit.reset_classifier(0)
    assert m_vit.global_pool == ""
