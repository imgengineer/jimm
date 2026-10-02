"""Unit tests for jimm.layers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jimm.layers import (
    ClassifierMixin,
    ConvBNAct,
    DropPath,
    Mlp,
    PatchEmbed,
    SqueezeExcite,
    conv_general_dilated,
    create_act_layer,
    drop_path,
    gelu,
    global_pool_nhwc,
    hswish,
    relu6,
    use_fast_grouped_conv_grads,
)


def test_gelu_exact_erf():
    # timm/PyTorch nn.GELU() semantics, not the tanh approximation.
    x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=jnp.float32)
    expected = 0.5 * x * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))
    np.testing.assert_allclose(gelu(x), expected, rtol=1e-5, atol=1e-6)
    assert not jnp.allclose(gelu(x), nnx.gelu(x))  # tanh approximation differs


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
    assert bool(jnp.all(x_train == x_train[:, :1, :1, :1]))  # one mask per sample

    dp1 = DropPath(1.0, rngs=rngs)
    assert jnp.all(dp1(x) == 0)

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
    c3 = ConvBNAct(
        in_chs=32, out_chs=64, kernel=3, stride=2, padding="VALID", act="silu", rngs=rngs
    )
    assert c3(x2).shape == (2, 3, 3, 64)

    # 4. No BN (use_bn=False) and identity act
    c4 = ConvBNAct(
        in_chs=32, out_chs=32, kernel=1, stride=1, use_bn=False, act="identity", rngs=rngs
    )
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


def test_functional_drop_path():
    x = jnp.ones((4, 8, 8, 16), dtype=jnp.float32)
    # rate 0 is identity
    assert jnp.allclose(drop_path(x, rate=0.0), x)
    # deterministic is identity
    assert jnp.allclose(drop_path(x, rate=0.5, deterministic=True), x)
    # rate > 0
    out = drop_path(x, rate=0.5, deterministic=False)
    assert out.shape == x.shape


def test_drop_path_invalid_rate():
    with pytest.raises(ValueError, match="rate must be between 0 and 1"):
        DropPath(-0.1, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="rate must be between 0 and 1"):
        DropPath(1.5, rngs=nnx.Rngs(0))


def test_create_act_layer():
    assert create_act_layer("relu") is nnx.relu
    assert create_act_layer("gelu") is gelu
    assert create_act_layer("silu") is nnx.silu
    assert create_act_layer("hswish") is nnx.hard_swish
    assert create_act_layer("relu6") is nnx.relu6
    assert create_act_layer("identity") is None

    with pytest.raises(ValueError, match="unsupported activation"):
        create_act_layer("nonexistent_act")


def test_global_pool_pass_through():
    x = jnp.ones((2, 4, 4, 8), dtype=jnp.float32)
    assert global_pool_nhwc(x, "") is x


def test_conv_bn_act_tuple_stride():
    stride_tuple = (2, 2)
    block = ConvBNAct(8, 16, kernel=3, stride=stride_tuple, act="silu", rngs=nnx.Rngs(0))
    x = jnp.ones((2, 16, 16, 8), dtype=jnp.float32)
    out = block(x)
    assert out.shape == (2, 8, 8, 16)


def _depthwise_loss(conv, padding, groups):
    def loss(x, w):
        y = conv(
            x,
            w,
            (1, 1),
            padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=groups,
        )
        return jnp.sum(jnp.sin(y))

    return loss


@pytest.mark.parametrize(
    "size,padding", [(7, ((3, 3), (3, 3))), (5, "SAME"), (6, ((2, 3), (1, 4))), (3, "VALID")]
)
def test_full_map_depthwise_conv_matches_lax(size, padding):
    x = jax.random.normal(jax.random.key(0), (3, size, size, 4))
    w = jax.random.normal(jax.random.key(1), (size, size, 1, 4))
    with jax.default_matmul_precision("float32"):
        expected = jax.value_and_grad(
            _depthwise_loss(jax.lax.conv_general_dilated, padding, 4), argnums=(0, 1)
        )(x, w)
        actual = jax.value_and_grad(
            _depthwise_loss(conv_general_dilated, padding, 4), argnums=(0, 1)
        )(x, w)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, rtol=1e-5, atol=1e-5)


def test_conv_general_dilated_reroutes_only_full_map_depthwise():
    def rerouted(x_shape, w_shape, groups):
        loss = _depthwise_loss(conv_general_dilated, "SAME", groups)
        jaxpr = jax.make_jaxpr(loss)(jnp.zeros(x_shape), jnp.zeros(w_shape))
        return any("custom_vjp" in eqn.primitive.name for eqn in jaxpr.eqns)

    assert rerouted((2, 7, 7, 8), (7, 7, 1, 8), 8)
    assert not rerouted((2, 8, 8, 8), (7, 7, 1, 8), 8)  # map larger than the kernel
    assert not rerouted((2, 7, 7, 8), (7, 7, 8, 8), 1)  # dense convolution
    assert not rerouted((2, 7, 7, 8), (7, 7, 2, 8), 4)  # grouped, not depthwise


@pytest.mark.parametrize(
    "size,groups,dilation,stride,padding",
    [(8, 4, 2, 1, ((2, 2), (2, 2))), (7, 4, 2, 2, ((2, 2), (2, 2))), (9, 8, 3, 1, "SAME")],
)
def test_dilated_grouped_conv_matches_lax(size, groups, dilation, stride, padding):
    x = jax.random.normal(jax.random.key(0), (2, size, size, 8))
    w = jax.random.normal(jax.random.key(1), (3, 3, 8 // groups, 8))

    def loss(conv):
        def f(x, w):
            y = conv(
                x,
                w,
                (stride, stride),
                padding,
                rhs_dilation=(dilation, dilation),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                feature_group_count=groups,
            )
            return jnp.sum(jnp.sin(y))

        return f

    with jax.default_matmul_precision("float32"):
        expected = jax.value_and_grad(loss(jax.lax.conv_general_dilated), argnums=(0, 1))(x, w)
        actual = jax.value_and_grad(loss(conv_general_dilated), argnums=(0, 1))(x, w)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, rtol=1e-5, atol=1e-5)


def test_use_fast_grouped_conv_grads_routes_grouped_convs():
    class Net(nnx.Module):
        def __init__(self, rngs):
            self.dense = nnx.Conv(4, 8, (3, 3), rngs=rngs)
            self.depthwise = nnx.Conv(8, 8, (3, 3), feature_group_count=8, rngs=rngs)

    net = use_fast_grouped_conv_grads(Net(nnx.Rngs(0)))
    assert net.dense.conv_general_dilated is jax.lax.conv_general_dilated
    assert net.depthwise.conv_general_dilated is conv_general_dilated
