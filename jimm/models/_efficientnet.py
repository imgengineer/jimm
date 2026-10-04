"""EfficientNet / MobileNetV3 family in flax nnx, NHWC. Mirrors timm's efficientnet builder.

timm builds EfficientNet, MobileNet V1-V4, MNASNet, MixNet, FBNet, LCNet, TinyNet,
HardCoReNAS and related models from decoded block strings. ``_efficientnet_cfgs`` records
the resulting block constructor arguments per model; this module rebuilds the same blocks:
depthwise-separable, inverted-residual (with CondConv experts or MixConv kernels),
edge-residual, conv-bn-act, universal inverted residual and mobile multi-query attention,
with BatchNorm/GroupNorm/LayerNorm/EvoNorm, squeeze-excite or global-context attention,
anti-aliased downsampling and TF "SAME" padding.
"""

import itertools
import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, gelu, make_divisible
from ..registry import _cfg, register_model
from ._efficientnet_cfgs import EFF_CFGS, MODEL_DEFAULTS


def _hard_sigmoid(x):
    return nnx.relu6(x + 3.0) / 6.0


_ACTS = {
    "ReLU": nnx.relu,
    "ReLU6": nnx.relu6,
    "SiLU": nnx.silu,
    "Swish": nnx.silu,
    "Hardswish": lambda x: x * _hard_sigmoid(x),
    "GELU": gelu,
    "Sigmoid": nnx.sigmoid,
    "sigmoid": nnx.sigmoid,
    "hard_sigmoid": _hard_sigmoid,
}


def _pad(kernel, stride, dilation, pad_type):
    """timm ``create_conv2d`` padding: symmetric PyTorch padding, or TF "SAME"."""
    if pad_type == "same":
        return "SAME"
    p = ((stride - 1) + dilation * (kernel - 1)) // 2
    return ((p, p), (p, p))


def _split_channels(chs, groups):
    split = [chs // groups] * groups
    split[0] += chs - sum(split)
    return split


class MixedConv(nnx.Module):
    """timm ``MixedConv2d``: channel groups convolved with different kernel sizes."""

    def __init__(self, in_chs, out_chs, kernels, stride, dilation, pad_type, depthwise, *, rngs):
        ins, outs = _split_channels(in_chs, len(kernels)), _split_channels(out_chs, len(kernels))
        self.splits = tuple(itertools.accumulate(ins))[:-1]
        self.convs = nnx.List(
            [
                nnx.Conv(
                    i,
                    o,
                    (k, k),
                    strides=stride,
                    padding=_pad(k, stride, dilation, pad_type),
                    kernel_dilation=dilation,
                    feature_group_count=i if depthwise else 1,
                    use_bias=False,
                    rngs=rngs,
                )  # fmt: skip
                for k, i, o in zip(kernels, ins, outs)
            ]
        )

    def __call__(self, x):
        xs = jnp.split(x, self.splits, axis=-1)
        return jnp.concatenate([c(xi) for c, xi in zip(self.convs, xs)], axis=-1)


class CondConv(nnx.Module):
    """timm ``CondConv2d``: per-sample kernels mixed from ``num_experts`` expert kernels.

    ``kernel`` is (out * in/groups * k * k, experts), flattened in PyTorch's OIHW order.
    """

    def __init__(
        self, in_chs, out_chs, kernel, stride, dilation, groups, pad_type, experts, *, rngs
    ):
        self.shape = (out_chs, in_chs // groups, kernel, kernel)
        self.stride, self.dilation, self.groups = stride, dilation, groups
        self.padding = _pad(kernel, stride, dilation, pad_type)
        fan_in = math.prod(self.shape[1:])
        init = nnx.initializers.normal(fan_in**-0.5)
        self.kernel = nnx.Param(init(rngs.params(), (math.prod(self.shape), experts)))

    def __call__(self, x, routing):
        w = (routing @ self.kernel[...].T).reshape(-1, *self.shape).transpose(0, 3, 4, 2, 1)

        def conv(xi, wi):
            return jax.lax.conv_general_dilated(
                xi[None], wi.astype(xi.dtype), (self.stride,) * 2, self.padding,
                rhs_dilation=(self.dilation,) * 2, feature_group_count=self.groups,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )[0]  # fmt: skip

        return jax.vmap(conv)(x, w)


def _conv(in_chs, out_chs, kernel, stride=1, dilation=1, groups=1, pad_type="", bias=False,
          experts=0, *, rngs):  # fmt: skip
    if isinstance(kernel, (list, tuple)):
        return MixedConv(
            in_chs, out_chs, kernel, stride, dilation, pad_type, groups == in_chs, rngs=rngs
        )
    if experts:
        return CondConv(
            in_chs, out_chs, kernel, stride, dilation, groups, pad_type, experts, rngs=rngs
        )
    return nnx.Conv(
        in_chs, out_chs, (kernel, kernel), strides=stride,
        padding=_pad(kernel, stride, dilation, pad_type), kernel_dilation=dilation,
        feature_group_count=groups, use_bias=bias, rngs=rngs,
    )  # fmt: skip


def _apply(conv, x, routing=None):
    return conv(x, routing) if isinstance(conv, CondConv) else conv(x)


class BatchNormAct(BatchNorm):
    def __init__(self, chs, eps=1e-5, act=None, *, rngs):
        super().__init__(chs, epsilon=eps, rngs=rngs)
        self.act = act

    def __call__(self, x):
        x = super().__call__(x)
        return x if self.act is None else self.act(x)


class GroupNormAct(nnx.GroupNorm):
    def __init__(self, chs, group_size, act=None, *, rngs):
        super().__init__(chs, num_groups=chs // group_size, epsilon=1e-5, rngs=rngs)
        self.act = act

    def __call__(self, x):
        x = super().__call__(x)
        return x if self.act is None else self.act(x)


class LayerNormAct2d(nnx.LayerNorm):
    def __init__(self, chs, act=None, *, rngs):
        super().__init__(chs, epsilon=1e-5, rngs=rngs)
        self.act = act

    def __call__(self, x):
        x = super().__call__(x)
        return x if self.act is None else self.act(x)


class EvoNorm2dS0(nnx.Module):
    """timm ``EvoNorm2dS0``: ``x * sigmoid(v * x) / group_std(x)`` then affine (its own act)."""

    def __init__(self, chs, group_size, apply_act=True, eps=1e-5):
        self.groups, self.eps = chs // group_size, eps
        self.scale = nnx.Param(jnp.ones((chs,)))
        self.bias = nnx.Param(jnp.zeros((chs,)))
        self.v = nnx.Param(jnp.ones((chs,))) if apply_act else None

    def __call__(self, x):
        if self.v is not None:
            B, H, W, C = x.shape
            g = x.reshape(B, H, W, self.groups, C // self.groups).astype(jnp.float32)
            var = jnp.var(g, axis=(1, 2, 4), keepdims=True)
            std = jnp.sqrt(var + self.eps)
            std = jnp.broadcast_to(std, g.shape).reshape(x.shape).astype(x.dtype)
            x = x * nnx.sigmoid(x * self.v[...]) / std
        return x * self.scale[...] + self.bias[...]


def _norm_act(norm, chs, act, apply_act=True, *, rngs):
    """Builds timm's norm-act layer from a recorded ``norm_layer`` descriptor."""
    act = act if apply_act else None
    kind, kw = (norm, {}) if isinstance(norm, str) else norm
    if kind == "BatchNorm2d":
        return BatchNormAct(chs, kw.get("eps", 1e-5), act, rngs=rngs)
    if kind == "GroupNormAct":
        return GroupNormAct(chs, kw["group_size"], act, rngs=rngs)
    if kind == "LayerNormAct2d":
        return LayerNormAct2d(chs, act, rngs=rngs)
    if kind == "EvoNorm2dS0":
        return EvoNorm2dS0(chs, kw["group_size"], apply_act)
    raise ValueError(f"unsupported norm layer {norm!r}")


class BlurPool(nnx.Module):
    """timm ``BlurPool2d`` (3x3 binomial filter, stride 2); ``blurpc`` pads with zeros."""

    def __init__(self, stride, pad_mode):
        self.stride, self.pad_mode = stride, pad_mode

    def __call__(self, x):
        C = x.shape[-1]
        f = jnp.array([1.0, 2.0, 1.0], x.dtype) / 4
        k = jnp.broadcast_to((f[:, None] * f[None, :])[:, :, None, None], (3, 3, 1, C))
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode=self.pad_mode)
        return jax.lax.conv_general_dilated(
            x, k, (self.stride,) * 2, "VALID", feature_group_count=C,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )  # fmt: skip


def _aa(aa_layer, stride):
    """timm ``create_aa`` (applied only where the conv would have strided)."""
    if not aa_layer or stride == 1:
        return None
    if aa_layer == "avg":
        return lambda x: nnx.avg_pool(x, (stride, stride), strides=(stride, stride))
    return BlurPool(stride, "constant" if aa_layer == "blurpc" else "reflect")


def _round_channels(chs, multiplier=1.0, divisor=8, channel_min=None, round_limit=0.9):
    if not multiplier:
        return chs
    return make_divisible(chs * multiplier, divisor, channel_min, round_limit=round_limit)


class SqueezeExcite(nnx.Module):
    def __init__(self, chs, act, rd_ratio=0.25, gate_layer="Sigmoid", force_act_layer=None,
                 rd_round_fn=None, *, rngs):  # fmt: skip
        if rd_round_fn is None:
            rd = round(chs * rd_ratio)
        else:
            name, kw = (rd_round_fn, {}) if isinstance(rd_round_fn, str) else rd_round_fn
            assert name == "round_channels", rd_round_fn
            rd = _round_channels(chs * rd_ratio, **kw)
        self.act = _ACTS[force_act_layer] if force_act_layer else act
        self.gate = _ACTS[gate_layer]
        self.conv_reduce = nnx.Conv(chs, rd, (1, 1), rngs=rngs)
        self.conv_expand = nnx.Conv(rd, chs, (1, 1), rngs=rngs)

    def __call__(self, x):
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        return x * self.gate(self.conv_expand(self.act(self.conv_reduce(s))))


class _ConvMlp(nnx.Module):
    def __init__(self, chs, hidden, act, *, rngs):
        self.fc1 = nnx.Conv(chs, hidden, (1, 1), rngs=rngs)
        self.norm = nnx.LayerNorm(hidden, epsilon=1e-6, rngs=rngs)
        self.act = act
        self.fc2 = nnx.Conv(hidden, chs, (1, 1), rngs=rngs)

    def __call__(self, x):
        return self.fc2(self.act(self.norm(self.fc1(x))))


class GlobalContext(nnx.Module):
    """timm ``GlobalContext`` ('gc'): attention-pooled context gating the channels."""

    def __init__(self, chs, act, rd_ratio=1 / 8, *, rngs):
        rd = make_divisible(chs * rd_ratio, 1, round_limit=0.0)
        self.conv_attn = nnx.Conv(chs, 1, (1, 1), rngs=rngs)
        self.mlp_scale = _ConvMlp(chs, rd, act, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        attn = jax.nn.softmax(self.conv_attn(x).reshape(B, H * W), axis=-1)
        context = jnp.einsum("bnc,bn->bc", x.reshape(B, H * W, C), attn)[:, None, None, :]
        return x * nnx.sigmoid(self.mlp_scale(context))


def _se(se_layer, chs, act, *, rngs):
    if not se_layer:
        return None
    name, kw = (se_layer, {}) if isinstance(se_layer, str) else se_layer
    if name in ("gc", "GlobalContext"):
        return GlobalContext(chs, act, **kw, rngs=rngs)
    return SqueezeExcite(chs, act, **kw, rngs=rngs)


def _groups(group_size, chs):
    return chs // group_size if group_size else 1


class _Block(nnx.Module):
    """Shared residual tail: drop path plus the identity shortcut when ``has_skip``."""

    def _residual(self, x, shortcut):
        return self.drop_path(x) + shortcut if self.has_skip else x


class DepthwiseSeparableConv(_Block):
    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1,
                 pad_type="", noskip=False, pw_kernel_size=1, pw_act=False, s2d=0, act=None,
                 norm_layer=None, aa_layer=None, se_layer=None, drop_path=0.0, *, rngs):  # fmt: skip
        self.has_skip = stride == 1 and in_chs == out_chs and not noskip
        use_aa = bool(aa_layer) and stride > 1
        dw_pad = pad_type
        conv_s2d = bn_s2d = None
        if s2d == 1:
            sd = in_chs * 4
            conv_s2d = _conv(in_chs, sd, 2, 2, pad_type="same", rngs=rngs)
            bn_s2d = _norm_act(norm_layer, sd, act, rngs=rngs)
            dw_kernel_size = (dw_kernel_size + 1) // 2
            dw_pad = "same" if dw_kernel_size == 2 else pad_type
            in_chs, use_aa = sd, False
        self.conv_s2d, self.bn_s2d = conv_s2d, bn_s2d
        self.conv_dw = _conv(
            in_chs, in_chs, dw_kernel_size, 1 if use_aa else stride, dilation,
            _groups(group_size, in_chs), dw_pad, rngs=rngs,
        )  # fmt: skip
        self.bn1 = _norm_act(norm_layer, in_chs, act, rngs=rngs)
        self.aa = _aa(aa_layer, stride) if use_aa else None
        self.se = _se(se_layer, in_chs, act, rngs=rngs)
        self.conv_pw = _conv(in_chs, out_chs, pw_kernel_size, pad_type=pad_type, rngs=rngs)
        self.bn2 = _norm_act(norm_layer, out_chs, act, pw_act, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        if self.conv_s2d is not None:
            x = self.bn_s2d(self.conv_s2d(x))
        x = self.bn1(self.conv_dw(x))
        if self.aa is not None:
            x = self.aa(x)
        if self.se is not None:
            x = self.se(x)
        x = self.bn2(self.conv_pw(x))
        return self._residual(x, shortcut)


class InvertedResidual(_Block):
    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1,
                 pad_type="", noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 s2d=0, act=None, norm_layer=None, aa_layer=None, se_layer=None, num_experts=0,
                 drop_path=0.0, *, rngs):  # fmt: skip
        self.has_skip = in_chs == out_chs and stride == 1 and not noskip
        use_aa = bool(aa_layer) and stride > 1
        dw_pad = pad_type
        conv_s2d = bn_s2d = None
        if s2d == 1:
            sd = in_chs * 4
            conv_s2d = _conv(in_chs, sd, 2, 2, pad_type="same", rngs=rngs)
            bn_s2d = _norm_act(norm_layer, sd, act, rngs=rngs)
            dw_kernel_size = (dw_kernel_size + 1) // 2
            dw_pad = "same" if dw_kernel_size == 2 else pad_type
            in_chs, use_aa = sd, False
        self.conv_s2d, self.bn_s2d = conv_s2d, bn_s2d
        mid = make_divisible(in_chs * exp_ratio)
        e = num_experts
        self.routing_fn = nnx.Linear(in_chs, e, rngs=rngs) if e else None
        self.conv_pw = _conv(in_chs, mid, exp_kernel_size, pad_type=pad_type, experts=e, rngs=rngs)
        self.bn1 = _norm_act(norm_layer, mid, act, rngs=rngs)
        self.conv_dw = _conv(
            mid, mid, dw_kernel_size, 1 if use_aa else stride, dilation, _groups(group_size, mid),
            dw_pad, experts=e, rngs=rngs,
        )  # fmt: skip
        self.bn2 = _norm_act(norm_layer, mid, act, rngs=rngs)
        self.aa = _aa(aa_layer, stride) if use_aa else None
        self.se = _se(se_layer, mid, act, rngs=rngs)
        self.conv_pwl = _conv(mid, out_chs, pw_kernel_size, pad_type=pad_type, experts=e, rngs=rngs)
        self.bn3 = _norm_act(norm_layer, out_chs, act, False, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        routing = None
        if self.routing_fn is not None:
            routing = nnx.sigmoid(self.routing_fn(jnp.mean(x, axis=(1, 2))))
        if self.conv_s2d is not None:
            x = self.bn_s2d(self.conv_s2d(x))
        x = self.bn1(_apply(self.conv_pw, x, routing))
        x = self.bn2(_apply(self.conv_dw, x, routing))
        if self.aa is not None:
            x = self.aa(x)
        if self.se is not None:
            x = self.se(x)
        x = self.bn3(_apply(self.conv_pwl, x, routing))
        return self._residual(x, shortcut)


class EdgeResidual(_Block):
    def __init__(self, in_chs, out_chs, exp_kernel_size=3, stride=1, dilation=1, group_size=0,
                 pad_type="", force_in_chs=0, noskip=False, exp_ratio=1.0, pw_kernel_size=1,
                 act=None, norm_layer=None, aa_layer=None, se_layer=None, drop_path=0.0, *, rngs):  # fmt: skip
        mid = make_divisible((force_in_chs if force_in_chs > 0 else in_chs) * exp_ratio)
        self.has_skip = in_chs == out_chs and stride == 1 and not noskip
        use_aa = bool(aa_layer) and stride > 1
        self.conv_exp = _conv(
            in_chs, mid, exp_kernel_size, 1 if use_aa else stride, dilation,
            _groups(group_size, mid), pad_type, rngs=rngs,
        )  # fmt: skip
        self.bn1 = _norm_act(norm_layer, mid, act, rngs=rngs)
        self.aa = _aa(aa_layer, stride) if use_aa else None
        self.se = _se(se_layer, mid, act, rngs=rngs)
        self.conv_pwl = _conv(mid, out_chs, pw_kernel_size, pad_type=pad_type, rngs=rngs)
        self.bn2 = _norm_act(norm_layer, out_chs, act, False, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        x = self.bn1(self.conv_exp(x))
        if self.aa is not None:
            x = self.aa(x)
        if self.se is not None:
            x = self.se(x)
        x = self.bn2(self.conv_pwl(x))
        return self._residual(x, shortcut)


class ConvBnAct(_Block):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, dilation=1, group_size=0,
                 pad_type="", skip=False, act=None, norm_layer=None, aa_layer=None,
                 drop_path=0.0, *, rngs):  # fmt: skip
        self.has_skip = skip and stride == 1 and in_chs == out_chs
        use_aa = bool(aa_layer) and stride > 1
        self.conv = _conv(
            in_chs, out_chs, kernel_size, 1 if use_aa else stride, dilation,
            _groups(group_size, in_chs), pad_type, rngs=rngs,
        )  # fmt: skip
        self.bn1 = _norm_act(norm_layer, out_chs, act, rngs=rngs)
        self.aa = _aa(aa_layer, stride) if use_aa else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        x = self.bn1(self.conv(x))
        if self.aa is not None:
            x = self.aa(x)
        return self._residual(x, shortcut)


class ConvNormAct(nnx.Module):
    """timm ``ConvNormAct`` (conv, norm-act, optional anti-aliasing after a strided conv)."""

    def __init__(self, in_chs, out_chs, kernel, stride=1, dilation=1, groups=1, pad_type="",
                 apply_act=True, act=None, norm_layer=None, aa_layer=None, *, rngs):  # fmt: skip
        use_aa = bool(aa_layer) and stride > 1
        self.conv = _conv(
            in_chs, out_chs, kernel, 1 if use_aa else stride, dilation, groups, pad_type, rngs=rngs
        )
        self.bn = _norm_act(norm_layer, out_chs, act, apply_act, rngs=rngs)
        self.aa = _aa(aa_layer, stride) if use_aa else None

    def __call__(self, x):
        x = self.bn(self.conv(x))
        return x if self.aa is None else self.aa(x)


class LayerScale2d(nnx.Module):
    def __init__(self, dim, init_values):
        self.gamma = nnx.Param(jnp.full((dim,), init_values, jnp.float32))

    def __call__(self, x):
        return x * self.gamma[...]


class UniversalInvertedResidual(_Block):
    def __init__(self, in_chs, out_chs, dw_kernel_size_start=0, dw_kernel_size_mid=3,
                 dw_kernel_size_end=0, stride=1, dilation=1, group_size=1, pad_type="",
                 noskip=False, exp_ratio=1.0, act=None, norm_layer=None, aa_layer=None,
                 se_layer=None, layer_scale_init_value=1e-5, drop_path=0.0, *, rngs):  # fmt: skip
        self.has_skip = in_chs == out_chs and stride == 1 and not noskip
        common = dict(dilation=dilation, pad_type=pad_type, act=act, norm_layer=norm_layer)
        dw_start = dw_mid = dw_end = None
        if dw_kernel_size_start:
            dw_start = ConvNormAct(
                in_chs, in_chs, dw_kernel_size_start, stride if not dw_kernel_size_mid else 1,
                groups=_groups(group_size, in_chs), apply_act=False, aa_layer=aa_layer,
                rngs=rngs, **common,
            )  # fmt: skip
        mid = make_divisible(in_chs * exp_ratio)
        self.dw_start = dw_start
        self.pw_exp = ConvNormAct(
            in_chs, mid, 1, pad_type=pad_type, act=act, norm_layer=norm_layer, rngs=rngs
        )
        if dw_kernel_size_mid:
            dw_mid = ConvNormAct(
                mid, mid, dw_kernel_size_mid, stride, groups=_groups(group_size, mid),
                aa_layer=aa_layer, rngs=rngs, **common,
            )  # fmt: skip
        self.dw_mid = dw_mid
        self.se = _se(se_layer, mid, act, rngs=rngs)
        self.pw_proj = ConvNormAct(
            mid, out_chs, 1, pad_type=pad_type, apply_act=False, act=act, norm_layer=norm_layer,
            rngs=rngs,
        )  # fmt: skip
        if dw_kernel_size_end:
            end_stride = stride if not dw_kernel_size_start and not dw_kernel_size_mid else 1
            dw_end = ConvNormAct(
                out_chs, out_chs, dw_kernel_size_end, end_stride,
                groups=_groups(group_size, out_chs), apply_act=False, rngs=rngs, **common,
            )  # fmt: skip
        self.dw_end = dw_end
        self.layer_scale = (
            LayerScale2d(out_chs, layer_scale_init_value)
            if layer_scale_init_value is not None
            else None
        )
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        if self.dw_start is not None:
            x = self.dw_start(x)
        x = self.pw_exp(x)
        if self.dw_mid is not None:
            x = self.dw_mid(x)
        if self.se is not None:
            x = self.se(x)
        x = self.pw_proj(x)
        if self.dw_end is not None:
            x = self.dw_end(x)
        if self.layer_scale is not None:
            x = self.layer_scale(x)
        return self._residual(x, shortcut)


class MultiQueryAttention2d(nnx.Module):
    """timm ``MultiQueryAttention2d``: many query heads share one key and one value head;
    keys and values may be computed on a depthwise-strided map (``kv_stride``)."""

    def __init__(self, dim, dim_out, num_heads, key_dim, value_dim, kv_stride, dw_kernel_size,
                 dilation, pad_type, norm_layer, *, rngs):  # fmt: skip
        self.num_heads, self.key_dim, self.value_dim = num_heads, key_dim, value_dim
        self.query_proj = nnx.Conv(dim, num_heads * key_dim, (1, 1), use_bias=False, rngs=rngs)
        self.kv_stride = kv_stride
        for name, out in (("key", key_dim), ("value", value_dim)):
            if kv_stride > 1:
                setattr(self, f"{name}_down_conv", _conv(
                    dim, dim, dw_kernel_size, kv_stride, dilation, dim, pad_type, rngs=rngs
                ))  # fmt: skip
                setattr(self, f"{name}_norm", _norm_act(norm_layer, dim, None, False, rngs=rngs))
            setattr(self, f"{name}_proj", nnx.Conv(dim, out, (1, 1), use_bias=False, rngs=rngs))
        self.output_proj = nnx.Conv(
            num_heads * value_dim, dim_out, (1, 1), use_bias=False, rngs=rngs
        )

    def _kv(self, name, x):
        if self.kv_stride > 1:
            x = getattr(self, f"{name}_norm")(getattr(self, f"{name}_down_conv")(x))
        x = getattr(self, f"{name}_proj")(x)
        return x.reshape(x.shape[0], -1, 1, x.shape[-1])

    def __call__(self, x):
        from ..attention import dot_product_attention

        B, H, W, _ = x.shape
        q = self.query_proj(x).reshape(B, H * W, self.num_heads, self.key_dim)
        k, v = self._kv("key", x), self._kv("value", x)
        k = jnp.broadcast_to(k, (B, k.shape[1], self.num_heads, self.key_dim))
        v = jnp.broadcast_to(v, (B, v.shape[1], self.num_heads, self.value_dim))
        o = dot_product_attention(q, k, v).reshape(B, H, W, -1)
        return self.output_proj(o)


class MobileAttention(_Block):
    def __init__(self, in_chs, out_chs, stride=1, dw_kernel_size=3, dilation=1, group_size=1,
                 pad_type="", num_heads=8, key_dim=64, value_dim=64, use_multi_query=False,
                 query_strides=(1, 1), kv_stride=1, cpe_dw_kernel_size=3, noskip=False, act=None,
                 norm_layer=None, aa_layer=None, layer_scale_init_value=1e-5, use_bias=False,
                 use_cpe=False, drop_path=0.0, *, rngs):  # fmt: skip
        assert use_multi_query and not use_cpe and not use_bias
        assert all(
            s == 1
            for s in (
                query_strides if isinstance(query_strides, (list, tuple)) else (query_strides,)
            )
        )
        self.has_skip = stride == 1 and in_chs == out_chs and not noskip
        self.norm = _norm_act(norm_layer, in_chs, act, False, rngs=rngs)
        self.attn = MultiQueryAttention2d(
            in_chs, out_chs, num_heads, key_dim, value_dim, kv_stride, dw_kernel_size, dilation,
            pad_type, norm_layer, rngs=rngs,
        )  # fmt: skip
        self.layer_scale = (
            LayerScale2d(out_chs, layer_scale_init_value)
            if layer_scale_init_value is not None
            else None
        )
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        x = self.attn(self.norm(x))
        if self.layer_scale is not None:
            x = self.layer_scale(x)
        return self._residual(x, shortcut)


_BLOCKS = {
    "DepthwiseSeparableConv": DepthwiseSeparableConv,
    "InvertedResidual": InvertedResidual,
    "CondConvResidual": InvertedResidual,
    "EdgeResidual": EdgeResidual,
    "ConvBnAct": ConvBnAct,
    "UniversalInvertedResidual": UniversalInvertedResidual,
    "MobileAttention": MobileAttention,
}
_MODEL_LEVEL = ("pad_type", "norm_layer", "act_layer", "aa_layer", "layer_scale_init_value")


def _arg_names(cls):
    import inspect

    return set(inspect.signature(cls.__init__).parameters)


_ARGS = {name: _arg_names(cls) for name, cls in _BLOCKS.items()}


def _build_stages(stem_chs, stages, model, drop_path_rate, *, rngs):
    """Replays the recorded blocks; omitted arguments take the model-level settings (which
    timm's builder passes to every block) or the block class defaults."""
    total = sum(n for stage in stages for _, _, n in stage)
    out, in_chs, idx = [], stem_chs, 0
    for stage in stages:
        blocks = []
        for cls, kw, repeat in stage:
            args = {**{k: model[k] for k in _MODEL_LEVEL}, **kw}
            act = _ACTS[args.pop("act_layer")]
            args = {k: v for k, v in args.items() if k in _ARGS[cls]}
            for _ in range(repeat):
                rate = drop_path_rate * idx / total
                blocks.append(_BLOCKS[cls](in_chs, act=act, drop_path=rate, rngs=rngs, **args))
                in_chs, idx = kw["out_chs"], idx + 1
        out.append(nnx.List(blocks))
    return nnx.List(out), in_chs


class EfficientNet(ClassifierMixin, nnx.Module):
    """timm ``EfficientNet`` / ``MobileNetV3`` rebuilt from a recorded config.

    EfficientNet heads apply a 1x1 conv + norm-act before pooling; MobileNetV3 heads pool
    first, then apply a 1x1 conv (with bias, or a norm) and the activation.
    """

    def __init__(
        self, model, stages, kind="EfficientNet", num_classes=1000, in_chans=3,
        global_pool="avg", drop_rate=0.0, drop_path_rate=0.0, *, rngs,
    ):  # fmt: skip
        model = {**MODEL_DEFAULTS, **model}
        self.num_classes, self.global_pool, self.kind = num_classes, global_pool, kind
        act = _ACTS[model["act_layer"]]
        norm, pad = model["norm_layer"], model["pad_type"]
        stem_chs, stem_k = model["stem"]
        self.conv_stem = _conv(in_chans, stem_chs, stem_k, 2, pad_type=pad, rngs=rngs)
        self.bn1 = _norm_act(norm, stem_chs, act, rngs=rngs)
        self.blocks, chs = _build_stages(stem_chs, stages, model, drop_path_rate, rngs=rngs)
        head = model["head"]
        self.act = act
        conv_head = bn2 = norm_head = None
        if kind == "EfficientNet":
            if head:
                conv_head = _conv(chs, head, 1, pad_type=pad, rngs=rngs)
                bn2 = _norm_act(norm, head, act, rngs=rngs)
            self.num_features = self.head_hidden_size = head or chs
        else:
            # jimm's num_features is the pooled output width (timm's head_hidden_size).
            self.num_features = self.head_hidden_size = head or chs
            if head:
                use_norm = model.get("head_norm", False)
                bias = model.get("head_bias", True) and not use_norm
                conv_head = nnx.Conv(chs, head, (1, 1), use_bias=bias, rngs=rngs)
                norm_head = _norm_act(norm, head, act, rngs=rngs) if use_norm else None
        self.conv_head, self.bn2, self.norm_head = conv_head, bn2, norm_head
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.classifier = (
            nnx.Linear(self.head_hidden_size, num_classes, rngs=rngs) if num_classes > 0 else None
        )

    _classifier_attr = "classifier"

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.classifier = (
            nnx.Linear(self.head_hidden_size, num_classes, rngs=nnx.Rngs(0))
            if num_classes > 0
            else None
        )

    def forward_intermediates(self, x, out_indices=None):
        """Feature maps after the stem and after each stage."""
        from ..features import _select_features

        feats = [self.bn1(self.conv_stem(x))]
        x = feats[0]
        for stage in self.blocks:
            for blk in stage:
                x = blk(x)
            feats.append(x)
        return _select_features(feats, out_indices)

    def forward_features(self, x):
        x = self.bn1(self.conv_stem(x))
        for stage in self.blocks:
            for blk in stage:
                x = blk(x)
        if self.kind == "EfficientNet" and self.conv_head is not None:
            x = self.bn2(self.conv_head(x))
        return x

    def forward_head(self, x, pre_logits=False):
        if self.global_pool == "avg":
            x = jnp.mean(x, axis=(1, 2), keepdims=self.kind != "EfficientNet")
        elif self.global_pool == "max":
            x = jnp.max(x, axis=(1, 2), keepdims=self.kind != "EfficientNet")
        if self.kind != "EfficientNet":
            if self.conv_head is not None:
                x = self.conv_head(x)
                x = self.norm_head(x) if self.norm_head is not None else self.act(x)
            x = x.reshape(x.shape[0], -1) if self.global_pool else x
        x = self.head_drop(x)
        if pre_logits or self.classifier is None:
            return x
        return self.classifier(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


MobileNetV3 = EfficientNet


def _make(name):
    module, cls, model, stages, ev = EFF_CFGS[name]

    def entry(**kwargs):
        net = EfficientNet(model, stages, cls, **kwargs)
        net.default_cfg = _cfg(**ev)
        return net

    entry.__name__ = name
    entry.__module__ = f"jimm.models.{module}"
    return entry


def register_module(module):
    """Registers every recorded model of a timm module (efficientnet, mobilenetv3, ...)."""
    for name, (mod, *_rest) in EFF_CFGS.items():
        if mod == module:
            register_model(_make(name))
