"""MobileNetV5 (Gemma 3n vision tower) in flax nnx, NHWC. Mirrors timm.models.mobilenetv5.

A 3x3 stride-2 conv stem and four stages built from timm's EfficientNet block
strings: EdgeResidual blocks (fused 3x3 expansion), Universal Inverted
Residuals (optional depthwise start and middle convs around a 1x1 expansion,
with layer scale) and mobile multi-query attention (shared single-head keys
and values, optionally from a depthwise-strided map, with layer scale).
RMSNorm2d and tanh-approximate GELU are used throughout. The multi-scale
fusion adapter concatenates the last two stage outputs (nearest-upsampling
the smaller), runs a UIR feed-forward to 2048 channels, average-pools to its
output resolution if needed and applies RMSNorm; the classifier pools that
map. The ``_enc`` variant returns the fused map without a head.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, make_divisible
from ..registry import _cfg, register_model


def _gelu(x):
    return jax.nn.gelu(x, approximate=True)


class RmsNorm2d(nnx.Module):
    def __init__(self, chs, eps=1e-6):
        self.scale = nnx.Param(jnp.ones(chs))
        self.eps = eps

    def __call__(self, x):
        acc = jnp.promote_types(x.dtype, jnp.float32)
        var = jnp.mean(jnp.square(x.astype(acc)), axis=-1, keepdims=True)
        y = x * jax.lax.rsqrt(var + self.eps).astype(x.dtype)
        return y * self.scale[...].astype(x.dtype)


def _fan_out_normal(kernel, out_chs, groups):
    return nnx.initializers.normal(math.sqrt(2.0 / (kernel * kernel * out_chs // groups)))


def _conv(in_chs, out_chs, kernel=1, stride=1, groups=1, use_bias=False, same=False, *, rngs):
    if same:
        padding = "SAME"
    else:
        p = ((stride - 1) + (kernel - 1)) // 2
        padding = ((p, p), (p, p))
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=padding,
        feature_group_count=groups,
        use_bias=use_bias,
        kernel_init=_fan_out_normal(kernel, out_chs, groups),
        rngs=rngs,
    )


class ConvNormAct(nnx.Module):
    def __init__(
        self, in_chs, out_chs, kernel=1, stride=1, groups=1, act=True, bias=False, same=False,
        *, rngs,
    ):  # fmt: skip
        self.conv = _conv(in_chs, out_chs, kernel, stride, groups, bias, same, rngs=rngs)
        self.bn = RmsNorm2d(out_chs)
        self.act = act

    def __call__(self, x):
        x = self.bn(self.conv(x))
        return _gelu(x) if self.act else x


def _layer_scale(chs, value):
    return nnx.Param(jnp.full((chs,), value)) if value is not None else None


class EdgeResidual(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, exp_ratio, drop_path, same, *, rngs):
        mid = make_divisible(in_chs * exp_ratio)
        self.has_skip = in_chs == out_chs and stride == 1
        self.conv_exp = _conv(in_chs, mid, kernel, stride, same=same, rngs=rngs)
        self.bn1 = RmsNorm2d(mid)
        self.conv_pwl = _conv(mid, out_chs, same=same, rngs=rngs)
        self.bn2 = RmsNorm2d(out_chs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.bn2(self.conv_pwl(_gelu(self.bn1(self.conv_exp(x)))))
        return self.drop_path(y) + x if self.has_skip else y


class UniversalInvertedResidual(nnx.Module):
    def __init__(
        self, in_chs, out_chs, start_k, mid_k, stride, exp_ratio, drop_path, same,
        layer_scale=1e-5, noskip=False, *, rngs,
    ):  # fmt: skip
        self.has_skip = in_chs == out_chs and stride == 1 and not noskip
        self.dw_start = (
            ConvNormAct(
                in_chs,
                in_chs,
                start_k,
                1 if mid_k else stride,
                in_chs,
                False,
                same=same,
                rngs=rngs,
            )  # fmt: skip
            if start_k
            else None
        )
        mid = make_divisible(in_chs * exp_ratio)
        self.pw_exp = ConvNormAct(in_chs, mid, same=same, rngs=rngs)
        self.dw_mid = (
            ConvNormAct(mid, mid, mid_k, stride, mid, same=same, rngs=rngs) if mid_k else None
        )
        self.pw_proj = ConvNormAct(mid, out_chs, act=False, same=same, rngs=rngs)
        self.layer_scale = _layer_scale(out_chs, layer_scale)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        if self.dw_start is not None:
            x = self.dw_start(x)
        x = self.pw_exp(x)
        if self.dw_mid is not None:
            x = self.dw_mid(x)
        x = self.pw_proj(x)
        if self.layer_scale is not None:
            x = x * self.layer_scale[...]
        return self.drop_path(x) + shortcut if self.has_skip else x


class _KV(nnx.Module):
    def __init__(self, dim, out, kv_stride, kernel, same, *, rngs):
        if kv_stride > 1:
            self.down_conv = nnx.Conv(
                dim,
                dim,
                (kernel, kernel),
                strides=kv_stride,
                padding="SAME" if same else ((kernel // 2, kernel // 2),) * 2,
                feature_group_count=dim,
                use_bias=False,
                kernel_init=nnx.initializers.xavier_uniform(),
                rngs=rngs,
            )
            self.norm = RmsNorm2d(dim)
        else:
            self.down_conv = self.norm = None
        self.proj = nnx.Conv(
            dim, out, (1, 1), use_bias=False, kernel_init=nnx.initializers.xavier_uniform(),
            rngs=rngs,
        )  # fmt: skip

    def __call__(self, x):
        if self.down_conv is not None:
            x = self.norm(self.down_conv(x))
        return self.proj(x)


class _Proj(nnx.Module):
    def __init__(self, din, dout, *, rngs):
        self.proj = nnx.Conv(
            din, dout, (1, 1), use_bias=False, kernel_init=nnx.initializers.xavier_uniform(),
            rngs=rngs,
        )  # fmt: skip

    def __call__(self, x):
        return self.proj(x)


class MultiQueryAttention2d(nnx.Module):
    def __init__(
        self, dim, dim_out, num_heads, key_dim, value_dim, kv_stride, kernel, same, *, rngs
    ):
        self.num_heads, self.key_dim, self.value_dim = num_heads, key_dim, value_dim
        self.query = _Proj(dim, num_heads * key_dim, rngs=rngs)
        self.key = _KV(dim, key_dim, kv_stride, kernel, same, rngs=rngs)
        self.value = _KV(dim, value_dim, kv_stride, kernel, same, rngs=rngs)
        self.output = _Proj(value_dim * num_heads, dim_out, rngs=rngs)

    def __call__(self, x):
        B, H, W, _ = x.shape
        q = self.query(x).reshape(B, H * W, self.num_heads, self.key_dim)
        k = self.key(x).reshape(B, -1, 1, self.key_dim)
        v = self.value(x).reshape(B, -1, 1, self.value_dim)
        heads = (B, k.shape[1], self.num_heads)
        k = jnp.broadcast_to(k, (*heads, self.key_dim))
        v = jnp.broadcast_to(v, (*heads, self.value_dim))
        o = dot_product_attention(q, k, v).reshape(B, H, W, -1)
        return self.output(o)


class MobileAttention(nnx.Module):
    def __init__(
        self, chs, num_heads, key_dim, kv_stride, kernel, drop_path, same, layer_scale=1e-5,
        *, rngs,
    ):  # fmt: skip
        self.norm = RmsNorm2d(chs)
        self.attn = MultiQueryAttention2d(
            chs, chs, num_heads, key_dim, key_dim, kv_stride, kernel, same, rngs=rngs
        )
        self.layer_scale = _layer_scale(chs, layer_scale)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.attn(self.norm(x))
        if self.layer_scale is not None:
            y = y * self.layer_scale[...]
        return self.drop_path(y) + x


def _parse(block):
    kind, *opts = block.split("_")
    out = {"type": kind}
    for o in opts:
        key = o[0]
        out[key] = int(o[1:])
    return out


class MultiScaleFusionAdapter(nnx.Module):
    def __init__(self, in_chs, out_chs, output_resolution, same, *, rngs):
        self.output_resolution = output_resolution
        self.ffn = UniversalInvertedResidual(
            in_chs, out_chs, 0, 0, 1, 2.0, 0.0, same, layer_scale=None, noskip=True, rngs=rngs
        )
        self.norm = RmsNorm2d(out_chs)

    def __call__(self, inputs):
        B, H, W, _ = inputs[0].shape
        resized = []
        for img in inputs:
            h, w = img.shape[1:3]
            if h < H or w < W:
                # Nearest: output pixel i reads input floor(i * h / H).
                img = img[:, (jnp.arange(H) * h) // H][:, :, (jnp.arange(W) * w) // W]
            resized.append(img)
        x = self.ffn(jnp.concatenate(resized, axis=-1))
        oh = ow = self.output_resolution
        if (H, W) != (oh, ow):
            if H % oh or W % ow:
                x = jax.image.resize(x, (B, oh, ow, x.shape[-1]), "bilinear", antialias=False)
            else:
                sh, sw = H // oh, W // ow
                x = x.reshape(B, oh, sh, ow, sw, -1).mean(axis=(2, 4))
        return self.norm(x)


class MobileNetV5(ClassifierMixin, nnx.Module):
    _classifier_attr = "classifier"

    def __init__(
        self,
        arch,
        stem_size=64,
        num_features=2048,
        msfa_output_resolution=16,
        same_padding=False,
        encoder=False,
        layer_scale_init_value=1e-5,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.encoder = encoder
        self.num_classes = 0 if encoder else num_classes
        self.global_pool = "" if encoder else global_pool
        same, ls = same_padding, layer_scale_init_value
        self.conv_stem = ConvNormAct(in_chans, stem_size, 3, 2, bias=True, same=same, rngs=rngs)
        total = sum(len(s) for s in arch)
        stages, chs, idx, stage_chs = [], stem_size, 0, []
        for stage in arch:
            blocks = []
            for b, spec in enumerate(stage):
                a = _parse(spec)
                stride = a["s"] if b == 0 else 1
                dpr = drop_path_rate * idx / total
                out = a["c"]
                if a["type"] == "er":
                    blk = EdgeResidual(chs, out, a["k"], stride, a["e"], dpr, same, rngs=rngs)
                elif a["type"] == "uir":
                    blk = UniversalInvertedResidual(
                        chs, out, a.get("a", 0), a["k"], stride, a["e"], dpr, same, ls, rngs=rngs
                    )
                else:
                    blk = MobileAttention(
                        chs, a["h"], a["d"], a.get("v", 1), a["k"], dpr, same, ls, rngs=rngs
                    )
                blocks.append(blk)
                chs, idx = out, idx + 1
            stages.append(nnx.List(blocks))
            stage_chs.append(chs)
        self.blocks = nnx.List(stages)
        self.msfa = MultiScaleFusionAdapter(
            sum(stage_chs[-2:]), num_features, msfa_output_resolution, same, rngs=rngs
        )
        self.num_features = num_features
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.classifier = self._make_head(self.num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        r = 1.0 / math.sqrt(num_classes)

        def init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype, -r, r)

        return nnx.Linear(self.num_features, num_classes, kernel_init=init, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        assert not self.encoder, "the encoder variant has no classifier"
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.classifier = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.conv_stem(x)
        feats = []
        for stage in self.blocks:
            for blk in stage:
                x = blk(x)
            feats.append(x)
        return self.msfa(feats[-2:])

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x.mean(axis=(1, 2))
        x = self.head_drop(x)
        return self.classifier(x) if self.classifier is not None else x

    def __call__(self, x):
        x = self.forward_features(x)
        return x if self.encoder else self.forward_head(x)


_S1 = ["er_r1_k3_s2_e4_c128", "er_r1_k3_s1_e4_c128", "er_r1_k3_s1_e4_c128"]
_S2 = [
    "uir_r1_a3_k5_s2_e6_c256",
    "uir_r1_a5_k0_s1_e4_c256",
    "uir_r1_a3_k0_s1_e4_c256",
    "uir_r1_a5_k0_s1_e4_c256",
    "uir_r1_a3_k0_s1_e4_c256",
]
_ARCH_300M = [
    _S1,
    _S2,
    ["uir_r1_a5_k5_s2_e6_c640"]
    + ["uir_r1_a5_k0_s1_e4_c640"] * 7
    + ["uir_r1_a0_k0_s1_e1_c640"]
    + ["mqa_r1_k3_h12_v2_s1_d64_c640", "uir_r1_a0_k0_s1_e2_c640"] * 14,
    ["uir_r1_a5_k5_s2_e6_c1280"] + ["mqa_r1_k3_h16_s1_d96_c1280", "uir_r1_a0_k0_s1_e2_c1280"] * 19,
]
# timm's base stage-3 attention blocks say s2, but only a stage's first block may stride.
_ARCH_BASE = [
    _S1,
    _S2,
    ["uir_r1_a5_k5_s2_e6_c512"]
    + ["uir_r1_a5_k0_s1_e4_c512"] * 2
    + ["uir_r1_a0_k0_s1_e1_c512"]
    + ["mqa_r1_k3_h8_s2_d64_c512", "uir_r1_a0_k0_s1_e2_c512"] * 6,
    ["uir_r1_a5_k5_s2_e6_c1024"] + ["mqa_r1_k3_h16_s1_d64_c1024", "uir_r1_a0_k0_s1_e2_c1024"] * 7,
]
_INCEPTION = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
_ZERO = dict(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
_CFGS = {
    "mobilenetv5_300m_enc": (_ARCH_300M, dict(same_padding=True, encoder=True), 768, _ZERO),
    "mobilenetv5_300m": (_ARCH_300M, dict(), 768, _ZERO),
    "mobilenetv5_base": (_ARCH_BASE, dict(), 256, _INCEPTION),
}


def _make(name):
    arch, extra, size, norm = _CFGS[name]

    def entry(**kwargs):
        if extra.get("encoder"):
            for key in ("num_classes", "num_features", "global_pool"):
                kwargs.pop(key, None)
        model = MobileNetV5(arch, **{**extra, **kwargs})
        model.default_cfg = _cfg(
            input_size=(3, size, size), crop_pct=1.0, interpolation="bicubic", **norm
        )
        if model.encoder:
            model.default_cfg["num_classes"] = 0
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
