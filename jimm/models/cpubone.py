"""CPUBone in flax nnx, NHWC. Mirrors timm.models.cpubone.

A stride-2 conv stem and two stages of fused MBConv blocks (grouped 3x3
expand conv + BN + hard-swish, then 1x1 projection + BN; residual unless
striding) feed two attention stages. Each attention block applies, after a
single-group GroupNorm, a convolutional attention (depthwise 2x2 strided
projection, 1x1 conv to per-head q/k/v, attention over the reduced map, a 1x1
conv and nearest upsampling), a GroupNorm + 1x1-conv GELU MLP, and a local
fused or plain MBConv with 2x2 kernels, all residual. The head is a 1x1 conv,
average pooling, a linear + LayerNorm + hard-swish projection and the
classifier.
"""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import BatchNorm, ClassifierMixin, DropPath, gelu, hswish
from ..registry import _cfg, register_model

_NORM_MODES = {"proj": (False, False, True), "depth_proj": (False, True, True), "all": (True,) * 3}


class ConvLayer(nnx.Module):
    """Conv with timm's 'same' padding: k // 2 for odd k; 2x2 pads (1, 0) at stride 1, none at 2."""

    def __init__(
        self, in_chs, out_chs, kernel=3, stride=1, groups=1, use_bias=False, norm=True, act=None,
        *, rngs,
    ):  # fmt: skip
        if kernel == 2:
            pad = (0, 0) if stride == 2 else (1, 0)
        else:
            pad = (kernel // 2, kernel // 2)
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=(pad, pad),
            feature_group_count=groups,
            use_bias=use_bias,
            rngs=rngs,
        )
        self.norm = BatchNorm(out_chs, epsilon=1e-5, momentum=0.9, rngs=rngs) if norm else None
        self.act = act

    def __call__(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x) if self.act is not None else x


class FusedMBConv(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, expand_ratio, groups, bias, *, rngs):
        mid = round(in_chs * expand_ratio)
        self.spatial_conv = ConvLayer(
            in_chs, mid, kernel, stride, groups, use_bias=bias, act=hswish, rngs=rngs
        )
        self.point_conv = ConvLayer(mid, out_chs, 1, rngs=rngs)

    def __call__(self, x):
        return self.point_conv(self.spatial_conv(x))


class MBConv(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, expand_ratio, groups, norm_mode, *, rngs):
        mid = round(in_chs * expand_ratio)
        n1, n2, n3 = _NORM_MODES[norm_mode]
        self.inverted_conv = ConvLayer(
            in_chs, mid, 1, groups=groups, use_bias=not n1, norm=n1, act=hswish, rngs=rngs
        )
        self.depth_conv = ConvLayer(
            mid, mid, kernel, groups=mid, use_bias=not n2, norm=n2, act=hswish, rngs=rngs
        )
        self.point_conv = ConvLayer(mid, out_chs, 1, use_bias=not n3, norm=n3, rngs=rngs)

    def __call__(self, x):
        return self.point_conv(self.depth_conv(self.inverted_conv(x)))


class ResidualBlock(nnx.Module):
    def __init__(self, main, residual=True, drop_path=0.0, *, rngs):
        self.main = main
        self.residual = residual
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        if not self.residual:
            return self.main(x)
        return self.drop_path(self.main(x)) + x


class ConvAttention(nnx.Module):
    """Attention over a depthwise-strided map with fused 1x1 output conv and nearest upsampling."""

    def __init__(self, dim, att_stride, *, rngs):
        self.num_heads = int(max(1, dim * 0.5 // 30))
        self.head_dim = int(dim // self.num_heads * 0.5)
        self.att_stride = att_stride
        inner = self.head_dim * self.num_heads
        self.conv_proj = ConvLayer(dim, dim, 2, att_stride, groups=dim, rngs=rngs)
        self.pwise = nnx.Conv(dim, inner * 3, (1, 1), use_bias=False, rngs=rngs)
        self.upsampling = nnx.Conv(inner, dim, (1, 1), rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        s = self.att_stride
        if s > 1 and (H % s or W % s):
            x = jnp.pad(x, ((0, 0), (0, -H % s), (0, -W % s), (0, 0)))
        y = self.pwise(self.conv_proj(x))
        _, h, w, _ = y.shape
        qkv = y.reshape(B, h * w, self.num_heads, 3, self.head_dim)
        out = dot_product_attention(qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :])
        out = self.upsampling(out.reshape(B, h, w, -1))
        if s > 1:
            out = jnp.repeat(jnp.repeat(out, s, axis=1), s, axis=2)
        return out[:, :H, :W]


class CPUBoneBlock(nnx.Module):
    def __init__(
        self, dim, expand_ratio, groups, att_stride, mlp_ratio, proj_drop, drop_path, norm_mode,
        *, rngs,
    ):  # fmt: skip
        self.attn_norm = nnx.GroupNorm(dim, num_groups=1, epsilon=1e-5, rngs=rngs)
        self.attn = ConvAttention(dim, att_stride, rngs=rngs)
        self.mlp_norm = nnx.GroupNorm(dim, num_groups=1, epsilon=1e-5, rngs=rngs)
        self.mlp_fc1 = nnx.Conv(dim, dim * mlp_ratio, (1, 1), rngs=rngs)
        self.mlp_fc2 = nnx.Conv(dim * mlp_ratio, dim, (1, 1), rngs=rngs)
        self.mlp_drop = nnx.Dropout(proj_drop, rngs=rngs)
        if dim < 256:
            self.local = FusedMBConv(dim, dim, 2, 1, expand_ratio, groups, True, rngs=rngs)
        else:
            self.local = MBConv(dim, dim, 2, expand_ratio, groups, norm_mode, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.attn_norm(x)))
        y = self.mlp_drop(self.mlp_fc2(gelu(self.mlp_fc1(self.mlp_norm(x)))))
        x = x + self.drop_path(y)
        return x + self.drop_path(self.local(x))


class CPUBone(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        width_list,
        depth_list,
        head_widths=(1536, 1600),
        expand_ratio=4,
        attn_mlp_ratio=4,
        stem_expand_ratio=2,
        downsample_expand_ratios=None,
        expand_groups=2,
        local_mbconv_norm="proj",
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        proj_drop_rate=0.1,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        assert global_pool in ("", "avg")
        self.num_classes, self.global_pool = num_classes, global_pool
        num_stages = len(width_list) - 1
        down_ratios = downsample_expand_ratios or (expand_ratio,) * num_stages
        total = sum(depth_list)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        g = expand_groups

        def local(cin, cout, stride, ratio):
            return FusedMBConv(cin, cout, 3, stride, ratio, g, False, rngs=rngs)

        w0 = width_list[0]
        stem = [ConvLayer(in_chans, w0, 3, 2, act=hswish, rngs=rngs)]
        for i in range(depth_list[0]):
            stem.append(ResidualBlock(local(w0, w0, 1, stem_expand_ratio), True, dpr[i], rngs=rngs))
        self.stem = nnx.List(stem)
        stages, cin, k = [], w0, depth_list[0]
        for s, (width, depth) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            sdpr = dpr[k : k + depth]
            k += depth
            if s >= 3:
                blocks = [ResidualBlock(local(cin, width, 2, down_ratios[s - 1]), False, rngs=rngs)]
                blocks += [
                    CPUBoneBlock(
                        width,
                        expand_ratio,
                        g,
                        2 if s == 3 else 1,
                        attn_mlp_ratio,
                        proj_drop_rate,
                        d,
                        local_mbconv_norm,
                        rngs=rngs,
                    )
                    for d in sdpr
                ]
            else:
                blocks = []
                for i, d in enumerate(sdpr):
                    stride = 2 if i == 0 else 1
                    ratio = down_ratios[s - 1] if stride == 2 else expand_ratio
                    blk = local(cin if i == 0 else width, width, stride, ratio)
                    blocks.append(ResidualBlock(blk, stride == 1, d, rngs=rngs))
            stages.append(nnx.List(blocks))
            cin = width
        self.stages = nnx.List(stages)
        self.head_conv = ConvLayer(cin, head_widths[0], 1, act=hswish, rngs=rngs)
        self.pre_linear = nnx.Linear(head_widths[0], head_widths[1], use_bias=False, rngs=rngs)
        self.pre_norm = nnx.LayerNorm(head_widths[1], epsilon=1e-5, rngs=rngs)
        self.num_features = head_widths[1]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = self._make_fc(num_classes, rngs)

    def _make_fc(self, num_classes, rngs):
        return nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            assert global_pool in ("", "avg")
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.fc = self._make_fc(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def forward_head(self, x):
        x = self.head_conv(x)
        if self.global_pool == "avg":
            x = x.mean(axis=(1, 2))
        x = hswish(self.pre_norm(self.pre_linear(x)))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_B1 = dict(
    width_list=(16, 32, 64, 128, 256), depth_list=(0, 1, 1, 5, 5), downsample_expand_ratios=(6,) * 4
)
_CFGS = {
    "cpubone_nano": dict(width_list=(12, 24, 48, 96, 192), depth_list=(0, 1, 1, 1, 2)),
    "cpubone_t0": dict(width_list=(12, 24, 48, 96, 192), depth_list=(0, 1, 1, 2, 3)),
    "cpubone_s0": dict(width_list=(14, 28, 56, 112, 224), depth_list=(0, 1, 1, 2, 3)),
    "cpubone_b0_bfrobust": dict(
        width_list=(16, 32, 64, 128, 256), depth_list=(0, 1, 1, 3, 4), local_mbconv_norm="all"
    ),
    "cpubone_b1_bfrobust": dict(_B1, local_mbconv_norm="all"),
    "cpubone_b1_dwnorm": dict(_B1, local_mbconv_norm="depth_proj"),
    "cpubone_b2_bfrobust": dict(
        width_list=(20, 40, 80, 160, 320),
        depth_list=(0, 1, 1, 6, 6),
        head_widths=(2304, 2560),
        downsample_expand_ratios=(6,) * 4,
        drop_path_rate=0.1,
        local_mbconv_norm="all",
    ),
    "cpubone_b2pt5_dwnorm": dict(
        width_list=(24, 48, 96, 192, 384),
        depth_list=(0, 1, 1, 6, 6),
        head_widths=(2304, 2560),
        downsample_expand_ratios=(6,) * 4,
        local_mbconv_norm="depth_proj",
    ),
    "cpubone_b3": dict(
        width_list=(32, 64, 128, 256, 512),
        depth_list=(1, 2, 3, 6, 6),
        stem_expand_ratio=4,
        downsample_expand_ratios=(6,) * 4,
    ),
}


def _make(name):
    cfg = _CFGS[name]
    size = 256 if name.endswith("_dwnorm") else 224

    def entry(**kwargs):
        model = CPUBone(**{**cfg, **kwargs})
        model.default_cfg = _cfg(input_size=(3, size, size), crop_pct=0.95, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
