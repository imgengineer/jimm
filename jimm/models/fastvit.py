"""FastViT in flax nnx, NHWC. Mirrors timm.models.fastvit (train-time, un-reparameterized form).

A three-block MobileOne stem (each block sums a BN'd kxk conv, a BN'd 1x1
"scale" conv and a BN identity when shapes allow) reduces the image 4x.
Stages downsample with a large-kernel (7x7 + 3x3) grouped conv embedding and
a 1x1 MobileOne block, optionally add a depthwise conditional position
encoding, then run RepMixer blocks (``x + ls * (mixer(x) - norm(x))``) or
attention blocks, each followed by a depthwise-7x7 + 1x1 conv MLP with layer
scale. A grouped MobileOne conv with squeeze-excite doubles the width before
pooling and the classifier.
"""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import BatchNorm, ClassifierMixin, DropPath, gelu, global_pool_nhwc, make_divisible
from ..registry import _cfg, register_model

_trunc = nnx.initializers.truncated_normal(0.02)


def _conv(in_chs, out_chs, kernel, stride=1, groups=1, use_bias=True, kernel_init=None, *, rngs):
    pad = ((stride - 1) + (kernel - 1)) // 2
    extra = {} if kernel_init is None else {"kernel_init": kernel_init}
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        feature_group_count=groups,
        use_bias=use_bias,
        **extra,
        rngs=rngs,
    )


def _bn(chs, *, rngs):
    return BatchNorm(chs, epsilon=1e-5, momentum=0.9, rngs=rngs)


class ConvNorm(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride=1, groups=1, *, rngs):
        self.conv = _conv(in_chs, out_chs, kernel, stride, groups, use_bias=False, rngs=rngs)
        self.bn = _bn(out_chs, rngs=rngs)

    def __call__(self, x):
        return self.bn(self.conv(x))


class SqueezeExcite(nnx.Module):
    def __init__(self, chs, rd_ratio=1.0 / 16, rd_divisor=8, *, rngs):
        rd = make_divisible(chs * rd_ratio, rd_divisor, round_limit=0.0)
        self.fc1 = nnx.Conv(chs, rd, (1, 1), rngs=rngs)
        self.fc2 = nnx.Conv(rd, chs, (1, 1), rngs=rngs)

    def __call__(self, x):
        s = self.fc2(nnx.relu(self.fc1(x.mean(axis=(1, 2), keepdims=True))))
        return x * nnx.sigmoid(s)


def _groups(group_size, chs):
    return chs // group_size if group_size else 1


class MobileOneBlock(nnx.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        group_size=0,
        use_se=False,
        use_act=True,
        use_scale_branch=True,
        num_conv_branches=1,
        *,
        rngs,
    ):
        groups = _groups(group_size, in_chs)
        self.se = SqueezeExcite(out_chs, rd_divisor=1, rngs=rngs) if use_se else None
        self.identity = _bn(in_chs, rngs=rngs) if out_chs == in_chs and stride == 1 else None
        self.conv_kxk = (
            nnx.List(
                [
                    ConvNorm(in_chs, out_chs, kernel_size, stride, groups, rngs=rngs)
                    for _ in range(num_conv_branches)
                ]
            )
            if num_conv_branches > 0
            else None
        )
        self.conv_scale = (
            ConvNorm(in_chs, out_chs, 1, stride, groups, rngs=rngs)
            if kernel_size > 1 and use_scale_branch
            else None
        )
        self.use_act = use_act

    def __call__(self, x):
        out = 0.0
        if self.conv_scale is not None:
            out = out + self.conv_scale(x)
        if self.identity is not None:
            out = out + self.identity(x)
        if self.conv_kxk is not None:
            for branch in self.conv_kxk:
                out = out + branch(x)
        if self.se is not None:
            out = self.se(out)
        return gelu(out) if self.use_act else out


class ReparamLargeKernelConv(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride, group_size, use_se, use_act, *, rngs):
        groups = _groups(group_size, in_chs)
        self.large_conv = ConvNorm(in_chs, out_chs, kernel_size, stride, groups, rngs=rngs)
        self.small_conv = ConvNorm(in_chs, out_chs, 3, stride, groups, rngs=rngs)
        self.se = SqueezeExcite(out_chs, rd_ratio=0.25, rngs=rngs) if use_se else None
        self.use_act = use_act

    def __call__(self, x):
        out = self.large_conv(x) + self.small_conv(x)
        if self.se is not None:
            out = self.se(out)
        return gelu(out) if self.use_act else out


class PatchEmbed(nnx.Module):
    def __init__(self, patch_size, stride, in_chs, embed_dim, lkc_use_act, use_se, *, rngs):
        self.proj = nnx.List(
            [
                ReparamLargeKernelConv(
                    in_chs, embed_dim, patch_size, stride, 1, use_se, lkc_use_act, rngs=rngs
                ),
                MobileOneBlock(embed_dim, embed_dim, 1, rngs=rngs),
            ]
        )

    def __call__(self, x):
        for layer in self.proj:
            x = layer(x)
        return x


class RepMixer(nnx.Module):
    def __init__(self, dim, kernel_size=3, layer_scale_init_value=1e-5, *, rngs):
        self.norm = MobileOneBlock(
            dim,
            dim,
            kernel_size,
            group_size=1,
            use_act=False,
            use_scale_branch=False,
            num_conv_branches=0,
            rngs=rngs,
        )
        self.mixer = MobileOneBlock(dim, dim, kernel_size, group_size=1, use_act=False, rngs=rngs)
        self.layer_scale = (
            nnx.Param(jnp.full((dim,), layer_scale_init_value))
            if layer_scale_init_value is not None
            else None
        )

    def __call__(self, x):
        y = self.mixer(x) - self.norm(x)
        return x + (y * self.layer_scale[...] if self.layer_scale is not None else y)


class ConvMlp(nnx.Module):
    def __init__(self, in_chs, hidden, drop=0.0, *, rngs):
        self.conv = ConvNorm(in_chs, in_chs, 7, groups=in_chs, rngs=rngs)
        self.fc1 = _conv(in_chs, hidden, 1, kernel_init=_trunc, rngs=rngs)
        self.fc2 = _conv(hidden, in_chs, 1, kernel_init=_trunc, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        x = self.drop(gelu(self.fc1(self.conv(x))))
        return self.drop(self.fc2(x))


class RepConditionalPosEnc(nnx.Module):
    def __init__(self, dim, spatial_shape=7, *, rngs):
        self.pos_enc = _conv(dim, dim, spatial_shape, groups=dim, rngs=rngs)

    def __call__(self, x):
        return self.pos_enc(x) + x


def _layer_scale(dim, value):
    return nnx.Param(jnp.full((dim,), value)) if value is not None else None


def _scaled(x, gamma):
    return x * gamma[...] if gamma is not None else x


class RepMixerBlock(nnx.Module):
    def __init__(self, dim, kernel_size, mlp_ratio, drop, drop_path, ls_init, *, rngs):
        self.token_mixer = RepMixer(dim, kernel_size, ls_init, rngs=rngs)
        self.mlp = ConvMlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        self.layer_scale = _layer_scale(dim, ls_init)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = self.token_mixer(x)
        return x + self.drop_path(_scaled(self.mlp(x), self.layer_scale))


class Attention(nnx.Module):
    def __init__(self, dim, head_dim=32, *, rngs):
        self.num_heads = dim // head_dim
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=False, kernel_init=_trunc, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_trunc, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x.reshape(B, H * W, C)).reshape(B, H * W, 3, self.num_heads, -1)
        x = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        return self.proj(x.reshape(B, H, W, C))


class AttentionBlock(nnx.Module):
    def __init__(self, dim, mlp_ratio, layer_norm, drop, drop_path, ls_init, *, rngs):
        self.norm = (
            nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs) if layer_norm else _bn(dim, rngs=rngs)
        )
        self.token_mixer = Attention(dim, rngs=rngs)
        self.layer_scale_1 = _layer_scale(dim, ls_init)
        self.mlp = ConvMlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        self.layer_scale_2 = _layer_scale(dim, ls_init)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(_scaled(self.token_mixer(self.norm(x)), self.layer_scale_1))
        return x + self.drop_path(_scaled(self.mlp(x), self.layer_scale_2))


class FastVitStage(nnx.Module):
    def __init__(
        self,
        dim,
        dim_out,
        depth,
        token_mixer,
        downsample,
        se_downsample,
        pos_emb,
        kernel_size,
        mlp_ratio,
        layer_norm,
        drop,
        drop_path,
        ls_init,
        lkc_use_act,
        *,
        rngs,
    ):
        self.downsample = (
            PatchEmbed(7, 2, dim, dim_out, lkc_use_act, se_downsample, rngs=rngs)
            if downsample
            else None
        )
        self.pos_emb = RepConditionalPosEnc(dim_out, rngs=rngs) if pos_emb else None
        if token_mixer == "repmixer":
            blocks = [
                RepMixerBlock(dim_out, kernel_size, mlp_ratio, drop, d, ls_init, rngs=rngs)
                for d in drop_path
            ]
        else:
            blocks = [
                AttentionBlock(dim_out, mlp_ratio, layer_norm, drop, d, ls_init, rngs=rngs)
                for d in drop_path
            ]
        self.blocks = nnx.List(blocks)

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class FastVit(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        layers=(2, 2, 6, 2),
        token_mixers=("repmixer",) * 4,
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(4,) * 4,
        downsamples=(False, True, True, True),
        se_downsamples=(False,) * 4,
        pos_embs=(False,) * 4,
        repmixer_kernel_size=3,
        layer_norm=False,
        lkc_use_act=False,
        stem_use_scale_branch=True,
        cls_ratio=2.0,
        layer_scale_init_value=1e-5,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        d0, sb = embed_dims[0], stem_use_scale_branch
        self.stem = nnx.List(
            [
                MobileOneBlock(in_chans, d0, 3, 2, use_scale_branch=sb, rngs=rngs),
                MobileOneBlock(d0, d0, 3, 2, group_size=1, use_scale_branch=sb, rngs=rngs),
                MobileOneBlock(d0, d0, 1, use_scale_branch=sb, rngs=rngs),
            ]
        )
        total = sum(layers)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev, k = [], d0, 0
        for i, depth in enumerate(layers):
            stages.append(
                FastVitStage(
                    prev,
                    embed_dims[i],
                    depth,
                    token_mixers[i],
                    downsamples[i] or prev != embed_dims[i],
                    se_downsamples[i],
                    pos_embs[i],
                    repmixer_kernel_size,
                    mlp_ratios[i],
                    layer_norm,
                    proj_drop_rate,
                    dpr[k : k + depth],
                    layer_scale_init_value,
                    lkc_use_act,
                    rngs=rngs,
                )
            )
            prev, k = embed_dims[i], k + depth
        self.stages = nnx.List(stages)
        self.num_features = int(embed_dims[-1] * cls_ratio)
        self.final_conv = MobileOneBlock(
            prev, self.num_features, 3, group_size=1, use_se=True, rngs=rngs
        )
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = self._make_fc(num_classes, rngs)

    def _make_fc(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, kernel_init=_trunc, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.fc = self._make_fc(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        for blk in self.stem:
            x = blk(x)
        for stage in self.stages:
            x = stage(x)
        return self.final_conv(x)

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_RM = ("repmixer",) * 4
_SA = ("repmixer", "repmixer", "repmixer", "attention")
_PE = (False, False, False, True)
_MCI = dict(se_downsamples=(False, False, True, True), pos_embs=_PE, token_mixers=_SA)
_MCI_BIG = dict(
    mlp_ratios=(4,) * 5,
    se_downsamples=(False,) * 5,
    downsamples=(False, True, True, True, True),
    pos_embs=(False, False, False, True, True),
    token_mixers=("repmixer", "repmixer", "repmixer", "attention", "attention"),
    layer_norm=True,
    stem_use_scale_branch=False,
)
_D64 = (64, 128, 256, 512)
_CFGS = {
    "fastvit_t8": dict(layers=(2, 2, 4, 2), embed_dims=(48, 96, 192, 384), mlp_ratios=(3,) * 4),
    "fastvit_t12": dict(layers=(2, 2, 6, 2), embed_dims=_D64, mlp_ratios=(3,) * 4),
    "fastvit_s12": dict(layers=(2, 2, 6, 2), embed_dims=_D64),
    "fastvit_sa12": dict(layers=(2, 2, 6, 2), embed_dims=_D64, pos_embs=_PE, token_mixers=_SA),
    "fastvit_sa24": dict(layers=(4, 4, 12, 4), embed_dims=_D64, pos_embs=_PE, token_mixers=_SA),
    "fastvit_sa36": dict(layers=(6, 6, 18, 6), embed_dims=_D64, pos_embs=_PE, token_mixers=_SA),
    "fastvit_ma36": dict(
        layers=(6, 6, 18, 6), embed_dims=(76, 152, 304, 608), pos_embs=_PE, token_mixers=_SA
    ),
    "fastvit_mci0": dict(
        layers=(2, 6, 10, 2), embed_dims=_D64, mlp_ratios=(3,) * 4, lkc_use_act=True, **_MCI
    ),
    "fastvit_mci1": dict(
        layers=(4, 12, 20, 4), embed_dims=_D64, mlp_ratios=(3,) * 4, lkc_use_act=True, **_MCI
    ),
    "fastvit_mci2": dict(
        layers=(4, 12, 24, 4),
        embed_dims=(80, 160, 320, 640),
        mlp_ratios=(3,) * 4,
        lkc_use_act=True,
        **_MCI,
    ),
    "fastvit_mci3": dict(
        layers=(2, 12, 24, 4, 2), embed_dims=(96, 192, 384, 768, 1536), lkc_use_act=True, **_MCI_BIG
    ),
    "fastvit_mci4": dict(
        layers=(2, 12, 24, 4, 4),
        embed_dims=(128, 256, 512, 1024, 2048),
        lkc_use_act=True,
        **_MCI_BIG,
    ),
}
_CLIP = dict(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
_EXTRA = {
    "fastvit_ma36": dict(crop_pct=0.95),
    "fastvit_mci0": dict(crop_pct=0.95, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    "fastvit_mci1": dict(crop_pct=0.95, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    "fastvit_mci2": dict(crop_pct=0.95, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    "fastvit_mci3": dict(crop_pct=0.95, **_CLIP),
    "fastvit_mci4": dict(crop_pct=0.95, **_CLIP),
}


def _make(name):
    cfg = _CFGS[name]
    extra = {"crop_pct": 0.9, **_EXTRA.get(name, {})}

    def entry(**kwargs):
        model = FastVit(**{**cfg, **kwargs})
        model.default_cfg = _cfg(input_size=(3, 256, 256), interpolation="bicubic", **extra)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
