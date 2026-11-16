"""EfficientViT (MSRA) in flax nnx, NHWC. Mirrors timm.models.efficientvit_msra.

The M series uses cascaded group attention with 3x3 depthwise convolutions on
the queries, local windowed attention, and 1x1-conv MLPs. A conv stem and
subsample stages with patch merging form the backbone; a BatchNorm-Linear head
produces the logits.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, SqueezeExcite, global_pool_nhwc
from ..registry import _cfg, register_model


class ConvNorm(nnx.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel=1,
        stride=1,
        groups=1,
        bn_weight_init=1.0,
        *,
        rngs,
    ):
        pad = kernel // 2
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=((pad, pad), (pad, pad)),
            feature_group_count=groups,
            use_bias=False,
            rngs=rngs,
        )
        self.bn = BatchNorm(
            out_chs, scale_init=nnx.initializers.constant(bn_weight_init), rngs=rngs
        )

    def __call__(self, x):
        return self.bn(self.conv(x))


class NormLinear(nnx.Module):
    def __init__(self, in_features, out_features, drop=0.0, *, rngs):
        self.bn = BatchNorm(in_features, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        return self.linear(self.drop(self.bn(x)))


class PatchMerging(nnx.Module):
    def __init__(self, dim, out_dim, *, rngs):
        hid = int(dim * 4)
        self.conv1 = ConvNorm(dim, hid, rngs=rngs)
        self.conv2 = ConvNorm(hid, hid, 3, 2, groups=hid, rngs=rngs)
        self.se = SqueezeExcite(hid, rd_ratio=0.25, rngs=rngs)
        self.conv3 = ConvNorm(hid, out_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        return self.conv3(self.se(x))


class ConvMlp(nnx.Module):
    def __init__(self, dim, hidden, *, rngs):
        self.pw1 = ConvNorm(dim, hidden, rngs=rngs)
        self.pw2 = ConvNorm(hidden, dim, bn_weight_init=0.0, rngs=rngs)

    def __call__(self, x):
        return self.pw2(nnx.relu(self.pw1(x)))


def _bias_index(resolution):
    """Index of the absolute-offset relative position bias for every query/key pair."""
    r = resolution
    rows = np.arange(r * r) // r
    cols = np.arange(r * r) % r
    return np.abs(rows[:, None] - rows[None, :]) * r + np.abs(cols[:, None] - cols[None, :])


class CascadedGroupAttention(nnx.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=14,
        kernels=(5, 5, 5, 5),
        *,
        rngs,
    ):
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.resolution = resolution
        self.qkvs = nnx.List(
            [
                ConvNorm(dim // num_heads, key_dim * 2 + self.val_dim, rngs=rngs)
                for _ in range(num_heads)
            ]
        )
        self.dws = nnx.List(
            [
                ConvNorm(key_dim, key_dim, kernels[i], groups=key_dim, rngs=rngs)
                for i in range(num_heads)
            ]
        )
        self.proj = ConvNorm(self.val_dim * num_heads, dim, bn_weight_init=0.0, rngs=rngs)
        n = resolution * resolution
        self.attention_biases = nnx.Param(jnp.zeros((num_heads, n), jnp.float32))
        self.attention_bias_idxs = _bias_index(resolution)

    def __call__(self, x):
        B, H, W, _ = x.shape
        chunks = jnp.split(x, self.num_heads, axis=-1)
        feats_out = []
        feat = chunks[0]
        for i, (qkv, dw) in enumerate(zip(self.qkvs, self.dws)):
            if i > 0:
                # Cascade the previous head's value output into this head's input.
                feat = feat + chunks[i]
            y = qkv(feat)
            q, k, v = jnp.split(y, [self.key_dim, self.key_dim * 2], axis=-1)
            q = dw(q).reshape(B, H * W, self.key_dim)
            q = q * self.scale
            k = k.reshape(B, H * W, self.key_dim)
            v = v.reshape(B, H * W, self.val_dim)
            attn = jnp.einsum("bid,bjd->bij", q, k)
            bias = jnp.take(self.attention_biases[...][i], self.attention_bias_idxs)
            attn = jax.nn.softmax(attn + bias[None], axis=-1)
            feat = jnp.einsum("bij,bjc->bic", attn, v).reshape(B, H, W, self.val_dim)
            feats_out.append(feat)
        return self.proj(jnp.concatenate(feats_out, axis=-1))


class LocalWindowAttention(nnx.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=14,
        window_resolution=7,
        kernels=(5, 5, 5, 5),
        *,
        rngs,
    ):
        self.resolution = resolution
        assert window_resolution > 0
        self.window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(
            dim,
            key_dim,
            num_heads,
            attn_ratio=attn_ratio,
            resolution=self.window_resolution,
            kernels=kernels,
            rngs=rngs,
        )

    def __call__(self, x):
        resolution = self.resolution
        B, H, W, C = x.shape
        assert H == W == resolution, f"expected {(resolution, resolution)}, got {(H, W)}"
        ws = self.window_resolution
        if ws >= H and ws >= W:
            return self.attn(x)
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
        pH, pW = H + pad_b, W + pad_r
        nH, nW = pH // ws, pW // ws
        x = x.reshape(B, nH, ws, nW, ws, C).transpose(0, 1, 3, 2, 4, 5)
        x = self.attn(x.reshape(B * nH * nW, ws, ws, C))
        x = x.reshape(B, nH, nW, ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, pH, pW, C)
        return x[:, :H, :W]


class ResidualDrop(nnx.Module):
    def __init__(self, module):
        self.m = module

    def __call__(self, x):
        return x + self.m(x)


class EfficientVitBlock(nnx.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=14,
        window_resolution=7,
        kernels=(5, 5, 5, 5),
        *,
        rngs,
    ):
        self.dw0 = ResidualDrop(ConvNorm(dim, dim, 3, groups=dim, bn_weight_init=0.0, rngs=rngs))
        self.ffn0 = ResidualDrop(ConvMlp(dim, int(dim * 2), rngs=rngs))
        self.mixer = ResidualDrop(
            LocalWindowAttention(
                dim,
                key_dim,
                num_heads,
                attn_ratio=attn_ratio,
                resolution=resolution,
                window_resolution=window_resolution,
                kernels=kernels,
                rngs=rngs,
            )
        )
        self.dw1 = ResidualDrop(ConvNorm(dim, dim, 3, groups=dim, bn_weight_init=0.0, rngs=rngs))
        self.ffn1 = ResidualDrop(ConvMlp(dim, int(dim * 2), rngs=rngs))

    def __call__(self, x):
        x = self.dw0(x)
        x = self.ffn0(x)
        x = self.mixer(x)
        x = self.dw1(x)
        return self.ffn1(x)


class EfficientVitStage(nnx.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        key_dim,
        downsample=("", 1),
        num_heads=8,
        attn_ratio=4,
        resolution=14,
        window_resolution=7,
        kernels=(5, 5, 5, 5),
        depth=1,
        *,
        rngs,
    ):
        if downsample[0] == "subsample":
            self.resolution = (resolution - 1) // downsample[1] + 1
            self.res1_dw = ResidualDrop(ConvNorm(in_dim, in_dim, 3, groups=in_dim, rngs=rngs))
            self.res1_ffn = ResidualDrop(ConvMlp(in_dim, int(in_dim * 2), rngs=rngs))
            self.patchmerge = PatchMerging(in_dim, out_dim, rngs=rngs)
            self.res2_dw = ResidualDrop(ConvNorm(out_dim, out_dim, 3, groups=out_dim, rngs=rngs))
            self.res2_ffn = ResidualDrop(ConvMlp(out_dim, int(out_dim * 2), rngs=rngs))
            self.downsample = True
        else:
            assert in_dim == out_dim
            self.resolution = resolution
            self.downsample = False
        self.blocks = nnx.List(
            [
                EfficientVitBlock(
                    out_dim,
                    key_dim,
                    num_heads,
                    attn_ratio,
                    self.resolution,
                    window_resolution,
                    kernels,
                    rngs=rngs,
                )
                for _ in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample:
            x = self.res1_ffn(self.res1_dw(x))
            x = self.patchmerge(x)
            x = self.res2_ffn(self.res2_dw(x))
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbedding(nnx.Module):
    def __init__(self, in_chans, dim, *, rngs):
        self.conv1 = ConvNorm(in_chans, dim // 8, 3, 2, rngs=rngs)
        self.conv2 = ConvNorm(dim // 8, dim // 4, 3, 2, rngs=rngs)
        self.conv3 = ConvNorm(dim // 4, dim // 2, 3, 2, rngs=rngs)
        self.conv4 = ConvNorm(dim // 2, dim, 3, 2, rngs=rngs)
        self.patch_size = 16

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        return self.conv4(x)


class EfficientVitMsra(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=(64, 128, 192),
        key_dim=(16, 16, 16),
        depth=(1, 2, 3),
        num_heads=(4, 4, 4),
        window_size=(7, 7, 7),
        kernels=(5, 5, 5, 5),
        down_ops=(("", 1), ("subsample", 2), ("subsample", 2)),
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.drop_rate = drop_rate
        self.patch_embed = PatchEmbedding(in_chans, embed_dim[0], rngs=rngs)
        resolution = img_size // self.patch_embed.patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        stages, pre_ed = [], embed_dim[0]
        for ed, kd, dpth, nh, ar, wd, do in zip(
            embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops
        ):
            stage = EfficientVitStage(
                pre_ed,
                ed,
                kd,
                downsample=do,
                num_heads=nh,
                attn_ratio=ar,
                resolution=resolution,
                window_resolution=wd,
                kernels=kernels,
                depth=dpth,
                rngs=rngs,
            )
            pre_ed = ed
            resolution = stage.resolution
            stages.append(stage)
        self.stages = nnx.List(stages)
        self.num_features = embed_dim[-1]
        if num_classes > 0:
            self.head = NormLinear(self.num_features, num_classes, drop=drop_rate, rngs=rngs)
        else:
            self.head = None

    def get_classifier(self):
        return self.head.linear if self.head is not None else None

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.head = (
            NormLinear(self.num_features, num_classes, drop=self.drop_rate, rngs=nnx.Rngs(0))
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x, pre_logits=False):
        x = global_pool_nhwc(x, self.global_pool)
        return x if pre_logits or self.head is None else self.head(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "efficientvit_m0": ((64, 128, 192), (1, 2, 3), (4, 4, 4), (5, 5, 5, 5)),
    "efficientvit_m1": ((128, 144, 192), (1, 2, 3), (2, 3, 3), (7, 5, 3, 3)),
    "efficientvit_m2": ((128, 192, 224), (1, 2, 3), (4, 3, 2), (7, 5, 3, 3)),
    "efficientvit_m3": ((128, 240, 320), (1, 2, 3), (4, 3, 4), (5, 5, 5, 5)),
    "efficientvit_m4": ((128, 256, 384), (1, 2, 3), (4, 4, 4), (7, 5, 3, 3)),
    "efficientvit_m5": ((192, 288, 384), (1, 3, 4), (3, 3, 4), (7, 5, 3, 3)),
}


def _make(name):
    embed_dim, depth, num_heads, kernels = _CFGS[name]

    def entry(**kwargs):
        model = EfficientVitMsra(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            kernels=kernels,
            **kwargs,
        )
        model.default_cfg = _cfg(crop_pct=0.875, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
