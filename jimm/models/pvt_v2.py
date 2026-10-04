"""PVT v2 in flax nnx, NHWC. Mirrors timm.models.pvt_v2 (overlap patch embed + linear SRA)."""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model


def _adaptive_avg_pool(x, out):
    """PyTorch ``AdaptiveAvgPool2d(out)`` on NHWC maps (bins floor(i*n/out)..ceil((i+1)*n/out))."""
    B, H, W, C = x.shape

    def bins(n):
        return [(i * n // out, -(-(i + 1) * n // out)) for i in range(out)]

    rows = [jnp.mean(x[:, a:b], axis=1) for a, b in bins(H)]
    x = jnp.stack(rows, axis=1)
    cols = [jnp.mean(x[:, :, a:b], axis=2) for a, b in bins(W)]
    return jnp.stack(cols, axis=2)


class LinearAttention(nnx.Module):
    """Spatial-reduction attention (PVT v2): keys and values come from a strided-conv
    reduction, or with ``linear_attn`` from a 7x7 adaptive average pool, a 1x1 conv,
    LayerNorm and GELU."""

    def __init__(self, dim, num_heads, sr_ratio, qkv_bias=True, linear_attn=False, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.q = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.kv = nnx.Linear(dim, dim * 2, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.linear_attn = linear_attn
        if linear_attn:
            self.sr = nnx.Conv(dim, dim, (1, 1), rngs=rngs)
            self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        elif sr_ratio > 1:
            self.sr = nnx.Conv(
                dim, dim, (sr_ratio, sr_ratio), strides=(sr_ratio, sr_ratio), padding="VALID",
                rngs=rngs,
            )  # fmt: skip
            self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        else:
            self.sr = None
            self.norm = None

    def __call__(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        if self.linear_attn:
            t = self.sr(_adaptive_avg_pool(x.reshape(B, H, W, C), 7))
            t = gelu(self.norm(t.reshape(B, -1, C)))
        elif self.sr is not None and self.norm is not None:
            t = self.sr(x.reshape(B, H, W, C))
            t = self.norm(t.reshape(B, -1, C))
        else:
            t = x
        kv = self.kv(t).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]
        out = dot_product_attention(q, k, v).reshape(B, N, C)
        return self.proj(out)


class PVTMlp(nnx.Module):
    """MLP with a 3x3 depthwise conv and GELU between fc1 and fc2 (PVT v2); the linear
    variant adds a ReLU before the depthwise conv."""

    def __init__(self, dim, hidden_dim, extra_relu=False, *, rngs):
        self.extra_relu = extra_relu
        self.fc1 = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.dw = nnx.Conv(
            hidden_dim, hidden_dim, (3, 3), feature_group_count=hidden_dim, rngs=rngs
        )
        self.fc2 = nnx.Linear(hidden_dim, dim, rngs=rngs)

    def __call__(self, x, H, W):
        B = x.shape[0]
        x = self.fc1(x)
        if self.extra_relu:
            x = nnx.relu(x)
        x = self.dw(x.reshape(B, H, W, -1)).reshape(B, H * W, -1)
        x = gelu(x)
        return self.fc2(x)


class PVTBlock(nnx.Module):
    def __init__(
        self, dim, num_heads, sr_ratio, mlp_ratio=4.0, drop_path=0.0, linear_attn=False, *, rngs
    ):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = LinearAttention(dim, num_heads, sr_ratio, linear_attn=linear_attn, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = PVTMlp(dim, hidden_dim, linear_attn, rngs=rngs)

    def __call__(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        return x + self.drop_path(self.mlp(self.norm2(x), H, W))


class OverlapPatchEmbed(nnx.Module):
    def __init__(self, patch, stride, in_chs, dim, *, rngs):
        self.proj = nnx.Conv(
            in_chs, dim, (patch, patch), strides=(stride, stride), padding=patch // 2, rngs=rngs
        )
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        x = self.proj(x)
        B, H, W, C = x.shape
        return self.norm(x.reshape(B, H * W, C)), H, W


class PyramidVisionTransformerV2(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        embed_dims=(64, 128, 320, 512),
        depths=(3, 4, 6, 3),
        num_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(8, 8, 4, 4),
        linear=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dims[-1]
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        patches, stages = [], []
        k = 0
        for i in range(4):
            patches.append(
                OverlapPatchEmbed(
                    7 if i == 0 else 3,
                    4 if i == 0 else 2,
                    in_chans if i == 0 else embed_dims[i - 1],
                    embed_dims[i],
                    rngs=rngs,
                )
            )
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    PVTBlock(
                        embed_dims[i],
                        num_heads[i],
                        sr_ratios[i],
                        mlp_ratios[i],
                        dpr[k],
                        linear_attn=linear,
                        rngs=rngs,
                    )
                )
                k += 1
            stages.append(nnx.List(blocks))
        self.patches = nnx.List(patches)
        self.stages = nnx.List(stages)
        # timm closes every stage with a LayerNorm; the last one is the final norm.
        self.norms = nnx.List([nnx.LayerNorm(dim, rngs=rngs) for dim in embed_dims])
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(embed_dims[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for patch, stage, norm in zip(self.patches, self.stages, self.norms):
            x, H, W = patch(x)
            for blk in stage:
                x = blk(x, H, W)
            x = norm(x).reshape(x.shape[0], H, W, -1)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {  # embed_dims, depths, num_heads, mlp_ratios
    "pvt_v2_b0": ((32, 64, 160, 256), (2, 2, 2, 2), (1, 2, 5, 8), (8, 8, 4, 4)),
    "pvt_v2_b1": ((64, 128, 320, 512), (2, 2, 2, 2), (1, 2, 5, 8), (8, 8, 4, 4)),
    "pvt_v2_b2": ((64, 128, 320, 512), (3, 4, 6, 3), (1, 2, 5, 8), (8, 8, 4, 4)),
    "pvt_v2_b3": ((64, 128, 320, 512), (3, 4, 18, 3), (1, 2, 5, 8), (8, 8, 4, 4)),
    "pvt_v2_b4": ((64, 128, 320, 512), (3, 8, 27, 3), (1, 2, 5, 8), (8, 8, 4, 4)),
    "pvt_v2_b5": ((64, 128, 320, 512), (3, 6, 40, 3), (1, 2, 5, 8), (4, 4, 4, 4)),
    "pvt_v2_b2_li": ((64, 128, 320, 512), (3, 4, 6, 3), (1, 2, 5, 8), (8, 8, 4, 4)),
}


def _make(name):
    embed_dims, depths, heads, mlps = _CFGS[name]

    def entry(**kwargs):
        model = PyramidVisionTransformerV2(
            embed_dims=embed_dims, depths=depths, num_heads=heads, mlp_ratios=mlps,
            **{"linear": name.endswith("_li"), **kwargs},
        )  # fmt: skip
        model.default_cfg = _cfg(crop_pct=0.9, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
