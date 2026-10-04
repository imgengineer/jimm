"""CrossViT in flax nnx, NHWC. Mirrors timm.models.crossvit.

Two ViT branches see the image at different patch sizes: the small-patch
branch at full resolution, the large-patch branch on the image rescaled with
PyTorch's (non-antialiased, A=-0.75) bicubic interpolation. Each multi-scale
block runs the branches' transformer blocks, then fuses them through their
class tokens: each class token is projected to the other branch's width,
cross-attends over that branch's patch tokens and is projected back. The
logits average the two branches' class-token heads; "dagger" variants embed
patches with a three-conv stem.
"""

import math

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model
from .vision_transformer import Block

_trunc = nnx.initializers.truncated_normal(0.02)
_LINEAR = dict(kernel_init=_trunc, bias_init=nnx.initializers.zeros)


def _ln(dim, *, rngs):
    return nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)


def _cubic(t, a=-0.75):
    t = abs(t)
    if t <= 1:
        return ((a + 2) * t - (a + 3)) * t * t + 1
    if t < 2:
        return ((a * t - 5 * a) * t + 8 * a) * t - 4 * a
    return 0.0


def _bicubic_matrix(src, dst):
    """PyTorch ``interpolate(mode='bicubic', align_corners=False)`` along one axis."""
    m = np.zeros((dst, src), np.float32)
    scale = src / dst
    for i in range(dst):
        pos = (i + 0.5) * scale - 0.5
        base = math.floor(pos)
        t = pos - base
        for k in range(-1, 3):
            m[i, min(max(base + k, 0), src - 1)] += _cubic(k - t)
    return m


class PatchEmbed(nnx.Module):
    def __init__(self, patch_size, in_chans, embed_dim, multi_conv, *, rngs):
        if multi_conv:
            s2, p2 = (3, 0) if patch_size == 12 else (2, 1)
            s3, p3 = (1, 1) if patch_size == 12 else (2, 1)
            q, h = embed_dim // 4, embed_dim // 2
            self.proj = nnx.List(
                [
                    nnx.Conv(in_chans, q, (7, 7), strides=4, padding=((3, 3),) * 2, rngs=rngs),
                    nnx.Conv(q, h, (3, 3), strides=s2, padding=((p2, p2),) * 2, rngs=rngs),
                    nnx.Conv(h, embed_dim, (3, 3), strides=s3, padding=((p3, p3),) * 2, rngs=rngs),
                ]
            )
        else:
            p = (patch_size, patch_size)
            self.proj = nnx.Conv(in_chans, embed_dim, p, strides=p, padding="VALID", rngs=rngs)

    def __call__(self, x):
        if isinstance(self.proj, nnx.List):
            for i, conv in enumerate(self.proj):
                x = conv(x)
                if i < 2:
                    x = nnx.relu(x)
        else:
            x = self.proj(x)
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


class CrossAttention(nnx.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, *, rngs):
        self.num_heads = num_heads
        self.wq = nnx.Linear(dim, dim, use_bias=qkv_bias, **_LINEAR, rngs=rngs)
        self.wk = nnx.Linear(dim, dim, use_bias=qkv_bias, **_LINEAR, rngs=rngs)
        self.wv = nnx.Linear(dim, dim, use_bias=qkv_bias, **_LINEAR, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, **_LINEAR, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        h = self.num_heads
        q = self.wq(x[:, :1]).reshape(B, 1, h, C // h)
        k = self.wk(x).reshape(B, N, h, C // h)
        v = self.wv(x).reshape(B, N, h, C // h)
        return self.proj(dot_product_attention(q, k, v).reshape(B, 1, C))


class CrossAttentionBlock(nnx.Module):
    def __init__(self, dim, num_heads, qkv_bias, drop_path, *, rngs):
        self.norm1 = _ln(dim, rngs=rngs)
        self.attn = CrossAttention(dim, num_heads, qkv_bias, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        return x[:, :1] + self.drop_path(self.attn(self.norm1(x)))


class _Proj(nnx.Module):
    """LayerNorm, GELU and a linear map (timm's nn.Sequential indices 0 and 2)."""

    def __init__(self, din, dout, *, rngs):
        self.norm = _ln(din, rngs=rngs)
        self.fc = nnx.Linear(din, dout, **_LINEAR, rngs=rngs)

    def __call__(self, x):
        return self.fc(gelu(self.norm(x)))


class MultiScaleBlock(nnx.Module):
    def __init__(self, dims, depth, num_heads, mlp_ratio, qkv_bias, drop_path, *, rngs):
        n = len(dims)
        self.blocks = nnx.List(
            [
                nnx.List(
                    [
                        Block(
                            dims[d],
                            num_heads[d],
                            mlp_ratio[d],
                            qkv_bias,
                            0.0,
                            drop_path[i],
                            rngs=rngs,
                        )
                        for i in range(depth[d])
                    ]
                )
                for d in range(n)
            ]
        )
        self.projs = nnx.List([_Proj(dims[d], dims[(d + 1) % n], rngs=rngs) for d in range(n)])
        self.fusion = nnx.List(
            [
                CrossAttentionBlock(
                    dims[(d + 1) % n], num_heads[(d + 1) % n], qkv_bias, drop_path[-1], rngs=rngs
                )
                for d in range(n)
            ]
        )
        self.revert_projs = nnx.List(
            [_Proj(dims[(d + 1) % n], dims[d], rngs=rngs) for d in range(n)]
        )

    def __call__(self, xs):
        n = len(xs)
        outs_b = []
        for x, blocks in zip(xs, self.blocks):
            for blk in blocks:
                x = blk(x)
            outs_b.append(x)
        outs = []
        for i in range(n):
            cls = self.projs[i](outs_b[i][:, :1])
            tmp = self.fusion[i](jnp.concatenate([cls, outs_b[(i + 1) % n][:, 1:]], axis=1))
            cls = self.revert_projs[i](tmp[:, :1])
            outs.append(jnp.concatenate([cls, outs_b[i][:, 1:]], axis=1))
        return outs


class CrossViT(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = "token"

    def __init__(
        self,
        img_size=240,
        img_scale=(1.0, 224 / 240),
        patch_size=(12, 16),
        embed_dim=(192, 384),
        depth=((1, 4, 0),) * 3,
        num_heads=(6, 6),
        mlp_ratio=(4, 4, 1),
        multi_conv=False,
        qkv_bias=True,
        num_classes=1000,
        in_chans=3,
        global_pool="token",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        assert global_pool in ("token", "avg")
        self.num_classes, self.global_pool = num_classes, global_pool
        self.embed_dim = embed_dim
        self.num_features = sum(embed_dim)
        self.img_sizes = [int(img_size * s) for s in img_scale]
        self.patch_embed = nnx.List(
            [
                PatchEmbed(p, in_chans, d, multi_conv, rngs=rngs)
                for p, d in zip(patch_size, embed_dim)
            ]
        )
        for i, (s, p, d) in enumerate(zip(self.img_sizes, patch_size, embed_dim)):
            n = (s // p) ** 2
            setattr(self, f"pos_embed_{i}", nnx.Param(_trunc(rngs.params(), (1, 1 + n, d))))
            setattr(self, f"cls_token_{i}", nnx.Param(_trunc(rngs.params(), (1, 1, d))))
        total = sum(sum(c[-2:]) for c in depth)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        blocks, ptr = [], 0
        for cfg in depth:
            cur = max(cfg[:-1]) + cfg[-1]
            blocks.append(
                MultiScaleBlock(
                    embed_dim, cfg, num_heads, mlp_ratio, qkv_bias, dpr[ptr : ptr + cur], rngs=rngs
                )
            )
            ptr += cur
        self.blocks = nnx.List(blocks)
        self.norm = nnx.List([_ln(d, rngs=rngs) for d in embed_dim])
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.List([nnx.Linear(d, num_classes, **_LINEAR, rngs=rngs) for d in self.embed_dim])

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            assert global_pool in ("token", "avg")
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.head = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        B, H, W, _ = x.shape
        xs = []
        for i, embed in enumerate(self.patch_embed):
            s = self.img_sizes[i]
            xi = x
            if (H, W) != (s, s):
                mh = jnp.asarray(_bicubic_matrix(H, s), x.dtype)
                mw = jnp.asarray(_bicubic_matrix(W, s), x.dtype)
                xi = jnp.einsum("ih,jw,bhwc->bijc", mh, mw, x)
            t = embed(xi)
            cls = getattr(self, f"cls_token_{i}")[...]
            t = jnp.concatenate([jnp.broadcast_to(cls, (B, 1, t.shape[-1])), t], axis=1)
            xs.append(t + getattr(self, f"pos_embed_{i}")[...])
        for blk in self.blocks:
            xs = blk(xs)
        return [norm(t) for norm, t in zip(self.norm, xs)]

    def forward_head(self, xs):
        xs = [t[:, 1:].mean(axis=1) if self.global_pool == "avg" else t[:, 0] for t in xs]
        xs = [self.head_drop(t) for t in xs]
        if self.head is None:
            return jnp.concatenate(xs, axis=-1)
        return jnp.mean(jnp.stack([h(t) for h, t in zip(self.head, xs)]), axis=0)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _depth(n):
    return ((1, n, 0),) * 3


_CFGS = {  # embed_dim, depth, num_heads, mlp_ratio, multi_conv, input size
    "crossvit_tiny_240": ((96, 192), 4, (3, 3), (4, 4, 1), False, 240),
    "crossvit_small_240": ((192, 384), 4, (6, 6), (4, 4, 1), False, 240),
    "crossvit_base_240": ((384, 768), 4, (12, 12), (4, 4, 1), False, 240),
    "crossvit_9_240": ((128, 256), 3, (4, 4), (3, 3, 1), False, 240),
    "crossvit_15_240": ((192, 384), 5, (6, 6), (3, 3, 1), False, 240),
    "crossvit_18_240": ((224, 448), 6, (7, 7), (3, 3, 1), False, 240),
    "crossvit_9_dagger_240": ((128, 256), 3, (4, 4), (3, 3, 1), True, 240),
    "crossvit_15_dagger_240": ((192, 384), 5, (6, 6), (3, 3, 1), True, 240),
    "crossvit_15_dagger_408": ((192, 384), 5, (6, 6), (3, 3, 1), True, 408),
    "crossvit_18_dagger_240": ((224, 448), 6, (7, 7), (3, 3, 1), True, 240),
    "crossvit_18_dagger_408": ((224, 448), 6, (7, 7), (3, 3, 1), True, 408),
}


def _make(name):
    dims, depth, heads, ratios, multi_conv, size = _CFGS[name]
    small = 224 if size == 240 else 384

    def entry(**kwargs):
        model = CrossViT(
            **{
                "img_size": size,
                "img_scale": (1.0, small / size),
                "embed_dim": dims,
                "depth": _depth(depth),
                "num_heads": heads,
                "mlp_ratio": ratios,
                "multi_conv": multi_conv,
                **kwargs,
            }
        )
        model.default_cfg = _cfg(
            input_size=(3, size, size),
            crop_pct=0.875 if size == 240 else 1.0,
            interpolation="bicubic",
            fixed_input_size=True,
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
