"""Segment Anything ViT image encoder in flax nnx, NHWC. Mirrors timm.models.vision_transformer_sam.

Patch tokens stay on their 2D grid with an absolute position embedding
(resized with antialiased bicubic interpolation for other input sizes).
Blocks attend within (zero-padded) windows, except for a few global blocks,
and add SAM's decomposed relative position bias: the unscaled query is dotted
with per-axis relative embeddings for height and width. A conv + LayerNorm2d
neck reduces the width (or a LayerNorm2d alone for the 224 px classifier),
followed by average pooling and an optional linear head.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..features import _select_features
from ..layers import ClassifierMixin, DropPath, Mlp
from ..registry import _cfg, register_model

_trunc = nnx.initializers.truncated_normal(0.02)
_LINEAR = dict(kernel_init=_trunc, bias_init=nnx.initializers.zeros)


def _get_rel_pos(q_size, k_size, rel_pos):
    """timm's ``get_rel_pos``: resize the table if needed and gather (q, k) relative offsets."""
    max_dist = 2 * max(q_size, k_size) - 1
    if rel_pos.shape[0] != max_dist:
        out = (max_dist, rel_pos.shape[1])
        rel_pos = jax.image.resize(rel_pos, out, "linear", antialias=False)
    q = np.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k = np.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    idx = (q - k + (k_size - 1) * max(q_size / k_size, 1.0)).astype(np.int64)
    return rel_pos[idx]


class Attention(nnx.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, use_rel_pos=False, input_size=None, *, rngs):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, **_LINEAR, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, **_LINEAR, rngs=rngs)
        if use_rel_pos:
            self.rel_pos_h = nnx.Param(jnp.zeros((2 * input_size[0] - 1, head_dim)))
            self.rel_pos_w = nnx.Param(jnp.zeros((2 * input_size[1] - 1, head_dim)))
        else:
            self.rel_pos_h = self.rel_pos_w = None

    def __call__(self, x):
        B, H, W, C = x.shape
        h = self.num_heads
        qkv = self.qkv(x.reshape(B, H * W, C)).reshape(B, H * W, 3, h, C // h)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        bias = None
        if self.rel_pos_h is not None:
            rh = _get_rel_pos(H, H, self.rel_pos_h[...]).astype(q.dtype)
            rw = _get_rel_pos(W, W, self.rel_pos_w[...]).astype(q.dtype)
            r_q = q.reshape(B, H, W, h, -1)
            rel_h = jnp.einsum("bxyhc,xkc->bhxyk", r_q, rh)
            rel_w = jnp.einsum("bxyhc,ykc->bhxyk", r_q, rw)
            bias = (rel_h[..., :, None] + rel_w[..., None, :]).reshape(B, h, H * W, H * W)
        x = dot_product_attention(q, k, v, bias=bias)
        return self.proj(x.reshape(B, H * W, C)).reshape(B, H, W, C)


class Block(nnx.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio, qkv_bias, drop_path, use_rel_pos, window_size,
        input_size, *, rngs,
    ):  # fmt: skip
        self.window_size = window_size
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        size = input_size if window_size == 0 else (window_size, window_size)
        self.attn = Attention(dim, num_heads, qkv_bias, use_rel_pos, size, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), **_LINEAR, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        y = self.norm1(x)
        ws = self.window_size
        if ws > 0:
            ph, pw = -H % ws, -W % ws
            y = jnp.pad(y, ((0, 0), (0, ph), (0, pw), (0, 0)))
            Hp, Wp = H + ph, W + pw
            y = y.reshape(B, Hp // ws, ws, Wp // ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
            y = self.attn(y.reshape(-1, ws, ws, C))
            y = y.reshape(B, Hp // ws, Wp // ws, ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
            y = y.reshape(B, Hp, Wp, C)[:, :H, :W]
        else:
            y = self.attn(y)
        x = x + self.drop_path(y)
        return x + self.drop_path(self.mlp(self.norm2(x)))


def _ln2d(dim, *, rngs):
    """LayerNorm2d: in NHWC, a LayerNorm over channels."""
    return nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)


class PatchEmbed(nnx.Module):
    def __init__(self, patch_size, in_chans, embed_dim, *, rngs):
        p = (patch_size, patch_size)
        self.proj = nnx.Conv(in_chans, embed_dim, p, strides=p, padding="VALID", rngs=rngs)

    def __call__(self, x):
        return self.proj(x)


class VisionTransformerSAM(ClassifierMixin, nnx.Module):
    _classifier_attr = "fc"

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=False,
        window_size=14,
        global_attn_indexes=(),
        neck_chans=256,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim, rngs=rngs)
        grid = (img_size // patch_size, img_size // patch_size)
        self.pos_embed = nnx.Param(jnp.zeros((1, *grid, embed_dim))) if use_abs_pos else None
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    dpr[i],
                    use_rel_pos,
                    0 if i in global_attn_indexes else window_size,
                    grid,
                    rngs=rngs,
                )  # fmt: skip
                for i in range(depth)
            ]
        )
        if neck_chans:
            self.neck = nnx.List(
                [
                    nnx.Conv(embed_dim, neck_chans, (1, 1), use_bias=False, rngs=rngs),
                    _ln2d(neck_chans, rngs=rngs),
                    nnx.Conv(
                        neck_chans,
                        neck_chans,
                        (3, 3),
                        padding=((1, 1), (1, 1)),
                        use_bias=False,
                        rngs=rngs,
                    ),  # fmt: skip
                    _ln2d(neck_chans, rngs=rngs),
                ]
            )
            self.num_features = neck_chans
        else:
            self.neck = nnx.List([_ln2d(embed_dim, rngs=rngs)])
            self.num_features = embed_dim
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = self._make_fc(num_classes, rngs)

    def _make_fc(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, **_LINEAR, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.fc = self._make_fc(num_classes, nnx.Rngs(0))

    def _forward_features(self, x, intermediates=None):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pe = self.pos_embed[...]
            if pe.shape[1:3] != x.shape[1:3]:
                shape = (1, *x.shape[1:3], pe.shape[-1])
                pe = jax.image.resize(pe.astype(jnp.float32), shape, "cubic", antialias=True)
            x = x + pe.astype(x.dtype)
        for blk in self.blocks:
            x = blk(x)
            if intermediates is not None:
                intermediates.append(x)
        for layer in self.neck:
            x = layer(x)
        return x

    def forward_features(self, x):
        return self._forward_features(x)

    def forward_intermediates(self, x, out_indices=None):
        """NHWC block outputs, followed by the neck output."""
        features = []
        features.append(self._forward_features(x, features))
        return _select_features(features, out_indices)

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x.mean(axis=(1, 2))
        elif self.global_pool == "max":
            x = x.max(axis=(1, 2))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "samvit_base_patch16": (768, 12, 12, (2, 5, 8, 11), 1024),
    "samvit_large_patch16": (1024, 24, 16, (5, 11, 17, 23), 1024),
    "samvit_huge_patch16": (1280, 32, 16, (7, 15, 23, 31), 1024),
    "samvit_base_patch16_224": (768, 12, 12, (2, 5, 8, 11), 224),
}


def _make(name):
    dim, depth, heads, global_idx, size = _CFGS[name]
    small = size == 224
    extra = dict(use_abs_pos=False, neck_chans=None, num_classes=1000) if small else {}

    def entry(**kwargs):
        model = VisionTransformerSAM(
            **{
                "img_size": size,
                "embed_dim": dim,
                "depth": depth,
                "num_heads": heads,
                "global_attn_indexes": global_idx,
                "window_size": 14,
                "use_rel_pos": True,
                **extra,
                **kwargs,
            }
        )
        model.default_cfg = _cfg(
            input_size=(3, size, size),
            crop_pct=0.9 if small else 1.0,
            interpolation="bicubic",
            fixed_input_size=True,
            num_classes=1000 if small else 0,
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
