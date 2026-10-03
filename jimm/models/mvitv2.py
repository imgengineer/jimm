"""MViTv2 (Multiscale Vision Transformer v2) in flax nnx. Mirrors timm.models.mvitv2.

Pooling attention: per head, queries, keys, and values are pooled by
depthwise 3x3 convolutions (strided for queries at stage transitions and for
keys/values to keep their grid small) followed by LayerNorm; attention adds
decomposed relative position terms along height and width, and the pooled
queries are added back to the output (residual pooling). Block shortcuts are
max-pooled when queries are, and widened by a linear layer when the width
changes. The head averages the normalized tokens or reads the class token.
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..layers import ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


class Mlp(nnx.Module):
    def __init__(self, dim, hidden, out_dim, *, rngs):
        self.fc1 = nnx.Linear(dim, hidden, kernel_init=_init, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out_dim, kernel_init=_init, rngs=rngs)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))


def _rel_index(q_size, k_size):
    q_ratio, k_ratio = max(k_size / q_size, 1.0), max(q_size / k_size, 1.0)
    dist = np.arange(q_size)[:, None] * q_ratio - np.arange(k_size)[None, :] * k_ratio
    return (dist + (k_size - 1) * k_ratio).astype(np.int64)


class MultiScaleAttention(nnx.Module):
    def __init__(
        self,
        dim,
        dim_out,
        feat_size,
        num_heads=8,
        qkv_bias=True,
        stride_q=1,
        stride_kv=1,
        has_cls_token=True,
        residual_pooling=True,
        eps=1e-6,
        *,
        rngs,
    ):
        self.num_heads, self.dim_out = num_heads, dim_out
        self.head_dim = dim_out // num_heads
        self.has_cls_token, self.residual_pooling = has_cls_token, residual_pooling
        self.qkv = nnx.Linear(dim, 3 * dim_out, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim_out, dim_out, kernel_init=_init, rngs=rngs)
        hd = self.head_dim

        def pool(stride):
            # Depthwise over the head dimension, shared by all heads.
            return nnx.Conv(
                hd,
                hd,
                (3, 3),
                strides=(stride, stride),
                padding=((1, 1), (1, 1)),
                feature_group_count=hd,
                use_bias=False,
                rngs=rngs,
            )

        self.pool_q, self.pool_k, self.pool_v = pool(stride_q), pool(stride_kv), pool(stride_kv)
        self.norm_q = nnx.LayerNorm(hd, epsilon=eps, rngs=rngs)
        self.norm_k = nnx.LayerNorm(hd, epsilon=eps, rngs=rngs)
        self.norm_v = nnx.LayerNorm(hd, epsilon=eps, rngs=rngs)
        rel_dim = 2 * max(feat_size // stride_q, feat_size // stride_kv) - 1
        self.rel_pos_h = nnx.Param(_init(rngs.params(), (rel_dim, hd)))
        self.rel_pos_w = nnx.Param(_init(rngs.params(), (rel_dim, hd)))

    def _pool(self, x, feat_size, pool, norm):
        """(B, heads, N, hd) -> pooled and normalized tokens, new grid size."""
        B, h, _, hd = x.shape
        cls, x = (x[:, :, :1], x[:, :, 1:]) if self.has_cls_token else (None, x)
        x = pool(x.reshape(B * h, *feat_size, hd))
        size = x.shape[1:3]
        x = x.reshape(B, h, size[0] * size[1], hd)
        if cls is not None:
            x = jnp.concatenate([cls, x], axis=2)
        return norm(x), size

    def __call__(self, x, feat_size):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).transpose(2, 0, 3, 1, 4)
        q, q_size = self._pool(qkv[0], feat_size, self.pool_q, self.norm_q)
        k, k_size = self._pool(qkv[1], feat_size, self.pool_k, self.norm_k)
        v, _ = self._pool(qkv[2], feat_size, self.pool_v, self.norm_v)
        attn = (q * self.head_dim**-0.5) @ k.swapaxes(-2, -1)
        # Decomposed spatial relative positions, added to the patch-to-patch scores.
        sp = 1 if self.has_cls_token else 0
        (qh, qw), (kh, kw) = q_size, k_size
        rel_h = self.rel_pos_h[...][_rel_index(qh, kh)]
        rel_w = self.rel_pos_w[...][_rel_index(qw, kw)]
        r_q = q[:, :, sp:].reshape(B, self.num_heads, qh, qw, self.head_dim)
        rel = (
            jnp.einsum("byhwc,hkc->byhwk", r_q, rel_h)[..., :, None]
            + jnp.einsum("byhwc,wkc->byhwk", r_q, rel_w)[..., None, :]
        ).reshape(B, self.num_heads, qh * qw, kh * kw)
        attn = attn.at[:, :, sp:, sp:].add(rel)
        x = nnx.softmax(attn, axis=-1) @ v
        if self.residual_pooling:
            x = x + q
        x = x.swapaxes(1, 2).reshape(B, -1, self.dim_out)
        return self.proj(x), q_size


class MultiScaleBlock(nnx.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        feat_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        stride_q=1,
        stride_kv=1,
        has_cls_token=True,
        expand_attn=False,
        residual_pooling=True,
        eps=1e-6,
        *,
        rngs,
    ):
        proj_needed = dim != dim_out
        self.has_cls_token, self.stride_q = has_cls_token, stride_q
        att_dim = dim_out if expand_attn else dim
        self.norm1 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.shortcut_proj_attn = (
            nnx.Linear(dim, dim_out, kernel_init=_init, rngs=rngs)
            if proj_needed and expand_attn
            else None
        )
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            feat_size,
            num_heads,
            qkv_bias,
            stride_q,
            stride_kv,
            has_cls_token,
            residual_pooling,
            eps,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(att_dim, epsilon=eps, rngs=rngs)
        self.shortcut_proj_mlp = (
            nnx.Linear(dim, dim_out, kernel_init=_init, rngs=rngs)
            if proj_needed and not expand_attn
            else None
        )
        self.mlp = Mlp(att_dim, int(att_dim * mlp_ratio), dim_out, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def _shortcut_pool(self, x, feat_size):
        if self.stride_q == 1:
            return x
        cls, x = (x[:, :1], x[:, 1:]) if self.has_cls_token else (None, x)
        B, _, C = x.shape
        s = self.stride_q
        x = nnx.max_pool(
            x.reshape(B, *feat_size, C), (s + 1, s + 1), strides=(s, s), padding=((1, 1), (1, 1))
        )
        x = x.reshape(B, -1, C)
        return x if cls is None else jnp.concatenate([cls, x], axis=1)

    def __call__(self, x, feat_size):
        x_norm = self.norm1(x)
        shortcut = x if self.shortcut_proj_attn is None else self.shortcut_proj_attn(x_norm)
        shortcut = self._shortcut_pool(shortcut, feat_size)
        y, feat_size = self.attn(x_norm, feat_size)
        x = shortcut + self.drop_path(y)
        x_norm = self.norm2(x)
        shortcut = x if self.shortcut_proj_mlp is None else self.shortcut_proj_mlp(x_norm)
        return shortcut + self.drop_path(self.mlp(x_norm)), feat_size


class MultiScaleVit(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        depths=(2, 3, 16, 3),
        embed_dim=96,
        num_heads=1,
        mlp_ratio=4.0,
        expand_attn=True,
        use_cls_token=False,
        img_size=224,
        num_classes=1000,
        in_chans=3,
        global_pool=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        num_stages = len(depths)
        dims = [embed_dim * 2**i for i in range(num_stages)]
        heads = [num_heads * 2**i for i in range(num_stages)]
        strides_q = [1] + [2] * (num_stages - 1)
        strides_kv, kv = [], 4
        for s in strides_q:
            kv = max(kv // s, 1)
            strides_kv.append(kv)
        self.num_classes = num_classes
        self.global_pool = (
            global_pool if global_pool is not None else ("token" if use_cls_token else "avg")
        )
        self.num_prefix_tokens = 1 if use_cls_token else 0
        self.patch_embed = nnx.Conv(
            in_chans, dims[0], (7, 7), strides=(4, 4), padding=((3, 3), (3, 3)), rngs=rngs
        )
        self.cls_token = nnx.Param(_init(rngs.params(), (1, 1, dims[0]))) if use_cls_token else None
        feat = img_size // 4
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, dim = [], dims[0]
        for i, depth in enumerate(depths):
            dim_out = dims[i] if expand_attn else dims[min(i + 1, num_stages - 1)]
            out_dims = [dim_out] * depth if expand_attn else [dim] * (depth - 1) + [dim_out]
            blocks = []
            for j in range(depth):
                stride_q = strides_q[i] if j == 0 else 1
                blocks.append(
                    MultiScaleBlock(
                        dim,
                        out_dims[j],
                        heads[i],
                        feat,
                        mlp_ratio,
                        drop_path=rates[sum(depths[:i]) + j],
                        stride_q=stride_q,
                        stride_kv=strides_kv[i],
                        has_cls_token=use_cls_token,
                        expand_attn=expand_attn,
                        rngs=rngs,
                    )
                )
                dim = out_dims[j]
                feat //= stride_q
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = dim
        self.norm = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(dim, num_classes, kernel_init=_init, rngs=rngs) if num_classes > 0 else None
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        if self.cls_token is not None:
            x = jnp.concatenate([jnp.broadcast_to(self.cls_token[...], (B, 1, C)), x], axis=1)
        feat_size = (H, W)
        for stage in self.stages:
            for blk in stage:
                x, feat_size = blk(x, feat_size)
        return self.norm(x)

    def forward_head(self, x):
        if self.global_pool:
            x = (
                jnp.mean(x[:, self.num_prefix_tokens :], axis=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "mvitv2_tiny": dict(depths=(1, 2, 5, 2)),
    "mvitv2_small": dict(depths=(1, 2, 11, 2)),
    "mvitv2_base": dict(depths=(2, 3, 16, 3)),
    "mvitv2_large": dict(depths=(2, 6, 36, 4), embed_dim=144, num_heads=2, expand_attn=False),
    "mvitv2_small_cls": dict(depths=(1, 2, 11, 2), use_cls_token=True),
    "mvitv2_base_cls": dict(depths=(2, 3, 16, 3), use_cls_token=True),
    "mvitv2_large_cls": dict(depths=(2, 6, 36, 4), embed_dim=144, num_heads=2, use_cls_token=True),
    "mvitv2_huge_cls": dict(depths=(4, 8, 60, 8), embed_dim=192, num_heads=3, use_cls_token=True),
}


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = MultiScaleVit(**{**cfg, **kwargs})
        model.default_cfg = _cfg(crop_pct=0.9, interpolation="bicubic", fixed_input_size=True)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
