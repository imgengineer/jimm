"""VOLO (Vision Outlooker) in flax nnx, NHWC. Mirrors timm.models.volo.

A conv stem (7x7 stride 2 then two 3x3 convs, each with BatchNorm and ReLU)
and a stride-4 projection produce an 8x-downsampled map. The first stage
runs Outlookers: outlook attention predicts, from 2x2-average-pooled tokens,
a softmax over each 3x3 neighbourhood for every output position of that
neighbourhood, gathers the values at stride 2 and folds the weighted sums back
(overlaps add). A strided conv halves the map and adds a learned position
embedding; plain transformer stages follow, then two class-attention blocks
on a prepended class token. The class-token logits are combined with half the
token-wise maximum of an auxiliary head.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import BatchNorm, ClassifierMixin, DropPath, Mlp
from ..registry import _cfg, register_model

_trunc = nnx.initializers.truncated_normal(0.02)
_LINEAR = dict(kernel_init=_trunc, bias_init=nnx.initializers.zeros)


def _linear(din, dout, use_bias=True, *, rngs):
    return nnx.Linear(din, dout, use_bias=use_bias, **_LINEAR, rngs=rngs)


def _mlp(dim, ratio, *, rngs):
    return Mlp(dim, int(dim * ratio), **_LINEAR, rngs=rngs)


def _avg_pool_ceil(x, s):
    """``AvgPool2d(s, s, ceil_mode=True)``: partial windows average their in-bounds pixels."""
    B, H, W, C = x.shape
    ph, pw = -H % s, -W % s
    if not (ph or pw):
        return x.reshape(B, H // s, s, W // s, s, C).mean(axis=(2, 4))
    pad = ((0, 0), (0, ph), (0, pw), (0, 0))
    window = (1, s, s, 1)
    total = jax.lax.reduce_window(jnp.pad(x, pad), 0.0, jax.lax.add, window, window, "VALID")
    ones = jnp.pad(jnp.ones((1, H, W, 1), x.dtype), pad)
    count = jax.lax.reduce_window(ones, 0.0, jax.lax.add, window, window, "VALID")
    return total / count


class OutlookAttention(nnx.Module):
    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=2, qkv_bias=False, *, rngs):
        self.num_heads, self.k, self.padding, self.stride = num_heads, kernel_size, padding, stride
        self.scale = (dim // num_heads) ** -0.5
        self.v = _linear(dim, dim, qkv_bias, rngs=rngs)
        self.attn = _linear(dim, kernel_size**4 * num_heads, rngs=rngs)
        self.proj = _linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        k, p, s, nh = self.k, self.padding, self.stride, self.num_heads
        h, w = math.ceil(H / s), math.ceil(W / s)
        Hp, Wp = (h - 1) * s + k, (w - 1) * s + k
        v = jnp.pad(self.v(x), ((0, 0), (p, max(Hp - H - p, 0)), (p, max(Wp - W - p, 0)), (0, 0)))
        offsets = [(i, j) for i in range(k) for j in range(k)]
        # Unfold: [B, h*w, k*k, heads, Ch], one entry per neighbourhood position.
        patches = jnp.stack(
            [v[:, i : i + (h - 1) * s + 1 : s, j : j + (w - 1) * s + 1 : s] for i, j in offsets],
            axis=3,
        ).reshape(B, h * w, k * k, nh, C // nh)
        attn = self.attn(_avg_pool_ceil(x, s)).reshape(B, h * w, nh, k * k, k * k)
        attn = jax.nn.softmax(attn * self.scale, axis=-1)
        out = jnp.einsum("bnhij,bnjhc->bnihc", attn, patches).reshape(B, h, w, k * k, C)
        # Fold: overlapping contributions add up.
        folded = jnp.zeros((B, max(Hp, H + 2 * p), max(Wp, W + 2 * p), C), out.dtype)
        for idx, (i, j) in enumerate(offsets):
            folded = folded.at[:, i : i + (h - 1) * s + 1 : s, j : j + (w - 1) * s + 1 : s].add(
                out[:, :, :, idx]
            )
        return self.proj(folded[:, p : p + H, p : p + W])


class Outlooker(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.attn = OutlookAttention(dim, num_heads, qkv_bias=qkv_bias, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mlp = _mlp(dim, mlp_ratio, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Attention(nnx.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, *, rngs):
        self.num_heads = num_heads
        self.qkv = _linear(dim, dim * 3, qkv_bias, rngs=rngs)
        self.proj = _linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        x = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        return self.proj(x.reshape(B, H, W, C))


class Transformer(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.attn = Attention(dim, num_heads, qkv_bias, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mlp = _mlp(dim, mlp_ratio, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ClassAttention(nnx.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, *, rngs):
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.kv = _linear(dim, dim * 2, qkv_bias, rngs=rngs)
        self.q = _linear(dim, dim, qkv_bias, rngs=rngs)
        self.proj = _linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim)
        q = self.q(x[:, :1]).reshape(B, 1, self.num_heads, self.head_dim)
        cls = dot_product_attention(q, kv[:, :, 0], kv[:, :, 1]).reshape(B, 1, C)
        return self.proj(cls)


class ClassBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.attn = ClassAttention(dim, num_heads, qkv_bias, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mlp = _mlp(dim, mlp_ratio, rngs=rngs)

    def __call__(self, x):
        cls = x[:, :1] + self.attn(self.norm1(x))
        cls = cls + self.mlp(self.norm2(cls))
        return jnp.concatenate([cls, x[:, 1:]], axis=1)


class PatchEmbed(nnx.Module):
    def __init__(self, in_chans, hidden_dim, embed_dim, patch_size=8, *, rngs):
        def conv(cin, k, s):
            p = k // 2
            return nnx.Conv(
                cin, hidden_dim, (k, k), strides=s, padding=((p, p), (p, p)), use_bias=False,
                rngs=rngs,
            )  # fmt: skip

        def bn():
            return BatchNorm(hidden_dim, epsilon=1e-5, momentum=0.9, rngs=rngs)

        # timm's nn.Sequential interleaves ReLUs at 2, 5 and 8.
        self.conv = nnx.List(
            [conv(in_chans, 7, 2), bn(), conv(hidden_dim, 3, 1), bn(), conv(hidden_dim, 3, 1), bn()]
        )
        p = patch_size // 2
        self.proj = nnx.Conv(hidden_dim, embed_dim, (p, p), strides=p, padding="VALID", rngs=rngs)

    def __call__(self, x):
        for i in range(0, 6, 2):
            x = nnx.relu(self.conv[i + 1](self.conv[i](x)))
        return self.proj(x)


class Downsample(nnx.Module):
    def __init__(self, in_dim, out_dim, patch_size=2, *, rngs):
        p = (patch_size, patch_size)
        self.proj = nnx.Conv(in_dim, out_dim, p, strides=p, padding="VALID", rngs=rngs)

    def __call__(self, x):
        return self.proj(x)


class VOLO(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = "token"

    def __init__(
        self,
        layers,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        patch_size=8,
        stem_hidden_dim=64,
        embed_dims=(192, 384, 384, 384),
        num_heads=(6, 12, 12, 12),
        downsamples=(True, False, False, False),
        outlook_attention=(True, False, False, False),
        mlp_ratio=3.0,
        qkv_bias=False,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        drop_path_rate=0.0,
        post_layers=("ca", "ca"),
        use_aux_head=True,
        pooling_scale=2,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        mlp_ratio = (mlp_ratio,) * len(layers) if not isinstance(mlp_ratio, tuple) else mlp_ratio
        self.num_features = embed_dims[-1]
        self.patch_embed = PatchEmbed(
            in_chans, stem_hidden_dim, embed_dims[0], patch_size, rngs=rngs
        )
        grid = img_size // patch_size // pooling_scale
        self.pos_embed = nnx.Param(_trunc(rngs.params(), (1, grid, grid, embed_dims[-1])))
        self.pos_drop = nnx.Dropout(pos_drop_rate, rngs=rngs)
        network, total = [], sum(layers)
        for i, depth in enumerate(layers):
            dprs = [drop_path_rate * (j + sum(layers[:i])) / (total - 1) for j in range(depth)]
            if outlook_attention[i]:
                # timm never passes drop_path_rate to the outlooker stage.
                blocks = [
                    Outlooker(embed_dims[i], num_heads[i], mlp_ratio[i], qkv_bias, rngs=rngs)
                    for _ in range(depth)
                ]
            else:
                blocks = [
                    Transformer(embed_dims[i], num_heads[i], mlp_ratio[i], qkv_bias, d, rngs=rngs)
                    for d in dprs
                ]
            network.append(nnx.List(blocks))
            if downsamples[i]:
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], rngs=rngs))
        self.network = nnx.List(network)
        if post_layers is not None:
            self.post_network = nnx.List(
                [
                    ClassBlock(embed_dims[-1], num_heads[-1], mlp_ratio[-1], qkv_bias, rngs=rngs)
                    for _ in post_layers
                ]
            )
            self.cls_token = nnx.Param(_trunc(rngs.params(), (1, 1, embed_dims[-1])))
        else:
            self.post_network = self.cls_token = None
        self.use_aux_head = use_aux_head
        self.norm = nnx.LayerNorm(self.num_features, epsilon=1e-5, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self._make_heads(num_classes, rngs)

    def _make_heads(self, num_classes, rngs):
        self.head = _linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None
        self.aux_head = (
            _linear(self.num_features, num_classes, rngs=rngs)
            if num_classes > 0 and self.use_aux_head
            else None
        )

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self._make_heads(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.patch_embed(x)
        for idx, block in enumerate(self.network):
            if idx == 2:
                x = self.pos_drop(x + self.pos_embed[...])
            if isinstance(block, nnx.List):
                for blk in block:
                    x = blk(x)
            else:
                x = block(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        if self.post_network is not None:
            cls = jnp.broadcast_to(self.cls_token[...], (B, 1, C))
            x = jnp.concatenate([cls, x], axis=1)
            for blk in self.post_network:
                x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        if self.global_pool == "avg":
            out = x.mean(axis=1)
        elif self.global_pool == "token":
            out = x[:, 0]
        else:
            out = x
        # As in timm, dropout reaches only the auxiliary head's tokens.
        x = self.head_drop(x)
        if self.head is None:
            return out
        out = self.head(out)
        if self.aux_head is not None:
            out = out + 0.5 * self.aux_head(x[:, 1:]).max(axis=1)
        return out

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_ARCHS = {
    "d1": dict(layers=(4, 4, 8, 2), embed_dims=(192, 384, 384, 384), num_heads=(6, 12, 12, 12)),
    "d2": dict(layers=(6, 4, 10, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16)),
    "d3": dict(layers=(8, 8, 16, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16)),
    "d4": dict(layers=(8, 8, 16, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16)),
    "d5": dict(
        layers=(12, 12, 20, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        mlp_ratio=4.0,
        stem_hidden_dim=128,
    ),
}
_VARIANTS = {  # name: crop_pct
    "volo_d1_224": 0.96,
    "volo_d1_384": 1.0,
    "volo_d2_224": 0.96,
    "volo_d2_384": 1.0,
    "volo_d3_224": 0.96,
    "volo_d3_448": 1.0,
    "volo_d4_224": 0.96,
    "volo_d4_448": 1.15,
    "volo_d5_224": 0.96,
    "volo_d5_448": 1.15,
    "volo_d5_512": 1.15,
}


def _make(name):
    arch, size = _ARCHS[name.split("_")[1]], int(name.split("_")[2])
    crop_pct = _VARIANTS[name]

    def entry(**kwargs):
        model = VOLO(**{**arch, "img_size": size, **kwargs})
        model.default_cfg = _cfg(
            input_size=(3, size, size),
            crop_pct=crop_pct,
            interpolation="bicubic",
            fixed_input_size=True,
        )
        return model

    entry.__name__ = name
    return entry


for _name in _VARIANTS:
    register_model(_make(_name))
