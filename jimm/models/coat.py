"""CoaT (Co-Scale Conv-Attentional Image Transformer) in flax nnx. Mirrors timm.models.coat.

Four stages of strided patch embedding (conv + LayerNorm) each prepend a class
token and run serial blocks: a shared depthwise-conv position encoding, then
factorized attention (``q @ (softmax_N(k)^T @ v)``) plus a convolutional
relative position term (depthwise convs over the values, gated by the
queries), then an MLP. CoaT (non-lite) adds parallel blocks that attend in
stages 2-4 and exchange information across scales by bilinear resampling;
their classifier mixes the three class tokens with a 1x1 Conv1d.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, Mlp
from ..registry import _cfg, register_model

_trunc = nnx.initializers.truncated_normal(0.02)
_LINEAR = dict(kernel_init=_trunc, bias_init=nnx.initializers.zeros)


def _linear(din, dout, use_bias=True, *, rngs):
    return nnx.Linear(din, dout, use_bias=use_bias, **_LINEAR, rngs=rngs)


def _norm(dim, *, rngs):
    return nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)


def _tokens_to_map(x, size):
    B, _, C = x.shape
    return x[:, 1:].reshape(B, *size, C)


def _with_cls(x, img):
    B, H, W, C = img.shape
    return jnp.concatenate([x[:, :1], img.reshape(B, H * W, C)], axis=1)


class PatchEmbed(nnx.Module):
    def __init__(self, in_chs, dim, patch_size, *, rngs):
        p = (patch_size, patch_size)
        self.proj = nnx.Conv(in_chs, dim, p, strides=p, padding="VALID", rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        x = self.proj(x)
        B, H, W, C = x.shape
        return self.norm(x.reshape(B, H * W, C)), (H, W)


class ConvPosEnc(nnx.Module):
    """Residual depthwise 3x3 conv over the image tokens; the class token passes through."""

    def __init__(self, dim, k=3, *, rngs):
        self.proj = nnx.Conv(dim, dim, (k, k), padding=k // 2, feature_group_count=dim, rngs=rngs)

    def __call__(self, x, size):
        feat = _tokens_to_map(x, size)
        return _with_cls(x, self.proj(feat) + feat)


class ConvRelPosEnc(nnx.Module):
    """Depthwise convs of several window sizes over groups of heads of ``v``, gated by ``q``."""

    def __init__(self, head_chs, num_heads, window, *, rngs):
        convs, self.channel_splits = [], []
        for k, heads in window.items():
            chs = heads * head_chs
            convs.append(
                nnx.Conv(chs, chs, (k, k), padding=k // 2, feature_group_count=chs, rngs=rngs)
            )
            self.channel_splits.append(chs)
        self.conv_list = nnx.List(convs)

    def __call__(self, q, v, size):
        B, N, h, Ch = q.shape  # tokens-first layout: [B, N, h, Ch]
        H, W = size
        v_img = v[:, 1:].reshape(B, H, W, h * Ch)
        splits, start = [], 0
        for conv, chs in zip(self.conv_list, self.channel_splits):
            splits.append(conv(v_img[..., start : start + chs]))
            start += chs
        conv_v = jnp.concatenate(splits, axis=-1).reshape(B, H * W, h, Ch)
        ev = q[:, 1:] * conv_v
        return jnp.pad(ev, ((0, 0), (1, 0), (0, 0), (0, 0)))


class FactorAttnConvRelPosEnc(nnx.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, proj_drop=0.0, *, rngs):
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = _linear(dim, dim * 3, qkv_bias, rngs=rngs)
        self.proj = _linear(dim, dim, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x, crpe, size):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        k = jax.nn.softmax(k, axis=1)
        factor = jnp.einsum("bnhc,bnhd->bhcd", k, v)
        factor = jnp.einsum("bnhc,bhcd->bnhd", q, factor)
        x = self.scale * factor + crpe(q, v, size)
        return self.proj_drop(self.proj(x.reshape(B, N, C)))


def _mlp(dim, ratio, drop, rngs):
    return Mlp(dim, int(dim * ratio), drop, **_LINEAR, rngs=rngs)


class SerialBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, proj_drop, drop_path, *, rngs):
        self.norm1 = _norm(dim, rngs=rngs)
        self.factoratt_crpe = FactorAttnConvRelPosEnc(
            dim, num_heads, qkv_bias, proj_drop, rngs=rngs
        )
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = _norm(dim, rngs=rngs)
        self.mlp = _mlp(dim, mlp_ratio, proj_drop, rngs)

    def __call__(self, x, cpe, crpe, size):
        x = cpe(x, size)
        x = x + self.drop_path(self.factoratt_crpe(self.norm1(x), crpe, size))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def _resample(x, size, factor):
    """Bilinear (half-pixel, no antialias) rescale of the image tokens, keeping the class token."""
    feat = _tokens_to_map(x, size)
    B, H, W, C = feat.shape
    out = (B, int(H * factor), int(W * factor), C)
    return _with_cls(x, jax.image.resize(feat, out, "bilinear", antialias=False))


class ParallelBlock(nnx.Module):
    """Attention in stages 2-4 with cross-scale bilinear exchange; one MLP shared by all three."""

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, proj_drop, drop_path, *, rngs):
        for i in (2, 3, 4):
            setattr(self, f"norm1{i}", _norm(dim, rngs=rngs))
            setattr(
                self,
                f"factoratt_crpe{i}",
                FactorAttnConvRelPosEnc(dim, num_heads, qkv_bias, proj_drop, rngs=rngs),
            )
            setattr(self, f"norm2{i}", _norm(dim, rngs=rngs))
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.mlp2 = _mlp(dim, mlp_ratio, proj_drop, rngs)

    def __call__(self, x2, x3, x4, crpes, sizes):
        S2, S3, S4 = sizes
        c2 = self.factoratt_crpe2(self.norm12(x2), crpes[0], S2)
        c3 = self.factoratt_crpe3(self.norm13(x3), crpes[1], S3)
        c4 = self.factoratt_crpe4(self.norm14(x4), crpes[2], S4)
        n2 = c2 + _resample(c3, S3, 2.0) + _resample(c4, S4, 4.0)
        n3 = c3 + _resample(c4, S4, 2.0) + _resample(c2, S2, 0.5)
        n4 = c4 + _resample(c3, S3, 0.5) + _resample(c2, S2, 0.25)
        x2 = x2 + self.drop_path(n2)
        x3 = x3 + self.drop_path(n3)
        x4 = x4 + self.drop_path(n4)
        x2 = x2 + self.drop_path(self.mlp2(self.norm22(x2)))
        x3 = x3 + self.drop_path(self.mlp2(self.norm23(x3)))
        x4 = x4 + self.drop_path(self.mlp2(self.norm24(x4)))
        return x2, x3, x4


class CoaT(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = "token"

    def __init__(
        self,
        patch_size=4,
        embed_dims=(64, 128, 320, 512),
        serial_depths=(3, 4, 6, 3),
        parallel_depth=0,
        num_heads=8,
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=True,
        num_classes=1000,
        in_chans=3,
        global_pool="token",
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        crpe_window=None,
        *,
        rngs,
    ):
        assert global_pool in ("token", "avg")
        self.num_classes, self.global_pool = num_classes, global_pool
        crpe_window = crpe_window or {3: 2, 5: 3, 7: 3}
        self.num_features = embed_dims[-1]
        prev = in_chans
        for i, dim in enumerate(embed_dims, 1):
            setattr(
                self,
                f"patch_embed{i}",
                PatchEmbed(prev, dim, patch_size if i == 1 else 2, rngs=rngs),
            )
            setattr(self, f"cls_token{i}", nnx.Param(_trunc(rngs.params(), (1, 1, dim))))
            setattr(self, f"cpe{i}", ConvPosEnc(dim, rngs=rngs))
            setattr(
                self,
                f"crpe{i}",
                ConvRelPosEnc(dim // num_heads, num_heads, crpe_window, rngs=rngs),
            )
            prev = dim
        # timm applies the same drop path rate to every block.
        args = (num_heads, qkv_bias, proj_drop_rate, drop_path_rate)
        for i, (dim, depth, ratio) in enumerate(zip(embed_dims, serial_depths, mlp_ratios), 1):
            blocks = [SerialBlock(dim, args[0], ratio, *args[1:], rngs=rngs) for _ in range(depth)]
            setattr(self, f"serial_blocks{i}", nnx.List(blocks))
        self.parallel_depth = parallel_depth
        if parallel_depth > 0:
            assert embed_dims[1] == embed_dims[2] == embed_dims[3]
            assert mlp_ratios[1] == mlp_ratios[2] == mlp_ratios[3]
            self.parallel_blocks = nnx.List(
                [
                    ParallelBlock(embed_dims[1], args[0], mlp_ratios[1], *args[1:], rngs=rngs)
                    for _ in range(parallel_depth)
                ]
            )
            self.norm2 = _norm(embed_dims[1], rngs=rngs)
            self.norm3 = _norm(embed_dims[2], rngs=rngs)
            self.aggregate = nnx.Conv(3, 1, (1,), rngs=rngs)
        else:
            self.parallel_blocks = self.norm2 = self.norm3 = self.aggregate = None
        self.norm4 = _norm(embed_dims[3], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        return _linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            assert global_pool in ("token", "avg")
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.head = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        """Final-stage tokens ``[B, 1 + HW, C]``, or a tuple of stage 2-4 tokens with parallel blocks."""
        xs, sizes = [], []
        for i in range(1, 5):
            if xs:
                x = _tokens_to_map(xs[-1], sizes[-1])
            t, size = getattr(self, f"patch_embed{i}")(x)
            cls = getattr(self, f"cls_token{i}")[...]
            t = jnp.concatenate([jnp.broadcast_to(cls, (t.shape[0], 1, t.shape[-1])), t], axis=1)
            cpe, crpe = getattr(self, f"cpe{i}"), getattr(self, f"crpe{i}")
            for blk in getattr(self, f"serial_blocks{i}"):
                t = blk(t, cpe, crpe, size)
            xs.append(t)
            sizes.append(size)
        if self.parallel_blocks is None:
            return self.norm4(xs[3])
        x2, x3, x4 = xs[1:]
        crpes = (self.crpe2, self.crpe3, self.crpe4)
        for blk in self.parallel_blocks:
            x2, x3, x4 = self.cpe2(x2, sizes[1]), self.cpe3(x3, sizes[2]), self.cpe4(x4, sizes[3])
            x2, x3, x4 = blk(x2, x3, x4, crpes, sizes[1:])
        return self.norm2(x2), self.norm3(x3), self.norm4(x4)

    def _pool(self, x):
        return x[:, 1:].mean(axis=1) if self.global_pool == "avg" else x[:, 0]

    def forward_head(self, x):
        if isinstance(x, tuple):
            x = jnp.stack([self._pool(t) for t in x], axis=-1)  # [B, C, 3]
            x = self.aggregate(x)[..., 0]
        else:
            x = self._pool(x)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "coat_tiny": dict(
        embed_dims=(152, 152, 152, 152), serial_depths=(2, 2, 2, 2), parallel_depth=6
    ),
    "coat_mini": dict(
        embed_dims=(152, 216, 216, 216), serial_depths=(2, 2, 2, 2), parallel_depth=6
    ),
    "coat_small": dict(
        embed_dims=(152, 320, 320, 320), serial_depths=(2, 2, 2, 2), parallel_depth=6
    ),
    "coat_lite_tiny": dict(
        embed_dims=(64, 128, 256, 320), serial_depths=(2, 2, 2, 2), mlp_ratios=(8, 8, 4, 4)
    ),
    "coat_lite_mini": dict(
        embed_dims=(64, 128, 320, 512), serial_depths=(2, 2, 2, 2), mlp_ratios=(8, 8, 4, 4)
    ),
    "coat_lite_small": dict(
        embed_dims=(64, 128, 320, 512), serial_depths=(3, 4, 6, 3), mlp_ratios=(8, 8, 4, 4)
    ),
    "coat_lite_medium": dict(embed_dims=(128, 256, 320, 512), serial_depths=(3, 6, 10, 8)),
    "coat_lite_medium_384": dict(embed_dims=(128, 256, 320, 512), serial_depths=(3, 6, 10, 8)),
}


def _make(name):
    cfg = _CFGS[name]
    extra = dict(input_size=(3, 384, 384), crop_pct=1.0, crop_mode="squash")
    extra = extra if name.endswith("_384") else dict(crop_pct=0.9)

    def entry(**kwargs):
        model = CoaT(**{**cfg, **kwargs})
        model.default_cfg = _cfg(interpolation="bicubic", fixed_input_size=True, **extra)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
