"""DeepSeek-V4 / V4.1 vision towers in Flax NNX (NHWC).

Adapted for JAX from timm 1.0.30, Copyright 2026 Yonghye Kwon.
Reference: https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/deepseek_vit.py
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..features import _select_features
from ..layers import ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model
from ._rope_vit import DynamicPatchEmbed, RopeAttention, axial_rope


class RmsNormFp32(nnx.Module):
    def __init__(self, dim, eps=1e-6, affine=True, *, rngs):
        self.eps = eps
        self.scale = nnx.Param(jnp.ones(dim)) if affine else None

    def __call__(self, x):
        calc = x.astype(jnp.float32)
        calc = calc * jax.lax.rsqrt(jnp.mean(calc**2, axis=-1, keepdims=True) + self.eps)
        if self.scale is not None:
            calc = calc * self.scale[...].astype(jnp.float32)
        return calc.astype(x.dtype)


class DeepseekVitBlock(nnx.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio, norm_eps, proj_drop, attn_drop, drop_path, *, rngs
    ):
        self.norm1 = RmsNormFp32(dim, norm_eps, rngs=rngs)
        self.attn = RopeAttention(dim, num_heads, proj_drop, attn_drop, rngs=rngs)
        self.norm2 = RmsNormFp32(dim, norm_eps, rngs=rngs)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nnx.Linear(dim, 2 * hidden, use_bias=False, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, use_bias=False, rngs=rngs)
        self.mlp_drop = nnx.Dropout(proj_drop, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x, rope):
        x = x + self.drop_path(self.attn(self.norm1(x), rope))
        gate, up = jnp.split(self.fc1(self.norm2(x)), 2, axis=-1)
        y = self.mlp_drop(nnx.silu(gate) * up)
        return x + self.drop_path(self.mlp_drop(self.fc2(y)))


class DeepseekVitAligner(nnx.Module):
    """Channel-major pixel unshuffle, zero padding, and the native VLM projector."""

    def __init__(self, dim, downsample_ratio=3, out_features=0, *, rngs):
        if out_features <= 0 or downsample_ratio <= 0:
            raise ValueError("the aligner requires positive out_features and downsample_ratio")
        self.downsample_ratio = downsample_ratio
        self.out_features = out_features
        self.fc1 = nnx.Linear(dim * downsample_ratio**2, out_features, rngs=rngs)
        self.fc2 = nnx.Linear(out_features, out_features, rngs=rngs)

    def unshuffle(self, x):
        batch, height, width, dim = x.shape
        ratio = self.downsample_ratio
        x = jnp.pad(x, ((0, 0), (0, -height % ratio), (0, -width % ratio), (0, 0)))
        h, w = x.shape[1] // ratio, x.shape[2] // ratio
        x = x.reshape(batch, h, ratio, w, ratio, dim).transpose(0, 1, 3, 5, 2, 4)
        return x.reshape(batch, h * w, dim * ratio**2)

    def __call__(self, x, pre_logits=False):
        x = gelu(self.fc1(self.unshuffle(x)))
        return x if pre_logits else self.fc2(x)


class DeepseekVitEncoder(nnx.Module):
    def __init__(
        self,
        img_size=546,
        patch_size=14,
        in_chans=3,
        out_features=0,
        global_pool="align",
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=2816 / 1024,
        downsample_ratio=3,
        rope_temperature=10000.0,
        norm_eps=1e-6,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        dynamic_img_pad=False,
        *,
        rngs,
    ):
        if global_pool not in ("", "avg", "align"):
            raise ValueError("global_pool must be '', 'avg', or 'align'")
        self.num_classes, self.global_pool = 0, global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_heads, self.rope_temperature = num_heads, rope_temperature
        self.patch_embed = DynamicPatchEmbed(
            img_size, patch_size, in_chans, embed_dim, dynamic_img_pad, rngs=rngs
        )
        self.blocks = nnx.List(
            [
                DeepseekVitBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    norm_eps,
                    proj_drop_rate,
                    attn_drop_rate,
                    drop_path_rate * i / max(depth - 1, 1),
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        self.norm = RmsNormFp32(embed_dim, norm_eps, rngs=rngs)
        self.aligner = (
            DeepseekVitAligner(embed_dim, downsample_ratio, out_features, rngs=rngs)
            if global_pool == "align"
            else None
        )

    def _embed(self, x):
        x = self.patch_embed(x)
        batch, height, width, dim = x.shape
        rope = axial_rope(height, width, dim // self.num_heads, self.rope_temperature)
        return x.reshape(batch, height * width, dim), rope, (batch, height, width, dim)

    def forward_intermediates(self, x, out_indices=None):
        x, rope, shape = self._embed(x)
        features = []
        for block in self.blocks:
            x = block(x, rope)
            features.append(x.reshape(shape))
        return _select_features(features, out_indices)

    def forward_features(self, x):
        x, rope, shape = self._embed(x)
        for block in self.blocks:
            x = block(x, rope)
        return self.norm(x).reshape(shape)

    def forward_head(self, x):
        if self.aligner is not None:
            return self.aligner(x)
        return jnp.mean(x, axis=(1, 2)) if self.global_pool == "avg" else x

    def get_classifier(self):
        return None

    def reset_classifier(self, num_classes, global_pool=None):
        if num_classes:
            raise ValueError("use a DeepseekVitClassifier for classification")
        if global_pool is not None:
            if global_pool not in ("", "avg", "align"):
                raise ValueError("global_pool must be '', 'avg', or 'align'")
            if global_pool == "align" and self.aligner is None:
                raise ValueError("this encoder has no aligner")
            self.global_pool = global_pool
            if global_pool != "align":
                self.aligner = None

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


class DeepseekVitClassifier(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        num_classes=1000,
        global_pool="avg",
        encoder_pool="",
        drop_rate=0.0,
        final_norm=True,
        *,
        rngs,
        **kwargs,
    ):
        if global_pool not in ("", "avg") or encoder_pool not in ("", "align"):
            raise ValueError(
                "classification pooling must be '' or 'avg'; encoder pooling '' or 'align'"
            )
        self.num_classes, self.global_pool = num_classes, global_pool
        self.encoder = DeepseekVitEncoder(global_pool=encoder_pool, rngs=rngs, **kwargs)
        self.num_features = (
            self.encoder.aligner.out_features
            if self.encoder.aligner is not None
            else self.encoder.num_features
        )
        self.norm = (
            RmsNormFp32(self.num_features, kwargs.get("norm_eps", 1e-6), affine=False, rngs=rngs)
            if final_norm
            else None
        )
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None
        )

    def forward_intermediates(self, x, out_indices=None):
        return self.encoder.forward_intermediates(x, out_indices)

    def forward_features(self, x):
        return self.encoder(x)

    def forward_head(self, x, pre_logits=False):
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        if self.global_pool == "avg":
            x = jnp.mean(x, axis=1)
        if self.norm is not None:
            x = self.norm(x)
        x = self.head_drop(x)
        return x if pre_logits or self.head is None else self.head(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _default_cfg(encoder=False):
    return _cfg(
        input_size=(3, 546, 546),
        interpolation="bicubic",
        crop_pct=1.0,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        num_classes=0 if encoder else 1000,
    )


def _make(suffix):
    def entry(**kwargs):
        args = {}
        if suffix == "_align":
            args.update(encoder_pool="align", out_features=5120)
        elif suffix == "_enc":
            args.update(out_features=5120)
            kwargs.pop("num_classes", None)
            kwargs.pop("drop_rate", None)
        if kwargs.get("encoder_pool") == "align":
            args.setdefault("out_features", 5120)
        model_cls = DeepseekVitEncoder if suffix == "_enc" else DeepseekVitClassifier
        model = model_cls(**dict(args, **kwargs))
        model.default_cfg = _default_cfg(suffix == "_enc")
        return model

    entry.__name__ = f"deepseek_vit_412m{suffix}"
    return entry


for _suffix in ("", "_align", "_enc"):
    register_model(_make(_suffix), default_cfg=_default_cfg(_suffix == "_enc"))
