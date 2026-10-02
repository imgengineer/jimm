"""Qwen3-VL / Qwen3.5 / Qwen3.8 vision towers in Flax NNX (NHWC).

Adapted for JAX from timm 1.0.30, Copyright 2026 Yonghye Kwon.
Reference: https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/qwen3_vit.py
"""

import jax.numpy as jnp
from flax import nnx

from ..features import _select_features
from ..layers import ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model
from ._rope_vit import DynamicPatchEmbed, RopeAttention, axial_rope, resample_pos_embed_grid


class Qwen3VitBlock(nnx.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio, norm_eps, proj_drop, attn_drop, drop_path, *, rngs
    ):
        self.norm1 = nnx.LayerNorm(dim, epsilon=norm_eps, rngs=rngs)
        self.attn = RopeAttention(dim, num_heads, proj_drop, attn_drop, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=norm_eps, rngs=rngs)
        self.fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        self.mlp_drop = nnx.Dropout(proj_drop, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x, rope):
        x = x + self.drop_path(self.attn(self.norm1(x), rope))
        y = self.mlp_drop(nnx.gelu(self.fc1(self.norm2(x)), approximate=True))
        return x + self.drop_path(self.mlp_drop(self.fc2(y)))


class Qwen3VitPatchMerger(nnx.Module):
    def __init__(self, dim, merge_size=2, out_features=0, norm_eps=1e-6, *, rngs):
        self.merge_size = merge_size
        self.hidden_size = dim * merge_size**2
        self.out_features = out_features if out_features > 0 else self.hidden_size
        self.norm = nnx.LayerNorm(dim, epsilon=norm_eps, rngs=rngs)
        self.fc1 = nnx.Linear(self.hidden_size, self.hidden_size, rngs=rngs)
        self.fc2 = (
            nnx.Linear(self.hidden_size, out_features, rngs=rngs) if out_features > 0 else None
        )

    def merge(self, x):
        batch, height, width, dim = x.shape
        size = self.merge_size
        if height % size or width % size:
            raise ValueError("patch grid height and width must be divisible by merge_size")
        x = x.reshape(batch, height // size, size, width // size, size, dim)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(batch, (height // size) * (width // size), size * size * dim)

    def __call__(self, x, pre_logits=False):
        x = gelu(self.fc1(self.merge(self.norm(x))))
        return x if pre_logits or self.fc2 is None else self.fc2(x)


class Qwen3VitEncoder(nnx.Module):
    def __init__(
        self,
        img_size=768,
        patch_size=16,
        in_chans=3,
        out_features=0,
        global_pool="merge",
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=4304 / 1152,
        merge_size=2,
        pos_embed_grid_size=48,
        rope_temperature=10000.0,
        norm_eps=1e-6,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        if global_pool not in ("", "avg", "merge"):
            raise ValueError("global_pool must be '', 'avg', or 'merge'")
        self.num_classes, self.global_pool = 0, global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pos_embed_grid_size = pos_embed_grid_size
        self.rope_temperature = rope_temperature
        self.patch_embed = DynamicPatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        self.pos_embed = nnx.Param(
            nnx.initializers.truncated_normal(0.02)(
                rngs.params(),
                (1, pos_embed_grid_size**2, embed_dim),
            )
        )
        self.blocks = nnx.List(
            [
                Qwen3VitBlock(
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
        self.merger = (
            Qwen3VitPatchMerger(embed_dim, merge_size, out_features, norm_eps, rngs=rngs)
            if global_pool == "merge"
            else None
        )

    def _embed(self, x):
        x = self.patch_embed(x)
        batch, height, width, dim = x.shape
        pos = resample_pos_embed_grid(
            self.pos_embed[...], self.pos_embed_grid_size, (height, width)
        )
        x = x.reshape(batch, height * width, dim) + pos.astype(x.dtype)
        rope = axial_rope(height, width, dim // self.num_heads, self.rope_temperature)
        return x, rope, (batch, height, width, dim)

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
        return x.reshape(shape)

    def forward_head(self, x):
        if self.merger is not None:
            return self.merger(x)
        return jnp.mean(x, axis=(1, 2)) if self.global_pool == "avg" else x

    def get_classifier(self):
        return None

    def reset_classifier(self, num_classes, global_pool=None):
        if num_classes:
            raise ValueError("use a Qwen3VitClassifier for classification")
        if global_pool is not None:
            if global_pool not in ("", "avg", "merge"):
                raise ValueError("global_pool must be '', 'avg', or 'merge'")
            if global_pool == "merge" and self.merger is None:
                raise ValueError("this encoder has no spatial merger")
            self.global_pool = global_pool
            if global_pool != "merge":
                self.merger = None

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


class Qwen3VitClassifier(ClassifierMixin, nnx.Module):
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
        if global_pool not in ("", "avg") or encoder_pool not in ("", "merge"):
            raise ValueError(
                "classification pooling must be '' or 'avg'; encoder pooling '' or 'merge'"
            )
        self.num_classes, self.global_pool = num_classes, global_pool
        self.encoder = Qwen3VitEncoder(global_pool=encoder_pool, rngs=rngs, **kwargs)
        self.num_features = (
            self.encoder.merger.out_features
            if self.encoder.merger is not None
            else self.encoder.num_features
        )
        self.norm = (
            nnx.LayerNorm(
                self.num_features,
                epsilon=kwargs.get("norm_eps", 1e-6),
                use_bias=False,
                use_scale=False,
                rngs=rngs,
            )
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


_SPECS = {
    "88m": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0),
    "306m": dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0),
    "416m": dict(embed_dim=1152, depth=27, num_heads=16, mlp_ratio=4304 / 1152),
}
_OUT_FEATURES = {"88m": 1024, "306m": 2048, "416m": 5120}


def _default_cfg(encoder=False):
    return _cfg(
        input_size=(3, 768, 768),
        interpolation="bicubic",
        crop_pct=1.0,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        num_classes=0 if encoder else 1000,
    )


def _make(size, suffix):
    def entry(**kwargs):
        args = dict(_SPECS[size])
        if suffix == "_merge":
            args.update(encoder_pool="merge", out_features=_OUT_FEATURES[size])
        elif suffix == "_enc":
            args.update(out_features=_OUT_FEATURES[size])
            kwargs.pop("num_classes", None)
            kwargs.pop("drop_rate", None)
        if kwargs.get("encoder_pool") == "merge":
            args.setdefault("out_features", _OUT_FEATURES[size])
        model_cls = Qwen3VitEncoder if suffix == "_enc" else Qwen3VitClassifier
        model = model_cls(**dict(args, **kwargs))
        model.default_cfg = _default_cfg(suffix == "_enc")
        return model

    entry.__name__ = f"qwen3_vit_{size}{suffix}"
    return entry


for _size in _SPECS:
    for _suffix in ("", "_merge", "_enc"):
        register_model(_make(_size, _suffix), default_cfg=_default_cfg(_suffix == "_enc"))
