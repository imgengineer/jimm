"""MLP-Mixer, gMixer, ResMLP and gMLP in flax nnx. Mirrors timm.models.mlp_mixer.

timm builds all four from one ``MlpMixer`` with different block, MLP and norm layers:
Mixer blocks (token and channel MLPs; GLU MLPs with SiLU for gMixer), ResMLP blocks (affine
norms, a linear token mixer and layer scale) and gMLP spatial gating blocks.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model


class Affine(nnx.Module):
    """ResMLP's per-channel ``alpha * x + beta`` in place of normalization."""

    def __init__(self, dim, *, rngs=None):
        self.alpha = nnx.Param(jnp.ones((1, 1, dim)))
        self.beta = nnx.Param(jnp.zeros((1, 1, dim)))

    def __call__(self, x):
        return self.beta[...] + self.alpha[...] * x


class Mlp(nnx.Module):
    """timm ``Mlp``, or ``GluMlp`` (``glu=True``, gate on the second half) for gMixer."""

    def __init__(self, dim, hidden, act, glu=False, drop=0.0, *, rngs):
        self.act, self.glu = act, glu
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden // 2 if glu else hidden, dim, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        x = self.fc1(x)
        if self.glu:
            x1, x2 = jnp.split(x, 2, axis=-1)
            x = x1 * self.act(x2)
        else:
            x = self.act(x)
        return self.drop(self.fc2(self.drop(x)))


class MixerBlock(nnx.Module):
    def __init__(self, dim, seq_len, mlp_ratio, cfg, drop=0.0, drop_path=0.0, *, rngs):
        tokens_dim, channels_dim = (int(r * dim) for r in mlp_ratio)
        act, glu = cfg["act"], cfg["glu"]
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act, glu, drop, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp_channels = Mlp(dim, channels_dim, act, glu, drop, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.mlp_tokens(self.norm1(x).transpose(0, 2, 1)).transpose(0, 2, 1)
        x = x + self.drop_path(y)
        return x + self.drop_path(self.mlp_channels(self.norm2(x)))


class ResBlock(nnx.Module):
    """ResMLP block: affine norms, a linear token mixer and per-channel layer scale."""

    def __init__(self, dim, seq_len, mlp_ratio, cfg, drop=0.0, drop_path=0.0, *, rngs):
        init_values = cfg["init_values"]
        self.norm1 = Affine(dim)
        self.linear_tokens = nnx.Linear(seq_len, seq_len, rngs=rngs)
        self.norm2 = Affine(dim)
        self.mlp_channels = Mlp(dim, int(dim * mlp_ratio), cfg["act"], drop=drop, rngs=rngs)
        self.ls1 = nnx.Param(jnp.full((dim,), init_values))
        self.ls2 = nnx.Param(jnp.full((dim,), init_values))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.linear_tokens(self.norm1(x).transpose(0, 2, 1)).transpose(0, 2, 1)
        x = x + self.drop_path(self.ls1[...] * y)
        return x + self.drop_path(self.ls2[...] * self.mlp_channels(self.norm2(x)))


class SpatialGatingUnit(nnx.Module):
    """Gates the first channel half with a token projection of the normalized second half."""

    def __init__(self, dim, seq_len, *, rngs):
        self.norm = nnx.LayerNorm(dim // 2, epsilon=1e-5, rngs=rngs)
        self.proj = nnx.Linear(seq_len, seq_len, rngs=rngs)

    def __call__(self, x):
        u, v = jnp.split(x, 2, axis=-1)
        v = self.proj(self.norm(v).transpose(0, 2, 1)).transpose(0, 2, 1)
        return u * v


class GatedMlp(nnx.Module):
    def __init__(self, dim, hidden, seq_len, act, drop=0.0, *, rngs):
        self.act = act
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.gate = SpatialGatingUnit(hidden, seq_len, rngs=rngs)
        self.fc2 = nnx.Linear(hidden // 2, dim, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        x = self.gate(self.drop(self.act(self.fc1(x))))
        return self.drop(self.fc2(x))


class SpatialGatingBlock(nnx.Module):
    def __init__(self, dim, seq_len, mlp_ratio, cfg, drop=0.0, drop_path=0.0, *, rngs):
        self.norm = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp_channels = GatedMlp(
            dim, int(dim * mlp_ratio), seq_len, cfg["act"], drop, rngs=rngs
        )
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        return x + self.drop_path(self.mlp_channels(self.norm(x)))


class PatchEmbed(nnx.Module):
    def __init__(self, img_size, patch_size, in_chans, dim, *, rngs):
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        p = (patch_size, patch_size)
        self.proj = nnx.Conv(in_chans, dim, p, strides=p, padding="VALID", rngs=rngs)

    def __call__(self, x):
        x = self.proj(x)
        return x.reshape(x.shape[0], -1, x.shape[-1])


_BLOCKS = {"mixer": MixerBlock, "res": ResBlock, "gmlp": SpatialGatingBlock}


class MlpMixer(ClassifierMixin, nnx.Module):
    """timm ``MlpMixer`` with ``block`` mixer, res (ResMLP) or gmlp."""

    _classifier_attr = "head"

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_blocks=8,
        embed_dim=512,
        mlp_ratio=(0.5, 4.0),
        block="mixer",
        act="gelu",
        glu=False,
        init_values=1e-4,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.stem = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        cfg = {"act": nnx.silu if act == "silu" else gelu, "glu": glu, "init_values": init_values}
        n = self.stem.num_patches
        # timm gives every block the full drop-path rate.
        self.blocks = nnx.List(
            [
                _BLOCKS[block](embed_dim, n, mlp_ratio, cfg, drop_rate, drop_path_rate, rngs=rngs)
                for _ in range(num_blocks)
            ]
        )
        self.norm = (
            Affine(embed_dim)
            if block == "res"
            else nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs)
        )
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_intermediates(self, x, out_indices=None):
        """Token feature maps (B, H, W, C) after each block."""
        from ..features import _select_features

        gh, gw = self.stem.grid_size
        x, feats = self.stem(x), []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x.reshape(x.shape[0], gh, gw, -1))
        return _select_features(feats, out_indices)

    def forward_features(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x, pre_logits=False):
        x = jnp.mean(x, axis=1) if self.global_pool == "avg" else x
        x = self.head_drop(x)
        if pre_logits or self.head is None:
            return x
        return self.head(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


ResMLP = MlpMixer

_HALF = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
_GMIXER = dict(mlp_ratio=(1.0, 4.0), glu=True, act="silu")
# name: (patch, blocks, width, overrides, eval cfg) — timm model registrations.
_CFGS = {
    "mixer_s32_224": (32, 8, 512, {}, _HALF),
    "mixer_s16_224": (16, 8, 512, {}, _HALF),
    "mixer_b32_224": (32, 12, 768, {}, _HALF),
    "mixer_b16_224": (16, 12, 768, {}, _HALF),
    "mixer_l32_224": (32, 24, 1024, {}, _HALF),
    "mixer_l16_224": (16, 24, 1024, {}, _HALF),
    "gmixer_12_224": (16, 12, 384, _GMIXER, {}),
    "gmixer_24_224": (16, 24, 384, _GMIXER, {}),
    "resmlp_12_224": (16, 12, 384, dict(mlp_ratio=4, block="res"), {}),
    "resmlp_24_224": (16, 24, 384, dict(mlp_ratio=4, block="res", init_values=1e-5), {}),
    "resmlp_36_224": (16, 36, 384, dict(mlp_ratio=4, block="res", init_values=1e-6), {}),
    "resmlp_big_24_224": (8, 24, 768, dict(mlp_ratio=4, block="res", init_values=1e-6), {}),
    "gmlp_ti16_224": (16, 30, 128, dict(mlp_ratio=6, block="gmlp"), _HALF),
    "gmlp_s16_224": (16, 30, 256, dict(mlp_ratio=6, block="gmlp"), _HALF),
    "gmlp_b16_224": (16, 30, 512, dict(mlp_ratio=6, block="gmlp"), _HALF),
}


def _make(name):
    patch, blocks, dim, overrides, ev = _CFGS[name]

    def entry(**kwargs):
        model = MlpMixer(
            patch_size=patch, num_blocks=blocks, embed_dim=dim, **{**overrides, **kwargs}
        )
        model.default_cfg = _cfg(interpolation="bicubic", **ev)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
