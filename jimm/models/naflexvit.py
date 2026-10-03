"""NaFlexViT in flax nnx, NHWC. Mirrors timm.models.naflexvit for dense image inputs.

Patches (row-major, channels last within a patch) go through a linear
projection; a learned 16x16 position grid is resized to the patch grid with
antialiased bicubic interpolation (square-resized then cropped when aspect
ratio preserving), or factorized row and column tables are linearly resized
and summed. Register tokens are prepended and pre-norm ViT blocks with layer
scale follow. Heads average the patch tokens with fc_norm, or attend from a
learned latent (MAP). Pre-patchified NaFlex inputs with padding are not
supported.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model

_trunc = nnx.initializers.truncated_normal(0.02)
_LINEAR = dict(kernel_init=_trunc, bias_init=nnx.initializers.zeros)


def _gelu_tanh(x):
    return jax.nn.gelu(x, approximate=True)


def _ln(dim, *, rngs):
    return nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)


def _scaled(std):
    return dict(
        kernel_init=nnx.initializers.truncated_normal(std), bias_init=nnx.initializers.zeros
    )


class Mlp(nnx.Module):
    def __init__(self, dim, hidden, act, drop=0.0, out_std=0.02, *, rngs):
        self.fc1 = nnx.Linear(dim, hidden, **_LINEAR, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, **_scaled(out_std), rngs=rngs)
        self.act = act
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nnx.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, proj_drop=0.0, out_std=0.02, *, rngs):
        self.num_heads = num_heads
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, **_LINEAR, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, **_scaled(out_std), rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        x = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        return self.proj_drop(self.proj(x.reshape(B, N, C)))


class Block(nnx.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio, qkv_bias, init_values, act, drop, dpr, out_std,
        *, rngs,
    ):  # fmt: skip
        self.norm1 = _ln(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, qkv_bias, drop, out_std, rngs=rngs)
        self.ls1 = nnx.Param(jnp.full((dim,), init_values)) if init_values else None
        self.drop_path = DropPath(dpr, rngs=rngs)
        self.norm2 = _ln(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act, drop, out_std, rngs=rngs)
        self.ls2 = nnx.Param(jnp.full((dim,), init_values)) if init_values else None

    def __call__(self, x):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y * self.ls1[...] if self.ls1 is not None else y)
        y = self.mlp(self.norm2(x))
        return x + self.drop_path(y * self.ls2[...] if self.ls2 is not None else y)


class NaFlexEmbeds(nnx.Module):
    def __init__(
        self, patch_size, in_chans, embed_dim, reg_tokens, pos_embed, grid, ar_preserving,
        *, rngs,
    ):  # fmt: skip
        self.patch_size, self.pos_embed_type, self.ar_preserving = (
            patch_size,
            pos_embed,
            ar_preserving,
        )
        self.proj = nnx.Linear(patch_size**2 * in_chans, embed_dim, **_LINEAR, rngs=rngs)
        normal = nnx.initializers.normal
        self.reg_token = (
            nnx.Param(normal(1e-6)(rngs.params(), (1, reg_tokens, embed_dim)))
            if reg_tokens
            else None
        )
        h, w = grid

        def table(*shape):
            return nnx.Param(normal(0.02)(rngs.params(), shape))

        factorized, learned = pos_embed == "factorized", pos_embed == "learned"
        self.pos_embed_y = table(1, h, embed_dim) if factorized else None
        self.pos_embed_x = table(1, w, embed_dim) if factorized else None
        self.pos_embed = table(1, h, w, embed_dim) if learned else None

    def _learned(self, gh, gw):
        pe = self.pos_embed[...]
        if pe.shape[1:3] != (gh, gw):
            size = (max(gh, gw),) * 2 if self.ar_preserving else (gh, gw)
            pe = jax.image.resize(
                pe.astype(jnp.float32), (1, *size, pe.shape[-1]), "cubic", antialias=True
            )[:, :gh, :gw]
        return pe.reshape(1, gh * gw, -1)

    def _factorized(self, gh, gw):
        ly, lx = (max(gh, gw),) * 2 if self.ar_preserving else (gh, gw)

        def interp(table, length):
            if table.shape[1] == length:
                return table
            out = (1, length, table.shape[-1])
            return jax.image.resize(table.astype(jnp.float32), out, "linear", antialias=False)

        pe_y = interp(self.pos_embed_y[...], ly)[:, :gh]
        pe_x = interp(self.pos_embed_x[...], lx)[:, :gw]
        return (pe_y[:, :, None] + pe_x[:, None]).reshape(1, gh * gw, -1)

    def __call__(self, x):
        B, H, W, C = x.shape
        p = self.patch_size
        gh, gw = H // p, W // p
        x = x[:, : gh * p, : gw * p].reshape(B, gh, p, gw, p, C).transpose(0, 1, 3, 2, 4, 5)
        x = self.proj(x.reshape(B, gh * gw, p * p * C))
        if self.pos_embed_type == "learned":
            x = x + self._learned(gh, gw).astype(x.dtype)
        elif self.pos_embed_type == "factorized":
            x = x + self._factorized(gh, gw).astype(x.dtype)
        if self.reg_token is not None:
            reg = jnp.broadcast_to(self.reg_token[...], (B, *self.reg_token.shape[1:]))
            x = jnp.concatenate([reg, x], axis=1)
        return x


class AttentionPoolLatent(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio, act, *, rngs):
        self.num_heads = num_heads
        self.latent = nnx.Param(
            nnx.initializers.truncated_normal(dim**-0.5)(rngs.params(), (1, 1, dim))
        )
        self.q = nnx.Linear(dim, dim, **_LINEAR, rngs=rngs)
        self.kv = nnx.Linear(dim, dim * 2, **_LINEAR, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, **_LINEAR, rngs=rngs)
        self.norm = _ln(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(jnp.broadcast_to(self.latent[...], (B, 1, C))).reshape(B, 1, h, C // h)
        kv = self.kv(x).reshape(B, N, 2, h, C // h)
        x = self.proj(dot_product_attention(q, kv[:, :, 0], kv[:, :, 1]).reshape(B, 1, C))
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class NaFlexVit(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        reg_tokens=0,
        pos_embed="learned",
        pos_embed_grid_size=(16, 16),
        pos_embed_ar_preserving=False,
        global_pool="map",
        fc_norm=None,
        act_layer="gelu",
        num_classes=1000,
        in_chans=3,
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        assert global_pool in ("", "avg", "max", "avgmax", "map")
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.num_prefix_tokens = reg_tokens
        act = _gelu_tanh if act_layer == "gelu_tanh" else gelu
        self.embeds = NaFlexEmbeds(
            patch_size, in_chans, embed_dim, reg_tokens, pos_embed, pos_embed_grid_size,
            pos_embed_ar_preserving, rngs=rngs,
        )  # fmt: skip
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        # timm's fix_init scales each block's residual output projections by 1/sqrt(2 * depth).
        self.blocks = nnx.List(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    init_values,
                    act,
                    proj_drop_rate,
                    dpr[i],
                    0.02 / math.sqrt(2.0 * (i + 1)),
                    rngs=rngs,
                )  # fmt: skip
                for i in range(depth)
            ]
        )
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        self.norm = _ln(embed_dim, rngs=rngs) if not use_fc_norm else None
        self.attn_pool = (
            AttentionPoolLatent(embed_dim, num_heads, mlp_ratio, act, rngs=rngs)
            if global_pool == "map"
            else None
        )
        self.fc_norm = _ln(embed_dim, rngs=rngs) if use_fc_norm else None
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, **_LINEAR, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            assert global_pool in ("", "avg", "max", "avgmax", "map")
            if global_pool != "map":
                self.attn_pool = None
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.head = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.embeds(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x) if self.norm is not None else x

    def _pool(self, x):
        if self.attn_pool is not None:
            return self.attn_pool(x[:, self.num_prefix_tokens :])
        if not self.global_pool:
            return x
        x = x[:, self.num_prefix_tokens :]
        if self.global_pool == "avg":
            return x.mean(axis=1)
        if self.global_pool == "max":
            return x.max(axis=1)
        return 0.5 * (x.mean(axis=1) + x.max(axis=1))

    def forward_head(self, x):
        x = self._pool(x)
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_B = dict(embed_dim=768, depth=12, num_heads=12)
_SO150 = dict(embed_dim=832, depth=21, num_heads=13, mlp_ratio=34 / 13, qkv_bias=False)
_GAP = dict(init_values=1e-5, global_pool="avg", reg_tokens=4, fc_norm=True)
_CFGS = {
    "naflexvit_base_patch16_gap": dict(_B, **_GAP),
    "naflexvit_base_patch16_par_gap": dict(_B, pos_embed_ar_preserving=True, **_GAP),
    "naflexvit_base_patch16_parfac_gap": dict(
        _B, pos_embed_ar_preserving=True, pos_embed="factorized", **_GAP
    ),
    "naflexvit_base_patch16_map": dict(_B, init_values=1e-5, global_pool="map", reg_tokens=1),
    "naflexvit_so150m2_patch16_reg1_gap": dict(
        _SO150, init_values=1e-5, reg_tokens=1, global_pool="avg", fc_norm=True
    ),
    "naflexvit_so150m2_patch16_reg1_map": dict(
        _SO150, init_values=1e-5, reg_tokens=1, global_pool="map"
    ),
    "naflexvit_base_patch16_siglip": dict(_B, act_layer="gelu_tanh", global_pool="map"),
    "naflexvit_so400m_patch16_siglip": dict(
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7362,
        act_layer="gelu_tanh",
        global_pool="map",
    ),  # fmt: skip
}


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = NaFlexVit(**{**cfg, **kwargs})
        model.default_cfg = _cfg(
            input_size=(3, 384, 384),
            crop_pct=1.0,
            interpolation="bicubic",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
