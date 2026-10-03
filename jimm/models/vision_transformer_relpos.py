"""ViT with MLP-generated relative position bias in flax nnx. Mirrors timm.models.vision_transformer_relpos.

Each attention layer (or, for the ``srelpos`` variants, one shared module)
maps log-spaced 2D relative coordinates ``sign(d) * log(1 + |d|)`` through a
small ReLU MLP to a per-head bias over patch-token pairs; prefix tokens get
zero bias. Blocks are pre-norm with LayerScale, or for the ``rpn`` variants
res-post-norm (norm after attention/MLP, norm weights initialized to the
layer-scale value). There is no absolute position embedding.
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, Mlp, PatchEmbed
from ..registry import _cfg, register_model

_trunc = nnx.initializers.truncated_normal(0.02)
_LINEAR = dict(kernel_init=_trunc, bias_init=nnx.initializers.zeros)


def _rel_pos_tables(window):
    """timm's relative position index and 'cr' log coordinates for a ``window`` grid."""
    gh, gw = window
    coords = np.stack(np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")).reshape(2, -1)
    rel = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0)
    index = (rel[..., 0] + gh - 1) * (2 * gw - 1) + rel[..., 1] + gw - 1
    table = np.stack(
        np.meshgrid(np.arange(1 - gh, gh), np.arange(1 - gw, gw), indexing="ij"), axis=-1
    ).astype(np.float32)
    table = np.sign(table) * np.log1p(np.abs(table))
    return index, table


class RelPosMlp(nnx.Module):
    def __init__(self, window, num_heads, hidden_dim=128, prefix_tokens=0, *, rngs):
        self.num_heads, self.prefix_tokens = num_heads, prefix_tokens
        self.fc1 = nnx.Linear(2, hidden_dim, **_LINEAR, rngs=rngs)
        self.drop = nnx.Dropout(0.125, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, num_heads, **_LINEAR, rngs=rngs)
        index, table = _rel_pos_tables(window)
        # nnx.Variable: raw array attributes break nnx.cached_partial graph flattening
        self.relative_position_index = nnx.Variable(jnp.asarray(index))
        self.rel_coords_log = nnx.Variable(jnp.asarray(table))

    def __call__(self):
        """Bias ``[heads, N, N]`` with zero rows and columns for prefix tokens."""
        coords = self.rel_coords_log[...]
        bias = self.fc2(self.drop(nnx.relu(self.fc1(coords)))).reshape(-1, self.num_heads)
        bias = bias[self.relative_position_index[...]].transpose(2, 0, 1)
        p = self.prefix_tokens
        return jnp.pad(bias, ((0, 0), (p, 0), (p, 0))) if p else bias


class RelPosAttention(nnx.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, rel_pos=None, proj_drop=0.0, *, rngs):
        self.num_heads = num_heads
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, **_LINEAR, rngs=rngs)
        self.rel_pos = rel_pos
        self.proj = nnx.Linear(dim, dim, **_LINEAR, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x, shared_rel_pos=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        bias = self.rel_pos() if self.rel_pos is not None else shared_rel_pos
        x = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], bias=bias)
        return self.proj_drop(self.proj(x.reshape(B, N, C)))


def _norm(dim, scale=1.0, *, rngs):
    init = nnx.initializers.constant(scale)
    return nnx.LayerNorm(dim, epsilon=1e-6, scale_init=init, rngs=rngs)


class RelPosBlock(nnx.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio, qkv_bias, rel_pos, init_values, drop, dpr, *, rngs
    ):
        self.norm1 = _norm(dim, rngs=rngs)
        self.attn = RelPosAttention(dim, num_heads, qkv_bias, rel_pos, drop, rngs=rngs)
        self.ls1 = nnx.Param(init_values * jnp.ones(dim)) if init_values else None
        self.drop_path = DropPath(dpr, rngs=rngs)
        self.norm2 = _norm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, **_LINEAR, rngs=rngs)
        self.ls2 = nnx.Param(init_values * jnp.ones(dim)) if init_values else None

    def __call__(self, x, shared_rel_pos=None):
        y = self.attn(self.norm1(x), shared_rel_pos)
        x = x + self.drop_path(y * self.ls1[...] if self.ls1 is not None else y)
        y = self.mlp(self.norm2(x))
        return x + self.drop_path(y * self.ls2[...] if self.ls2 is not None else y)


class ResPostRelPosBlock(nnx.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio, qkv_bias, rel_pos, init_values, drop, dpr, *, rngs
    ):
        scale = init_values if init_values is not None else 1.0
        self.attn = RelPosAttention(dim, num_heads, qkv_bias, rel_pos, drop, rngs=rngs)
        self.norm1 = _norm(dim, scale, rngs=rngs)
        self.drop_path = DropPath(dpr, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, **_LINEAR, rngs=rngs)
        self.norm2 = _norm(dim, scale, rngs=rngs)

    def __call__(self, x, shared_rel_pos=None):
        x = x + self.drop_path(self.norm1(self.attn(x, shared_rel_pos)))
        return x + self.drop_path(self.norm2(self.mlp(x)))


class VisionTransformerRelPos(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=1e-6,
        class_token=False,
        fc_norm=False,
        rel_pos_dim=None,
        shared_rel_pos=False,
        res_post_norm=False,
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        window = self.patch_embed.grid_size

        def rel_pos():
            return RelPosMlp(
                window, num_heads, rel_pos_dim or 128, self.num_prefix_tokens, rngs=rngs
            )

        self.shared_rel_pos = rel_pos() if shared_rel_pos else None
        self.cls_token = (
            nnx.Param(1e-6 * nnx.initializers.normal(1.0)(rngs.params(), (1, 1, embed_dim)))
            if class_token
            else None
        )
        block = ResPostRelPosBlock if res_post_norm else RelPosBlock
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List(
            [
                block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    None if shared_rel_pos else rel_pos(),
                    init_values,
                    proj_drop_rate,
                    dpr[i],
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        self.norm = _norm(embed_dim, rngs=rngs) if not fc_norm else None
        self.fc_norm = _norm(embed_dim, rngs=rngs) if fc_norm else None
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, **_LINEAR, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.head = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        if self.cls_token is not None:
            x = jnp.concatenate([jnp.broadcast_to(self.cls_token[...], (B, 1, C)), x], axis=1)
        shared = self.shared_rel_pos() if self.shared_rel_pos is not None else None
        for blk in self.blocks:
            x = blk(x, shared)
        return self.norm(x) if self.norm is not None else x

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(axis=1)
        elif self.global_pool:
            x = x[:, 0]
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_S = dict(embed_dim=384, num_heads=6)
_M = dict(embed_dim=512, num_heads=8)
_B = dict(embed_dim=768, num_heads=12)
_P = dict(embed_dim=896, num_heads=14)
_CFGS = {
    "vit_relpos_base_patch32_plus_rpn_256": dict(
        _P, patch_size=32, img_size=256, res_post_norm=True
    ),
    "vit_relpos_base_patch16_plus_240": dict(_P, img_size=240),
    "vit_relpos_small_patch16_224": dict(_S, qkv_bias=False, fc_norm=True),
    "vit_relpos_medium_patch16_224": dict(_M, qkv_bias=False, fc_norm=True),
    "vit_relpos_base_patch16_224": dict(_B, qkv_bias=False, fc_norm=True),
    "vit_srelpos_small_patch16_224": dict(_S, qkv_bias=False, rel_pos_dim=384, shared_rel_pos=True),
    "vit_srelpos_medium_patch16_224": dict(
        _M, qkv_bias=False, rel_pos_dim=512, shared_rel_pos=True
    ),
    "vit_relpos_medium_patch16_cls_224": dict(
        _M, qkv_bias=False, rel_pos_dim=256, class_token=True, global_pool="token"
    ),
    "vit_relpos_base_patch16_cls_224": dict(
        _B, qkv_bias=False, class_token=True, global_pool="token"
    ),
    "vit_relpos_base_patch16_clsgap_224": dict(_B, qkv_bias=False, fc_norm=True, class_token=True),
    "vit_relpos_small_patch16_rpn_224": dict(_S, qkv_bias=False, res_post_norm=True),
    "vit_relpos_medium_patch16_rpn_224": dict(_M, qkv_bias=False, res_post_norm=True),
    "vit_relpos_base_patch16_rpn_224": dict(_B, qkv_bias=False, res_post_norm=True),
}


def _make(name):
    cfg = _CFGS[name]
    size = cfg.get("img_size", 224)

    def entry(**kwargs):
        model = VisionTransformerRelPos(**{**cfg, **kwargs})
        model.default_cfg = _cfg(
            input_size=(3, size, size),
            crop_pct=0.9,
            interpolation="bicubic",
            fixed_input_size=True,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
