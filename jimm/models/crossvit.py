"""CrossViT in flax nnx. Mirrors timm.models.crossvit (dual-scale patches + cross attention)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed, ClassifierMixin
from ..registry import register_model, _cfg
from .vision_transformer import Attention

class CrossAttention(nnx.Module):
    """cls token of one branch attends to tokens of the other branch."""

    def __init__(self, q_dim, kv_dim, num_heads, *, rngs):
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nnx.Linear(q_dim, q_dim, rngs=rngs)
        self.kv = nnx.Linear(kv_dim, q_dim * 2, rngs=rngs)
        self.proj = nnx.Linear(q_dim, q_dim, rngs=rngs)

    def __call__(self, cls, tokens):
        B = tokens.shape[0]
        q = self.q(cls).reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        kv = self.kv(tokens).reshape(B, -1, 2, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = nnx.softmax(q @ k.transpose(0, 1, 3, 2) * self.scale, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, 1, -1)
        return self.proj(out)

class ViTBlock(nnx.Module):
    def __init__(self, dim, num_heads, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, dim * 4, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class CrossViT(ClassifierMixin, nnx.Module):
    _default_global_pool = ""
    default_cfg: dict = {}

    def __init__(self, img_size=224, in_chans=3, num_classes=1000, global_pool="",
                 embed_dims=(384, 768), patch_sizes=(12, 16), depths=(2, 2), num_heads=(6, 12),
                 cross_depths=(1, 1), drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = sum(embed_dims)
        self.branches = nnx.List([])
        branches = []
        for dim, ps, d, h in zip(embed_dims, patch_sizes, depths, num_heads):
            grid = -(-img_size // ps)  # ceil: matches SAME-padded conv output
            branches.append(nnx.List([
                PatchEmbed(img_size, ps, in_chans, dim, rngs=rngs),
                nnx.Param(jnp.zeros((1, grid * grid + 1, dim))),
                nnx.Param(jnp.zeros((1, 1, dim))),
                nnx.List([ViTBlock(dim, h, drop_path_rate, rngs=rngs) for _ in range(d)]),
            ]))
        self.branches = nnx.List(branches)
        # cross attention: for each branch, a cross-attn that pulls from the other
        self.cross = nnx.List([
            nnx.List([CrossAttention(embed_dims[0], embed_dims[1], num_heads[0], rngs=rngs),
                      CrossAttention(embed_dims[1], embed_dims[0], num_heads[1], rngs=rngs)])
            for _ in range(cross_depths[0])])
        self.norms = nnx.List([nnx.LayerNorm(embed_dims[0], rngs=rngs),
                               nnx.LayerNorm(embed_dims[1], rngs=rngs)])
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        tokens = []
        for patch, pos_embed, cls_token, blocks in self.branches:
            dim = cls_token.value.shape[-1]
            t = patch(x).reshape(B, -1, dim)
            t = jnp.concatenate([jnp.broadcast_to(cls_token.value, (B, 1, dim)), t], axis=1)
            t = t + pos_embed.value
            for blk in blocks:
                t = blk(t)
            tokens.append(t)
        # cross attention between the two branches (cls <-> tokens)
        for cross01, cross10 in self.cross:
            cls0 = tokens[0][:, :1] + cross01(tokens[0][:, :1], tokens[1][:, 1:])
            cls1 = tokens[1][:, :1] + cross10(tokens[1][:, :1], tokens[0][:, 1:])
            tokens[0] = jnp.concatenate([cls0, tokens[0][:, 1:]], axis=1)
            tokens[1] = jnp.concatenate([cls1, tokens[1][:, 1:]], axis=1)
        return [self.norms[i](tokens[i]) for i in range(2)]

    def forward_head(self, xs):
        x = jnp.concatenate([xs[0][:, 0], xs[1][:, 0]], axis=-1)
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "crossvit_tiny_224": ((384, 768), (12, 16), (2, 2), (6, 12)),
    "crossvit_small_224": ((480, 960), (12, 16), (2, 2), (6, 12)),
}

def _make(name):
    embed_dims, patch_sizes, depths, heads = _CFGS[name]

    def entry(**kwargs):
        model = CrossViT(embed_dims=embed_dims, patch_sizes=patch_sizes, depths=depths,
                         num_heads=heads, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
