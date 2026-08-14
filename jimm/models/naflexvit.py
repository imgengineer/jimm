"""NAFlexViT (Non-Autoregressive Flexible Resolution ViT) in flax nnx, NHWC. Mirrors timm.models.naflexvit."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed, global_pool_nhwc, ClassifierMixin
from ..registry import register_model, _cfg
from .vision_transformer import Attention

class NAFlexBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class NAFlexViT(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    default_cfg: dict = {}

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 num_classes: int = 1000, global_pool: str = "avg", embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, drop_path_rate: float = 0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.pos_embed = nnx.Param(jnp.zeros((1, n, embed_dim)))

        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([
            NAFlexBlock(embed_dim, num_heads, mlp_ratio, dpr[i], rngs=rngs)
            for i in range(depth)])

        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.num_features)
        x = x + self.pos_embed.value
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = global_pool_nhwc(x.reshape(x.shape[0], 1, -1, self.num_features), self.global_pool)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "naflexvit_base_patch16_gap": (768, 12, 12, 4.0),
    "naflexvit_base_patch16_par_gap": (768, 12, 12, 4.0),
    "naflexvit_base_patch16_parfac_gap": (768, 12, 12, 4.0),
    "naflexvit_base_patch16_map": (768, 12, 12, 4.0),
    "naflexvit_so150m2_patch16_reg1_gap": (896, 16, 14, 4.3),
    "naflexvit_so150m2_patch16_reg1_map": (896, 16, 14, 4.3),
    "naflexvit_base_patch16_siglip": (768, 12, 12, 4.0),
    "naflexvit_so400m_patch16_siglip": (1152, 27, 16, 3.73),
}

def _make(name):
    embed_dim, depth, num_heads, mlp_ratio = _CFGS[name]

    def entry(**kwargs):
        model = NAFlexViT(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            **kwargs,
        )
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
