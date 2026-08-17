"""Hybrid ViT (CNN feature map as patch tokens) in flax nnx. Mirrors timm.models.vision_transformer_hybrid."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg
from .vision_transformer import Attention
from .resnet import Bottleneck

class HybridStem(nnx.Module):
    """Small ResNet-ish CNN front-end producing the token grid (no final pool)."""

    def __init__(self, in_chans, out_chs, *, rngs):
        self.conv1 = nnx.Conv(in_chans, 64, (7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)],
                              use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        blocks, chs = [], 64
        for i, n in enumerate([2, 2, 2]):
            width = 64 * 2**i
            for j in range(n):
                blocks.append(Bottleneck(chs, width, 2 if j == 0 else 1, rngs=rngs))
                chs = width * Bottleneck.expansion
        self.blocks = nnx.List(blocks)
        self.proj = nnx.Conv(chs, out_chs, (1, 1), use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = nnx.max_pool(nnx.relu(self.bn1(self.conv1(x))), (3, 3), strides=(2, 2), padding="SAME")
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)

class VisionTransformerHybrid(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = ""
    default_cfg: dict | None = None

    def __init__(self, in_chans=3, num_classes=1000, global_pool="", embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.stem = HybridStem(in_chans, embed_dim, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([_HybridBlock(embed_dim, num_heads, mlp_ratio, dpr[i], rngs=rngs)
                                for i in range(depth)])
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem(x).reshape(B, -1, self.num_features)
        x = jnp.concatenate([jnp.broadcast_to(self.cls_token[...], (B, 1, self.num_features)), x], axis=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = x[:, 0] if self.global_pool == "" else jnp.mean(x[:, 1:], axis=1)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

class _HybridBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

@register_model
def vit_base_hybrid_patch16_224(**kwargs):
    model = VisionTransformerHybrid(**kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_hybrid_patch16_224(**kwargs):
    model = VisionTransformerHybrid(embed_dim=384, num_heads=6, **kwargs)
    model.default_cfg = _cfg()
    return model
