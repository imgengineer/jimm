"""XCiT in flax nnx. Mirrors timm.models.xcit (cross-covariance attention + LPI + cls)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed, ClassifierMixin
from ..registry import register_model, _cfg

class XCA(nnx.Module):
    """Cross-Covariance Attention: attention over channels instead of tokens."""

    def __init__(self, dim, num_heads=8, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nnx.Param(jnp.ones((num_heads, 1, 1)))
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 4, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, head_dim, N)
        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-6)
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
        attn = nnx.softmax((q.transpose(0, 1, 3, 2) @ k) * self.temperature.value, axis=-1)
        x = (attn @ v.transpose(0, 1, 3, 2)).transpose(0, 3, 1, 2).reshape(B, N, C)
        return self.proj(x)

class XCABlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, init_values=1e-5, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.xca = XCA(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.gamma1 = nnx.Param(init_values * jnp.ones(dim))
        self.gamma2 = nnx.Param(init_values * jnp.ones(dim))

    def __call__(self, x):
        x = x + self.drop_path(self.gamma1.value * self.xca(self.norm1(x)))
        return x + self.drop_path(self.gamma2.value * self.mlp(self.norm2(x)))

class LPI(nnx.Module):
    """Local Patch Interaction: two 3x3 depthwise convs with residuals on 2D grid."""

    def __init__(self, dim, *, rngs):
        self.dw1 = nnx.Conv(dim, dim, (3, 3), feature_group_count=dim, rngs=rngs)
        self.dw2 = nnx.Conv(dim, dim, (3, 3), feature_group_count=dim, rngs=rngs)

    def __call__(self, x, H, W):
        B = x.shape[0]
        t = x.reshape(B, H, W, -1)
        t = t + self.dw1(t)
        t = t + self.dw2(t)
        return t.reshape(B, H * W, -1)

class XCiT(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = ""
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 global_pool="", embed_dim=384, depth=12, num_heads=8, mlp_ratio=4.0,
                 drop_rate=0.0, drop_path_rate=0.0, init_values=1e-5, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.grid = img_size // patch_size
        self.pos_embed = nnx.Param(jnp.zeros((1, n, embed_dim)))
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([XCABlock(embed_dim, num_heads, mlp_ratio, dpr[i],
                                         init_values, rngs=rngs) for i in range(depth)])
        self.lpi = nnx.List([LPI(embed_dim, rngs=rngs) for _ in range(depth)])
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.num_features)
        x = x + self.pos_embed.value
        x = jnp.concatenate([jnp.broadcast_to(self.cls_token.value, (B, 1, self.num_features)), x], axis=1)
        G = self.grid
        for blk, lpi in zip(self.blocks, self.lpi):
            cls, tokens = x[:, :1], x[:, 1:]
            tokens = lpi(tokens, G, G)
            x = jnp.concatenate([cls, tokens], axis=1)
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = x[:, 0] if self.global_pool == "" else jnp.mean(x[:, 1:], axis=1)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _xcit(embed_dim, depth, num_heads, **kwargs):
    model = XCiT(embed_dim=embed_dim, depth=depth, num_heads=num_heads, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def xcit_tiny_12_p16_224(**kwargs):
    return _xcit(192, 12, 4, **kwargs)

@register_model
def xcit_small_12_p16_224(**kwargs):
    return _xcit(384, 12, 8, **kwargs)

@register_model
def xcit_medium_24_p16_224(**kwargs):
    return _xcit(512, 24, 8, **kwargs)  # 512/12 not divisible; 8 heads x 64
