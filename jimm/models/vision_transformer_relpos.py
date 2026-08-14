"""ViT with 2D relative position bias in flax nnx. Mirrors timm.models.vision_transformer_relpos."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed, ClassifierMixin
from ..registry import register_model, _cfg

class RelPosAttention(nnx.Module):
    def __init__(self, dim, num_heads, grid, qkv_bias=True, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        gh, gw = grid, grid
        # rel pos over patch tokens (exclude cls)
        self.rel_bias = nnx.Param(jnp.zeros(((2 * gh - 1) * (2 * gw - 1), num_heads)))
        coords = jnp.stack(jnp.meshgrid(jnp.arange(gh), jnp.arange(gw), indexing="ij"))
        cf = coords.reshape(2, -1)
        rel = (cf[:, :, None] - cf[:, None, :]).transpose(1, 2, 0) + jnp.array([gh - 1, gw - 1])
        self.rel_index = rel[:, :, 0] * (2 * gw - 1) + rel[:, :, 1]  # (gh*gw, gh*gw)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(0, 1, 3, 2) * self.scale
        bias = self.rel_bias.value[self.rel_index].transpose(2, 0, 1)  # (heads, n, n)
        bias = jnp.pad(bias, ((0, 0), (1, 0), (1, 0)))  # cls row/col get 0 bias
        attn = attn + bias[None]
        attn = nnx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)

class RelPosBlock(nnx.Module):
    def __init__(self, dim, num_heads, grid, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = RelPosAttention(dim, num_heads, grid, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class VisionTransformerRelPos(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = ""
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 global_pool="", embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        grid = img_size // patch_size
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([RelPosBlock(embed_dim, num_heads, grid, mlp_ratio,
                                            dpr[i], rngs=rngs) for i in range(depth)])
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.num_features)
        x = jnp.concatenate([jnp.broadcast_to(self.cls_token.value, (B, 1, self.num_features)), x], axis=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = x[:, 0] if self.global_pool == "" else jnp.mean(x[:, 1:], axis=1)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def vit_relpos_base_patch16_224(**kwargs):
    model = VisionTransformerRelPos(**kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_relpos_small_patch16_224(**kwargs):
    model = VisionTransformerRelPos(embed_dim=384, num_heads=6, **kwargs)
    model.default_cfg = _cfg()
    return model
