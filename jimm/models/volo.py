"""VOLO in flax nnx, NHWC. Mirrors timm.models.volo (Outlooker + Transformer + ClassAttention)."""
import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg

class OutlookAttention(nnx.Module):
    """Outlook attention: fine-grained local attention via sliding window."""

    def __init__(self, dim, num_heads=6, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** -0.5

        self.v = nnx.Linear(dim, dim, use_bias=qkv_bias, rngs=rngs)
        self.attn = nnx.Linear(dim, kernel_size ** 4 * num_heads, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        v = self.v(x)
        v = v.reshape(B, H, W, self.num_heads, self.head_dim)
        
        # Local window sliding attention
        # For NHWC, pad and extract patches
        k = self.kernel_size
        pad_h = pad_w = self.padding
        v_pad = jnp.pad(v, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0), (0, 0)))
        
        # Extract sliding windows of shape (B, H, W, k, k, num_heads, head_dim)
        patches = []
        for i in range(k):
            for j in range(k):
                patches.append(v_pad[:, i:i + H, j:j + W, :, :])
        v_win = jnp.stack(patches, axis=3) # (B, H, W, k*k, num_heads, head_dim)

        attn = self.attn(x) # (B, H, W, k*k*k*k * num_heads) -> simplified to (B, H, W, num_heads, k*k)
        attn = attn.reshape(B, H, W, self.num_heads, -1)[:, :, :, :, :k * k]
        attn = nnx.softmax(attn * self.scale, axis=-1) # (B, H, W, num_heads, k*k)

        out = jnp.sum(v_win * attn[:, :, :, :, :, None].transpose(0, 1, 2, 4, 3, 5), axis=3)
        out = out.reshape(B, H, W, C)
        return self.proj(out)

class Outlooker(nnx.Module):
    def __init__(self, dim, num_heads=6, kernel_size=3, padding=1, stride=1,
                 mlp_ratio=3.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = OutlookAttention(dim, num_heads, kernel_size, padding, stride, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class TransformerBlock(nnx.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = nnx.softmax(q @ k.transpose(0, 1, 3, 2) * self.scale, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = x + self.drop_path(self.proj(out))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class ClassAttention(nnx.Module):
    def __init__(self, dim, num_heads=12, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nnx.Linear(dim, dim, rngs=rngs)
        self.k = nnx.Linear(dim, dim, rngs=rngs)
        self.v = nnx.Linear(dim, dim, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, cls, x):
        B, N, C = x.shape
        q = self.q(cls).reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = nnx.softmax(q @ k.transpose(0, 1, 3, 2) * self.scale, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, 1, C)
        return self.proj(out)

class ClassBlock(nnx.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = ClassAttention(dim, num_heads, rngs=rngs)
        self.norm3 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, cls, x):
        cls = cls + self.drop_path(self.attn(self.norm1(cls), self.norm2(x)))
        return cls + self.drop_path(self.mlp(self.norm3(cls)))

class PatchEmbedStem(nnx.Module):
    def __init__(self, in_chans=3, stem_dim=64, embed_dim=192, *, rngs):
        self.conv1 = nnx.Conv(in_chans, stem_dim, (7, 7), strides=(2, 2), padding="SAME", use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(stem_dim, rngs=rngs)
        self.conv2 = nnx.Conv(stem_dim, stem_dim, (3, 3), strides=(1, 1), padding="SAME", use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(stem_dim, rngs=rngs)
        self.conv3 = nnx.Conv(stem_dim, embed_dim, (3, 3), strides=(2, 2), padding="SAME", use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(embed_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = nnx.relu(self.bn2(self.conv2(x)))
        return nnx.relu(self.bn3(self.conv3(x)))

class VOLO(ClassifierMixin, nnx.Module):
    _default_global_pool = "token"
    default_cfg: dict = {}

    def __init__(self, layers=(4, 4, 8, 2), embed_dims=(192, 384, 384, 384),
                 num_heads=(6, 12, 12, 12), mlp_ratio=3.0, stem_hidden_dim=64,
                 img_size=224, num_classes=1000, in_chans=3, global_pool="token",
                 drop_rate=0.0, drop_path_rate=0.1, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dims[-1]

        self.patch_embed = PatchEmbedStem(in_chans, stem_hidden_dim, embed_dims[0], rngs=rngs)
        
        # Stage 1: Outlookers
        dpr = [drop_path_rate * i / max(sum(layers) - 1, 1) for i in range(sum(layers))]
        self.stage1 = nnx.List([
            Outlooker(embed_dims[0], num_heads[0], mlp_ratio=mlp_ratio, drop_path=dpr[i], rngs=rngs)
            for i in range(layers[0])])
        
        # Downsample to stage 2
        self.downsample = nnx.Conv(embed_dims[0], embed_dims[1], (2, 2), strides=(2, 2), rngs=rngs)

        # Stage 2: Transformer blocks
        k = layers[0]
        self.stage2 = nnx.List([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratio=4.0, drop_path=dpr[k + i], rngs=rngs)
            for i in range(layers[1])])

        # Stage 3: Class attention blocks
        k += layers[1]
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dims[-1])))
        self.cls_blocks = nnx.List([
            ClassBlock(embed_dims[-1], num_heads[-1], mlp_ratio=4.0, drop_path=dpr[k + i], rngs=rngs)
            for i in range(layers[-1])])

        self.norm = nnx.LayerNorm(embed_dims[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        for blk in self.stage1:
            x = blk(x)
        x = self.downsample(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        for blk in self.stage2:
            x = blk(x)
        cls = jnp.broadcast_to(self.cls_token.value, (B, 1, C))
        for blk in self.cls_blocks:
            cls = blk(cls, x)
        return self.norm(cls)[:, 0]

    def forward_head(self, x):
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "volo_d1_224": dict(layers=(4, 4, 8, 2), embed_dims=(192, 384, 384, 384), num_heads=(6, 12, 12, 12), mlp_ratio=3, stem_hidden_dim=64),
    "volo_d2_224": dict(layers=(6, 4, 10, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), mlp_ratio=3, stem_hidden_dim=64),
    "volo_d3_224": dict(layers=(8, 8, 16, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), mlp_ratio=3, stem_hidden_dim=64),
    "volo_d4_224": dict(layers=(8, 8, 16, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16), mlp_ratio=3, stem_hidden_dim=64),
    "volo_d5_224": dict(layers=(12, 12, 20, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16), mlp_ratio=4, stem_hidden_dim=128),
}

def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = VOLO(**dict(cfg, **kwargs))
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
