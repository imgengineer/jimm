"""MaxViT in flax nnx, NHWC. Mirrors timm.models.maxxvit (MBConv + window/grid attention)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg
from .swin_transformer import window_partition, window_reverse

class MaxViTMBConv(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, expand=4, drop_path=0.0, *, rngs):
        mid = in_chs * expand
        self.conv1 = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.dw = nnx.Conv(mid, mid, (3, 3), strides=(stride, stride), use_bias=False,
                           feature_group_count=mid, rngs=rngs)
        self.bn2 = nnx.BatchNorm(mid, rngs=rngs)
        self.se = SqueezeExcite(mid, 0.25, rngs=rngs)
        self.pw = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_chs, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.shortcut = nnx.Conv(in_chs, out_chs, (1, 1), strides=(stride, stride), rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = nnx.gelu(self.bn1(self.conv1(x)))
        y = nnx.gelu(self.bn2(self.dw(y)))
        y = self.se(y)
        y = self.bn3(self.pw(y))
        sc = x if self.shortcut is None else self.shortcut(x)
        return self.drop_path(y) + sc

class MaxViTAttention(nnx.Module):
    """Window/grid attention with relative position bias (proper q,k,v)."""

    def __init__(self, dim, num_heads, window_size, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ws = window_size
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        n = (2 * window_size - 1) ** 2
        self.rel_bias = nnx.Param(jnp.zeros((n, num_heads)))
        coords = jnp.stack(jnp.meshgrid(jnp.arange(window_size), jnp.arange(window_size), indexing="ij"))
        cf = coords.reshape(2, -1)
        rel = (cf[:, :, None] - cf[:, None, :]).transpose(1, 2, 0) + window_size - 1
        self.rel_index = rel[:, :, 0] * (2 * window_size - 1) + rel[:, :, 1]

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(0, 1, 3, 2) * self.scale
        attn = attn + self.rel_bias.value[self.rel_index].transpose(2, 0, 1)[None]
        attn = nnx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)

class MaxViTBlock(nnx.Module):
    """Block attention (window or grid) + FFN."""

    def __init__(self, dim, num_heads, window_size, is_grid, drop_path=0.0, *, rngs):
        self.is_grid = is_grid
        self.ws = window_size
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = MaxViTAttention(dim, num_heads, window_size, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, dim * 4, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        ws = self.ws
        if self.is_grid:
            # grid attention: tokens are strided (dilated) — transpose windows so each
            # "window" contains every ws-th pixel
            t = x.reshape(B, ws, H // ws, ws, W // ws, C).transpose(0, 2, 4, 1, 3, 5)
            t = t.reshape(-1, ws * ws, C)
        else:
            t = window_partition(x, ws)
        t = t + self.drop_path(self.attn(self.norm1(t)))
        t = t + self.drop_path(self.mlp(self.norm2(t)))
        if self.is_grid:
            t = t.reshape(B, H // ws, W // ws, ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
            x = t.reshape(B, H, W, C)
        else:
            x = window_reverse(t, ws, H, W, B)
        return x

class MaxViTStage(nnx.Module):
    def __init__(self, in_chs, out_chs, depth, num_heads, window_size, stride,
                 drop_path=0.0, *, rngs):
        blocks = []
        for i in range(depth):
            blocks.append(MaxViTMBConv(in_chs if i == 0 else out_chs, out_chs,
                                       stride if i == 0 else 1, rngs=rngs))
            blocks.append(MaxViTBlock(out_chs, num_heads, window_size, is_grid=False, rngs=rngs))
            blocks.append(MaxViTBlock(out_chs, num_heads, window_size, is_grid=True, rngs=rngs))
        self.blocks = nnx.List(blocks)

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class MaxViT(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(96, 192, 384, 768), depths=(2, 2, 5, 2), head_dim=32,
                 window_size=7, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.List([nnx.Conv(in_chans, channels[0] // 2, (3, 3), strides=(2, 2), rngs=rngs),
                              nnx.Conv(channels[0] // 2, channels[0], (3, 3), rngs=rngs)])
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, chs, k = [], channels[0], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            stages.append(MaxViTStage(chs, c, d, max(c // head_dim, 1), window_size,
                                      1 if i == 0 else 2, dpr[k], rngs=rngs))
            chs = c
            k += d
        self.stages = nnx.List(stages)
        self.norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "maxvit_tiny_rw_224": ((96, 192, 384, 768), (2, 2, 5, 2)),
    "maxvit_small_rw_224": ((96, 192, 384, 768), (2, 2, 13, 2)),
    "maxvit_base_rw_224": ((96, 192, 384, 768), (2, 6, 14, 2)),
}

def _make(name):
    channels, depths = _CFGS[name]

    def entry(**kwargs):
        model = MaxViT(channels, depths, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
