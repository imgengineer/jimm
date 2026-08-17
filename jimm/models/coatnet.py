"""CoAtNet in flax nnx, NHWC. Mirrors timm.models.maxxvit-style coatnet (MBConv + rel-pos attention stages)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg

class MBConvBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, expand=4, *, rngs):
        mid = in_chs * expand
        self.conv1 = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.dw = nnx.Conv(mid, mid, (3, 3), strides=(stride, stride), use_bias=False,
                           feature_group_count=mid, rngs=rngs)
        self.bn2 = nnx.BatchNorm(mid, rngs=rngs)
        self.se = SqueezeExcite(mid, 0.25, rngs=rngs)
        self.pw = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_chs, rngs=rngs)
        self.shortcut = nnx.Conv(in_chs, out_chs, (1, 1), strides=(stride, stride), rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = nnx.gelu(self.bn1(self.conv1(x)))
        y = nnx.gelu(self.bn2(self.dw(y)))
        y = self.se(y)
        y = self.bn3(self.pw(y))
        sc = x if self.shortcut is None else self.shortcut(x)
        return y + sc

class RelPosAttention(nnx.Module):
    """Window-free global attention with 2D relative position bias (coatnet)."""

    def __init__(self, dim, num_heads, grid_size, qkv_bias=True, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.grid = grid_size
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        gh, gw = grid_size
        self.rel_bias = nnx.Param(jnp.zeros(((2 * gh - 1) * (2 * gw - 1), num_heads)))
        coords_h = jnp.arange(gh)
        coords_w = jnp.arange(gw)
        rel_h = coords_h[:, None, None, None] - coords_h[None, :, None, None] + gh - 1
        rel_w = coords_w[None, None, :, None] - coords_w[None, None, None, :] + gw - 1
        self.rel_index = (rel_h * (2 * gw - 1) + rel_w).reshape(-1)  # broadcast over (h1,h2,w1,w2)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(0, 1, 3, 2) * self.scale
        gh, gw = self.grid
        idx = self.rel_index.reshape(gh, gh, gw, gw).transpose(0, 2, 1, 3).reshape(N, N)
        attn = attn + self.rel_bias[...][idx].transpose(2, 0, 1)[None]
        attn = nnx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)

class AttnBlock(nnx.Module):
    def __init__(self, dim, num_heads, grid_size, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = RelPosAttention(dim, num_heads, grid_size, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, dim * 4, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, t):
        t = t + self.drop_path(self.attn(self.norm1(t)))
        return t + self.drop_path(self.mlp(self.norm2(t)))

class AttnStage(nnx.Module):
    def __init__(self, in_chs, out_chs, depth, num_heads, grid_size, drop_path=0.0, *, rngs):
        self.downsample = nnx.Sequential(
            lambda x: nnx.max_pool(x, (2, 2), strides=(2, 2), padding="SAME"),
            nnx.Conv(in_chs, out_chs, (1, 1), rngs=rngs)) if in_chs != out_chs else None
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([AttnBlock(out_chs, num_heads, grid_size, dpr[i], rngs=rngs)
                                for i in range(depth)])

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        B, H, W, C = x.shape
        t = x.reshape(B, H * W, C)
        for blk in self.blocks:
            t = blk(t)
        return t.reshape(B, H, W, C)

class CoAtNet(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(96, 192, 384, 768), blocks=(2, 2, 3, 5, 2),
                 head_dim=32, img_size=224, num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.List([nnx.Conv(in_chans, channels[0], (3, 3), strides=(2, 2), rngs=rngs),
                              nnx.Conv(channels[0], channels[0], (3, 3), rngs=rngs)])
        grid = img_size // 4
        # conv stages 0-1 (MBConv), attention stages 2-4
        conv_stages, chs = [], channels[0]
        for si in range(2):
            out = channels[si]
            stage = []
            for j in range(blocks[si]):
                stage.append(MBConvBlock(chs, out, 2 if j == 0 and si > 0 else 1, rngs=rngs))
                chs = out
            conv_stages.append(nnx.List(stage))
        self.conv_stages = nnx.List(conv_stages)
        # attn stages 2-4: resolutions 28, 14, 14 for img 224 (stage 4 keeps channels + grid)
        attn_stages = []
        chs = channels[1]
        cur_grid = img_size // 4  # after stage 1 (stem /2, stage1 /2)
        for si in range(2, 5):
            out = channels[si] if si < len(channels) else channels[-1]
            down = out != chs
            if down:
                cur_grid //= 2
            attn_stages.append(AttnStage(chs, out, blocks[si], max(out // head_dim, 1),
                                         (cur_grid, cur_grid), drop_path_rate, rngs=rngs))
            chs = out
        self.attn_stages = nnx.List(attn_stages)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.conv_stages:
            for blk in stage:
                x = blk(x)
        for stage in self.attn_stages:
            x = stage(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {  # channels (s0..s4), blocks
    "coatnet_0_rw_224": ((96, 192, 384, 768), (2, 2, 3, 5, 2)),
    "coatnet_1_rw_224": ((96, 192, 384, 768), (2, 2, 6, 14, 2)),
    "coatnet_2_rw_224": ((128, 256, 512, 1024), (2, 2, 6, 14, 2)),
}

def _make(name):
    channels, blocks = _CFGS[name]

    def entry(**kwargs):
        model = CoAtNet(channels, blocks, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
