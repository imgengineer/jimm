"""MobileViT in flax nnx, NHWC. Mirrors timm.models.mobilevit (conv blocks + transformer stages)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, DropPath, ClassifierMixin
from ..registry import register_model, _cfg
from .mobilenetv2 import InvertedResidual
from .vision_transformer import Attention

class MViTTransformerBlock(nnx.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.fc2(nnx.gelu(self.fc1(self.norm2(x)))))

class MobileViTStage(nnx.Module):
    """conv to dim -> unfold patches -> transformer -> fold back -> fuse."""

    def __init__(self, in_chs, out_chs, depth, patch=2, num_heads=4, *, rngs):
        self.conv_in = ConvBNAct(in_chs, out_chs, 1, act="silu", rngs=rngs)
        self.patch = patch
        self.blocks = nnx.List([MViTTransformerBlock(out_chs, num_heads, rngs=rngs)
                                for _ in range(depth)])
        self.conv_out = ConvBNAct(out_chs, out_chs, 3, act="silu", rngs=rngs)
        self.fuse = ConvBNAct(in_chs + out_chs, out_chs, 3, act="silu", rngs=rngs)

    def __call__(self, x):
        y = self.conv_in(x)
        B, H, W, C = y.shape
        p = self.patch
        t = y.reshape(B, H // p, p, W // p, p, C).transpose(0, 1, 3, 2, 4, 5)
        t = t.reshape(-1, p * p, C)
        for blk in self.blocks:
            t = blk(t)
        y = t.reshape(B, H // p, W // p, p, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        y = self.conv_out(y)
        return self.fuse(jnp.concatenate([x, y], axis=-1))

class MobileViT(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(32, 64, 96), tf_dims=(144, 192, 240), tf_depths=(2, 4, 3),
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = tf_dims[-1]
        self.stem = ConvBNAct(in_chans, channels[0], 3, 2, act="silu", rngs=rngs)
        self.stage1 = nnx.List([InvertedResidual(channels[0], channels[0], 1, 1, rngs=rngs),
                                InvertedResidual(channels[0], channels[1], 2, 4, rngs=rngs)])
        self.stage2 = nnx.List([InvertedResidual(channels[1], channels[2], 1, 4, rngs=rngs),
                                InvertedResidual(channels[2], channels[2], 1, 4, rngs=rngs)])
        self.mv2_3 = InvertedResidual(channels[2], tf_dims[0], 2, 4, rngs=rngs)
        self.tf1 = MobileViTStage(tf_dims[0], tf_dims[0], tf_depths[0], 2, 4, rngs=rngs)
        self.mv2_4 = InvertedResidual(tf_dims[0], tf_dims[1], 2, 4, rngs=rngs)
        self.tf2 = MobileViTStage(tf_dims[1], tf_dims[1], tf_depths[1], 2, 4, rngs=rngs)
        self.mv2_5 = InvertedResidual(tf_dims[1], tf_dims[2], 2, 4, rngs=rngs)
        self.tf3 = MobileViTStage(tf_dims[2], tf_dims[2], tf_depths[2], 2, 4, rngs=rngs)
        self.conv_head = ConvBNAct(tf_dims[2], 4 * tf_dims[2], 1, act="silu", rngs=rngs)
        self.num_features = 4 * tf_dims[2]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for blk in self.stage1:
            x = blk(x)
        for blk in self.stage2:
            x = blk(x)
        x = self.tf1(self.mv2_3(x))
        x = self.tf2(self.mv2_4(x))
        x = self.tf3(self.mv2_5(x))
        return self.conv_head(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "mobilevit_xxs": ((32, 64, 80), (96, 120, 144), (2, 4, 3)),
    "mobilevit_xs": ((48, 96, 120), (144, 160, 192), (2, 4, 3)),
    "mobilevit_s": ((64, 128, 160), (192, 256, 320), (2, 6, 4)),
}

def _make(name):
    channels, tf_dims, tf_depths = _CFGS[name]

    def entry(**kwargs):
        model = MobileViT(channels, tf_dims, tf_depths, **kwargs)
        model.default_cfg = _cfg(input_size=(3, 256, 256))
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
