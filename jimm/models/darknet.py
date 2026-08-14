"""DarkNet53 + CSPDarkNet53 + CSPResNet50/CSPResNeXt50 in flax nnx, NHWC. Mirrors timm.models.cspnet."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, ClassifierMixin
from ..registry import register_model, _cfg

class ResBlock(nnx.Module):
    def __init__(self, chs, *, rngs):
        self.conv1 = ConvBNAct(chs, chs // 2, 1, act="silu", rngs=rngs)
        self.conv2 = ConvBNAct(chs // 2, chs, 3, act="silu", rngs=rngs)

    def __call__(self, x):
        return x + self.conv2(self.conv1(x))

class CSPStage(nnx.Module):
    """CSP stage: split into two paths, one through blocks, merge at output."""

    def __init__(self, in_chs, out_chs, num_blocks, bottleneck=True, *, rngs):
        self.down = ConvBNAct(in_chs, out_chs, 3, 2, act="silu", rngs=rngs)
        half = out_chs // 2
        self.blocks = nnx.List([ResBlock(half, rngs=rngs) for _ in range(num_blocks)])
        self.conv_t1 = ConvBNAct(half, half, 1, act="silu", rngs=rngs)
        self.conv_t2 = ConvBNAct(half, half, 1, act="silu", rngs=rngs)
        self.conv_out = ConvBNAct(out_chs, out_chs, 1, act="silu", rngs=rngs)

    def __call__(self, x):
        x = self.down(x)
        x1, x2 = jnp.split(x, 2, axis=-1)
        for blk in self.blocks:
            x1 = blk(x1)
        return self.conv_out(jnp.concatenate([self.conv_t1(x1), self.conv_t2(x2)], axis=-1))

class CSPDarkNet(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(64, 128, 256, 512, 1024), blocks=(2, 4, 8, 4),
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = ConvBNAct(in_chans, 32, 3, act="silu", rngs=rngs)
        self.stages = nnx.List([CSPStage(32 if i == 0 else channels[i - 1], c, n, rngs=rngs)
                                for i, (c, n) in enumerate(zip(channels, blocks))])
        self.num_features = channels[-1]
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

class DarkNetResBlock(nnx.Module):
    def __init__(self, chs, *, rngs):
        self.conv1 = ConvBNAct(chs, chs // 2, 1, rngs=rngs)
        self.conv2 = ConvBNAct(chs // 2, chs, 3, rngs=rngs)

    def __call__(self, x):
        return x + self.conv2(self.conv1(x))

class DarkNet53(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = ConvBNAct(in_chans, 32, 3, rngs=rngs)
        cfg = [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)]
        stages, chs = [], 32
        for out, n in cfg:
            layers = [ConvBNAct(chs, out, 3, 2, rngs=rngs)]
            layers += [DarkNetResBlock(out, rngs=rngs) for _ in range(n)]
            stages.append(nnx.List(layers))
            chs = out
        self.stages = nnx.List(stages)
        self.num_features = 1024
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(1024, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for layer in stage:
                x = layer(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _csp(channels, blocks, **kwargs):
    model = CSPDarkNet(channels, blocks, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def darknet53(**kwargs):
    model = DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model

@register_model
def cspdarknet53(**kwargs):
    return _csp((64, 128, 256, 512, 1024), (1, 2, 8, 8, 4), **kwargs)  # YOLOv4 block counts

@register_model
def cspresnet50(**kwargs):
    return _csp((128, 256, 512, 1024), (3, 4, 6, 3), **kwargs)

@register_model
def cspresnext50(**kwargs):
    return _csp((256, 512, 1024, 2048), (3, 4, 6, 3), **kwargs)
