"""DLA (Deep Layer Aggregation) in flax nnx, NHWC. Mirrors timm.models.dla."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, global_pool_nhwc
from ..registry import register_model, _cfg


class DlaBasic(nnx.Module):
    def __init__(self, in_chs, out_chs, stride=1, *, rngs):
        self.conv1 = ConvBNAct(in_chs, out_chs, 3, stride, rngs=rngs)
        self.conv2 = ConvBNAct(out_chs, out_chs, 3, act="identity", rngs=rngs)

    def __call__(self, x, shortcut=None):
        sc = x if shortcut is None else shortcut
        y = self.conv2(self.conv1(x))
        return nnx.relu(y + sc)


class DlaBottleneck(nnx.Module):
    expansion = 2

    def __init__(self, in_chs, out_chs, stride=1, *, rngs):
        mid = out_chs // self.expansion
        self.conv1 = ConvBNAct(in_chs, mid, 1, rngs=rngs)
        self.conv2 = ConvBNAct(mid, mid, 3, stride, rngs=rngs)
        self.conv3 = ConvBNAct(mid, out_chs, 1, act="identity", rngs=rngs)
        self.short_proj = ConvBNAct(in_chs, out_chs, 1, act="identity", rngs=rngs) \
            if in_chs != out_chs else None

    def __call__(self, x, shortcut=None):
        if shortcut is None:
            shortcut = x if self.short_proj is None else self.short_proj(x)
        y = self.conv3(self.conv2(self.conv1(x)))
        return nnx.relu(y + shortcut)


class DlaRoot(nnx.Module):
    def __init__(self, in_chs, out_chs, shortcut=False, *, rngs):
        self.conv = ConvBNAct(in_chs, out_chs, 1, act="identity", rngs=rngs)
        self.shortcut = shortcut

    def __call__(self, x_children):
        y = self.conv(jnp.concatenate(x_children, axis=-1))
        if self.shortcut:
            y = y + x_children[0]
        return nnx.relu(y)


class DlaTree(nnx.Module):
    def __init__(self, levels, block, in_chs, out_chs, stride=1, root_dim=0,
                 root_shortcut=False, *, rngs):
        self.levels = levels
        if root_dim == 0:
            root_dim = 2 * out_chs
        self.down_stride = stride
        root = None
        project = None
        if levels == 1:
            self.tree1 = block(in_chs, out_chs, stride, rngs=rngs)
            self.tree2 = block(out_chs, out_chs, 1, rngs=rngs)
            project = ConvBNAct(in_chs, out_chs, 1, act="identity", rngs=rngs) \
                if in_chs != out_chs else None
            root = DlaRoot(root_dim, out_chs, root_shortcut, rngs=rngs)
        else:
            self.tree1 = DlaTree(levels - 1, block, in_chs, out_chs, stride, 0,
                                 root_shortcut, rngs=rngs)
            self.tree2 = DlaTree(levels - 1, block, out_chs, out_chs, 1,
                                 root_dim + out_chs, root_shortcut, rngs=rngs)
        self.project = project
        self.root = root

    def __call__(self, x, shortcut=None, children=None):
        children = [] if children is None else list(children)
        shortcut = x
        if self.down_stride > 1:
            shortcut = nnx.max_pool(x, (self.down_stride, self.down_stride),
                                    strides=(self.down_stride, self.down_stride))
        if self.root is not None:
            if self.project is not None:
                shortcut = self.project(shortcut)
            x1 = self.tree1(x, shortcut)
            x2 = self.tree2(x1)
            return self.root([x2, x1] + children)
        x1 = self.tree1(x)
        children.append(x1)
        return self.tree2(x1, children=children)


class DLA(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, levels, channels, block, root_shortcut=False, num_classes=1000,
                 in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.base_layer = ConvBNAct(in_chans, channels[0], 7, 1, rngs=rngs)
        self.level0 = ConvBNAct(channels[0], channels[0], 3, 1, rngs=rngs)
        self.level1 = ConvBNAct(channels[0], channels[1], 3, 2, rngs=rngs)
        self.level2 = DlaTree(levels[2], block, channels[1], channels[2], 2,
                              root_shortcut=root_shortcut, rngs=rngs)
        self.level3 = DlaTree(levels[3], block, channels[2], channels[3], 2,
                              root_shortcut=root_shortcut, rngs=rngs)
        self.level4 = DlaTree(levels[4], block, channels[3], channels[4], 2,
                              root_shortcut=root_shortcut, rngs=rngs)
        self.level5 = DlaTree(levels[5], block, channels[4], channels[5], 2,
                              root_shortcut=root_shortcut, rngs=rngs)
        self.num_features = channels[5]
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(channels[5], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.level1(self.level0(self.base_layer(x)))
        return self.level5(self.level4(self.level3(self.level2(x))))

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes, self.global_pool = num_classes, global_pool
        if num_classes > 0 and self.fc is None:
            raise RuntimeError("cannot re-add classifier to a num_classes=0 model")
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0)) if num_classes > 0 else None

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _dla(levels, channels, block, root_shortcut=False, **kwargs):
    model = DLA(levels, channels, block, root_shortcut, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def dla34(**kwargs):
    return _dla([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], DlaBasic, **kwargs)


@register_model
def dla60(**kwargs):
    return _dla([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], DlaBottleneck, **kwargs)


@register_model
def dla102(**kwargs):
    return _dla([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024], DlaBottleneck,
                root_shortcut=True, **kwargs)


@register_model
def dla169(**kwargs):
    return _dla([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024], DlaBottleneck,
                root_shortcut=True, **kwargs)
