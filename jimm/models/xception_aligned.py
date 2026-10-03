"""Aligned Xception in flax nnx, NHWC. Mirrors timm.models.xception_aligned.

A two-conv stem feeds Xception modules of three separable convolutions (3x3
depthwise + BatchNorm, 1x1 pointwise + BatchNorm), each preceded by ReLU, with
a 1x1 projection shortcut. The last separable convolution of a module carries
its stride instead of max pooling. The final module has no shortcut and
applies ReLU after every BatchNorm instead. The ``p`` variants use
pre-activation modules and end with a ReLU.
"""

from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


def _dw_conv(chs, stride, *, rngs):
    return nnx.Conv(
        chs,
        chs,
        (3, 3),
        strides=(stride, stride),
        padding=((1, 1), (1, 1)),
        feature_group_count=chs,
        use_bias=False,
        rngs=rngs,
    )


class SeparableConv2d(nnx.Module):
    def __init__(self, in_chs, out_chs, stride=1, act=None, eps=1e-3, *, rngs):
        self.conv_dw = _dw_conv(in_chs, stride, rngs=rngs)
        self.bn_dw = BatchNorm(in_chs, epsilon=eps, rngs=rngs)
        self.conv_pw = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn_pw = BatchNorm(out_chs, epsilon=eps, rngs=rngs)
        self.act = act

    def __call__(self, x):
        x = self.bn_dw(self.conv_dw(x))
        if self.act is not None:
            x = self.act(x)
        x = self.bn_pw(self.conv_pw(x))
        return self.act(x) if self.act is not None else x


class PreSeparableConv2d(nnx.Module):
    def __init__(self, in_chs, out_chs, stride=1, first_act=True, eps=1e-3, *, rngs):
        self.norm = BatchNorm(in_chs, epsilon=eps, rngs=rngs) if first_act else None
        self.conv_dw = _dw_conv(in_chs, stride, rngs=rngs)
        self.conv_pw = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)

    def __call__(self, x):
        if self.norm is not None:
            x = nnx.relu(self.norm(x))
        return self.conv_pw(self.conv_dw(x))


def _widths(out_chs):
    return out_chs if isinstance(out_chs, tuple) else (out_chs,) * 3


class XceptionModule(nnx.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=1,
        start_with_relu=True,
        no_skip=False,
        drop_path=0.0,
        eps=1e-3,
        *,
        rngs,
    ):
        out_chs = _widths(out_chs)
        self.no_skip, self.start_with_relu = no_skip, start_with_relu
        self.shortcut = (
            ConvNormAct(in_chs, out_chs[-1], 1, stride, eps=eps, rngs=rngs)
            if not no_skip and (out_chs[-1] != in_chs or stride != 1)
            else None
        )
        act = None if start_with_relu else nnx.relu
        convs = []
        for i, chs in enumerate(out_chs):
            convs.append(SeparableConv2d(in_chs, chs, stride if i == 2 else 1, act, eps, rngs=rngs))
            in_chs = chs
        self.stack = nnx.List(convs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        skip = x
        for conv in self.stack:
            x = conv(nnx.relu(x) if self.start_with_relu else x)
        if self.no_skip:
            return x
        if self.shortcut is not None:
            skip = self.shortcut(skip)
        return self.drop_path(x) + skip


class PreXceptionModule(nnx.Module):
    def __init__(self, in_chs, out_chs, stride=1, no_skip=False, drop_path=0.0, eps=1e-3, *, rngs):
        out_chs = _widths(out_chs)
        self.no_skip = no_skip
        self.shortcut = (
            nnx.Conv(
                in_chs, out_chs[-1], (1, 1), strides=(stride, stride), use_bias=False, rngs=rngs
            )
            if not no_skip and (out_chs[-1] != in_chs or stride != 1)
            else None
        )
        self.norm = BatchNorm(in_chs, epsilon=eps, rngs=rngs)
        convs = []
        for i, chs in enumerate(out_chs):
            convs.append(
                PreSeparableConv2d(in_chs, chs, stride if i == 2 else 1, i > 0, eps, rngs=rngs)
            )
            in_chs = chs
        self.stack = nnx.List(convs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = skip = nnx.relu(self.norm(x))
        for conv in self.stack:
            x = conv(x)
        if self.no_skip:
            return x
        if self.shortcut is not None:
            skip = self.shortcut(skip)
        return self.drop_path(x) + skip


class XceptionAligned(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        block_cfg,
        preact=False,
        eps=1e-3,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.preact = preact
        self.stem = nnx.List(
            [
                ConvNormAct(in_chans, 32, 3, 2, act=nnx.relu, eps=eps, rngs=rngs),
                # Pre-activation models normalize inside the first module instead.
                ConvNormAct(
                    32, 64, 3, norm=not preact, act=None if preact else nnx.relu, eps=eps, rngs=rngs
                ),
            ]
        )
        module = PreXceptionModule if preact else XceptionModule
        blocks = []
        for i, cfg in enumerate(block_cfg):
            dpr = drop_path_rate * i / max(len(block_cfg) - 1, 1)
            blocks.append(module(**cfg, drop_path=dpr, eps=eps, rngs=rngs))
        self.blocks = nnx.List(blocks)
        self.num_features = _widths(block_cfg[-1]["out_chs"])[-1]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for blk in self.blocks:
            x = blk(x)
        return nnx.relu(x) if self.preact else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _block_cfg(entry, middle, preact=False):
    exit_flow = dict(in_chs=1024, out_chs=(1536, 1536, 2048), stride=1, no_skip=True)
    if not preact:
        exit_flow["start_with_relu"] = False
    return [
        *(dict(in_chs=i, out_chs=o, stride=s) for i, o, s in entry),
        *(dict(in_chs=728, out_chs=728, stride=1) for _ in range(middle)),
        dict(in_chs=728, out_chs=(728, 1024, 1024), stride=2),
        exit_flow,
    ]


_ENTRY = ((64, 128, 2), (128, 256, 2), (256, 728, 2))
_ENTRY71 = ((64, 128, 2), (128, 256, 1), (256, 256, 2), (256, 728, 1), (728, 728, 2))
_CFGS = {  # entry flow, middle blocks, pre-activation, BatchNorm epsilon, crop_pct
    "xception41": (_ENTRY, 8, False, 1e-3, 0.903),
    "xception65": (_ENTRY, 16, False, 1e-3, 0.94),
    "xception71": (_ENTRY71, 16, False, 1e-3, 0.903),
    "xception41p": (_ENTRY, 8, True, 1e-5, 0.94),
    "xception65p": (_ENTRY, 16, True, 1e-3, 0.94),
}


def _make(name):
    entry_flow, middle, preact, eps, crop_pct = _CFGS[name]

    def entry(**kwargs):
        model = XceptionAligned(_block_cfg(entry_flow, middle, preact), preact, eps, **kwargs)
        model.default_cfg = _cfg(
            input_size=(3, 299, 299),
            crop_pct=crop_pct,
            interpolation="bicubic",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
