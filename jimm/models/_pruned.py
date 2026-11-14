"""Pruned EfficientNet and ECA-ResNet models, built from timm's shape specifications.

timm prunes a trained model by resizing every convolution and normalization to
the widths recorded in ``_pruned/*.txt``. The same specifications are parsed
here to build the equivalent networks directly.
"""

from pathlib import Path

import jax
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, SqueezeExcite


def _read_spec(name):
    path = Path(__file__).with_name("_pruned") / f"{name}.txt"
    spec = {}
    for part in path.read_text().strip().split("***"):
        key, shape = part.split(":")
        spec[key] = [int(x) for x in shape.strip("[]").split(",") if x]
    return spec


# ----------------------------------------------------------------------------- EfficientNet


class _DSConv(nnx.Module):
    """Depthwise separable block, optionally residual."""

    def __init__(self, in_chs, out_chs, kernel, stride, se_rd, residual, *, rngs):
        self.residual = residual
        self.dw = nnx.Conv(
            in_chs,
            in_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding="SAME",
            use_bias=False,
            feature_group_count=in_chs,
            rngs=rngs,
        )
        self.bn1 = BatchNorm(in_chs, epsilon=1e-3, rngs=rngs)
        self.se = (
            SqueezeExcite(in_chs, rd_channels=se_rd, act=nnx.silu, rngs=rngs) if se_rd else None
        )
        self.pw = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = BatchNorm(out_chs, epsilon=1e-3, rngs=rngs)

    def __call__(self, x):
        y = nnx.silu(self.bn1(self.dw(x)))
        if self.se is not None:
            y = self.se(y)
        y = self.bn2(self.pw(y))
        return x + y if self.residual else y


class _InvRes(nnx.Module):
    def __init__(self, in_chs, mid_chs, out_chs, kernel, stride, se_rd, residual, *, rngs):
        self.residual = residual
        self.pw = nnx.Conv(in_chs, mid_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = BatchNorm(mid_chs, epsilon=1e-3, rngs=rngs)
        self.dw = nnx.Conv(
            mid_chs,
            mid_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding="SAME",
            use_bias=False,
            feature_group_count=mid_chs,
            rngs=rngs,
        )
        self.bn2 = BatchNorm(mid_chs, epsilon=1e-3, rngs=rngs)
        self.se = (
            SqueezeExcite(mid_chs, rd_channels=se_rd, act=nnx.silu, rngs=rngs) if se_rd else None
        )
        self.pwl = nnx.Conv(mid_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = BatchNorm(out_chs, epsilon=1e-3, rngs=rngs)

    def __call__(self, x):
        y = nnx.silu(self.bn1(self.pw(x)))
        y = nnx.silu(self.bn2(self.dw(y)))
        if self.se is not None:
            y = self.se(y)
        y = self.bn3(self.pwl(y))
        return x + y if self.residual else y


class PrunedEfficientNet(ClassifierMixin, nnx.Module):
    def __init__(
        self, variant, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.2, *, rngs
    ):
        d = _read_spec(variant)
        stem = d["conv_stem.weight"][0]
        self.num_classes, self.global_pool = num_classes, global_pool
        self.conv_stem = nnx.Conv(
            in_chans, stem, (3, 3), strides=(2, 2), padding="SAME", use_bias=False, rngs=rngs
        )
        self.bn1 = BatchNorm(stem, epsilon=1e-3, rngs=rngs)
        blocks = {}
        for key, shape in d.items():
            parts = key.split(".")
            if parts[0] == "blocks":
                blocks.setdefault((int(parts[1]), int(parts[2])), {})[".".join(parts[3:])] = shape
        layer_blocks, prev = [], stem
        for stage in range(7):
            idx = 0
            while (stage, idx) in blocks:
                b = blocks[(stage, idx)]
                stride = 1 if stage == 0 else (2 if idx == 0 else 1)
                if "conv_pwl.weight" in b:
                    mid = b["conv_pw.weight"][0]
                    out = b["conv_pwl.weight"][0]
                    kernel = b["conv_dw.weight"][2]
                    residual = prev == out and stride == 1
                    layer_blocks.append(
                        _InvRes(
                            prev,
                            mid,
                            out,
                            kernel,
                            stride,
                            b["se.conv_reduce.weight"][0],
                            residual,
                            rngs=rngs,
                        )
                    )
                else:
                    out = b["conv_pw.weight"][0]
                    kernel = b["conv_dw.weight"][2]
                    residual = prev == out and stride == 1
                    layer_blocks.append(
                        _DSConv(
                            prev,
                            out,
                            kernel,
                            stride,
                            b["se.conv_reduce.weight"][0],
                            residual,
                            rngs=rngs,
                        )
                    )
                prev = out
                idx += 1
        self.blocks = nnx.List(layer_blocks)
        head = d["conv_head.weight"][0]
        self.conv_head = nnx.Conv(prev, head, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = BatchNorm(head, epsilon=1e-3, rngs=rngs)
        self.num_features = d["classifier.weight"][1]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.silu(self.bn1(self.conv_stem(x)))
        for blk in self.blocks:
            x = blk(x)
        return nnx.silu(self.bn2(self.conv_head(x)))

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


# ----------------------------------------------------------------------------- ECA-ResNet


class _Eca(nnx.Module):
    def __init__(self, chs, kernel, *, rngs):
        self.conv = nnx.Conv(
            1, 1, (kernel,), padding=(((kernel - 1) // 2,) * 2,), use_bias=False, rngs=rngs
        )

    def __call__(self, x):
        s = x.mean(axis=(1, 2))[:, :, None]
        s = self.conv(s)[:, :, 0]
        return x * jax.nn.sigmoid(s)[:, None, None, :]


class _Downsample(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, *, rngs):
        self.avg_pool = stride == 2
        self.conv = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn = BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        if self.avg_pool:
            x = nnx.avg_pool(x, (2, 2), strides=(2, 2), padding="VALID")
        return self.bn(self.conv(x))


class _EcaBottleneck(nnx.Module):
    def __init__(self, in_chs, c1, c2, c3, stride, kernel, downsample, *, rngs):
        self.conv1 = nnx.Conv(
            in_chs, c1, (1, 1), strides=(stride, stride), use_bias=False, rngs=rngs
        )
        self.bn1 = BatchNorm(c1, rngs=rngs)
        self.conv2 = nnx.Conv(c1, c2, (3, 3), padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs)
        self.bn2 = BatchNorm(c2, rngs=rngs)
        self.conv3 = nnx.Conv(c2, c3, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = BatchNorm(c3, rngs=rngs)
        self.eca = _Eca(c3, kernel, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        y = nnx.relu(self.bn1(self.conv1(x)))
        y = nnx.relu(self.bn2(self.conv2(y)))
        y = self.eca(self.bn3(self.conv3(y)))
        return nnx.relu(y + shortcut)


class PrunedECAResNet(ClassifierMixin, nnx.Module):
    def __init__(
        self, variant, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs
    ):
        d = _read_spec(variant)
        self.num_classes, self.global_pool = num_classes, global_pool
        stem_out = d["bn1.weight"][0]
        self.stem_conv1 = nnx.Conv(
            in_chans, d["conv1.0.weight"][0], (3, 3), strides=(2, 2),
            padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs,
        )  # fmt: skip
        self.stem_bn1 = BatchNorm(d["conv1.0.weight"][0], rngs=rngs)
        self.stem_conv2 = nnx.Conv(
            d["conv1.3.weight"][1], d["conv1.3.weight"][0], (3, 3),
            padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs,
        )  # fmt: skip
        self.stem_bn2 = BatchNorm(d["conv1.3.weight"][0], rngs=rngs)
        self.stem_conv3 = nnx.Conv(
            d["conv1.6.weight"][1], d["conv1.6.weight"][0], (3, 3),
            padding=((1, 1), (1, 1)), use_bias=False, rngs=rngs,
        )  # fmt: skip
        self.stem_bn3 = BatchNorm(stem_out, rngs=rngs)
        blocks = {}
        for key, shape in d.items():
            parts = key.split(".")
            if len(parts) > 2 and parts[0].startswith("layer") and parts[1].isdigit():
                blocks.setdefault((int(parts[0][5:]), int(parts[1])), {})[".".join(parts[2:])] = (
                    shape
                )
        layers, prev = [], stem_out
        for layer in range(1, 5):
            idx, layer_blocks = 0, []
            while (layer, idx) in blocks:
                b = blocks[(layer, idx)]
                c1, c2, c3 = (
                    b["conv1.weight"][0],
                    b["conv2.weight"][0],
                    b["conv3.weight"][0],
                )
                stride = 2 if (layer > 1 and idx == 0) else 1
                downsample = None
                if stride != 1 or prev != c3:
                    downsample = _Downsample(prev, c3, stride, rngs=rngs)
                layer_blocks.append(
                    _EcaBottleneck(
                        prev, c1, c2, c3, stride, b["se.conv.weight"][-1], downsample, rngs=rngs
                    )
                )
                prev = c3
                idx += 1
            layers.append(nnx.List(layer_blocks))
        self.layers = nnx.List(layers)
        self.num_features = d["fc.weight"][1]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.stem_bn1(self.stem_conv1(x)))
        x = nnx.relu(self.stem_bn2(self.stem_conv2(x)))
        x = nnx.relu(self.stem_bn3(self.stem_conv3(x)))
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        for layer in self.layers:
            for blk in layer:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))
