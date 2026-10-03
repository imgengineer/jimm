"""CSPNet, DarkNet, and CS3 variants in flax nnx, NHWC. Mirrors timm.models.cspnet.

Cross-stage partial stages downsample with a strided 3x3 conv, expand with a
1x1 conv, and split the result: one half runs through residual blocks (ResNet
bottlenecks, DarkNet 1x1 + 3x3 blocks, or "edge" 3x3 + 1x1 blocks) and is
concatenated with the other before a 1x1 transition (CSP stages add a
transition on the block path first). DarkNet stages apply the blocks to the
whole downsampled map. All convolutions are followed by BatchNorm and the
model's activation (leaky ReLU, or SiLU for the CS3 models).
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, make_divisible
from ..registry import _cfg, register_model
from ._conv import ConvNormAct

_ACTS = {"leaky_relu": nnx.leaky_relu, "silu": nnx.silu}


@dataclass
class StageCfg:
    depth: tuple
    out_chs: tuple
    stride: tuple = (2,)
    groups: tuple = (1,)
    block_ratio: tuple = (1.0,)
    bottle_ratio: tuple = (1.0,)
    expand_ratio: tuple = (1.0,)
    avg_down: bool = False
    down_growth: bool = False
    cross_linear: bool = False
    attn_rd_ratio: float | None = None  # squeeze-excite reduction, None for no attention
    stage_type: str = "csp"
    block_type: str = "bottle"

    def per_stage(self, name, i):
        values = getattr(self, name)
        return values[min(i, len(values) - 1)]


@dataclass
class CspCfg:
    stem_chs: tuple
    stem_stride: int
    stages: StageCfg
    stem_kernel: int = 3
    stem_pad: int | None = None
    stem_pool: bool = False
    act: str = "leaky_relu"
    extra: dict = field(default_factory=dict)


class SEModule(nnx.Module):
    def __init__(self, chs, rd_ratio, act, *, rngs):
        rd = make_divisible(chs * rd_ratio, 8, round_limit=0.0)
        self.fc1 = nnx.Linear(chs, rd, rngs=rngs)
        self.fc2 = nnx.Linear(rd, chs, rngs=rngs)
        self.act = act

    def __call__(self, x):
        s = self.fc2(self.act(self.fc1(jnp.mean(x, axis=(1, 2), keepdims=True))))
        return x * nnx.sigmoid(s)


def _cna(in_chs, out_chs, kernel=1, stride=1, groups=1, act=None, bn_weight_init=1.0, *, rngs):
    return ConvNormAct(
        in_chs, out_chs, kernel, stride, groups, act=act, bn_weight_init=bn_weight_init, rngs=rngs
    )


class BottleneckBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, bottle_ratio, groups, act, attn, drop_path, *, rngs):
        mid = int(round(out_chs * bottle_ratio))
        self.act = act
        self.conv1 = _cna(in_chs, mid, act=act, rngs=rngs)
        self.conv2 = _cna(mid, mid, 3, groups=groups, act=act, rngs=rngs)
        self.attn2 = SEModule(mid, attn, act, rngs=rngs) if attn else None
        self.conv3 = _cna(mid, out_chs, bn_weight_init=0.0, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.conv2(self.conv1(x))
        if self.attn2 is not None:
            y = self.attn2(y)
        return self.act(self.drop_path(self.conv3(y)) + x)


class DarkBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, bottle_ratio, groups, act, attn, drop_path, *, rngs):
        mid = int(round(out_chs * bottle_ratio))
        self.conv1 = _cna(in_chs, mid, act=act, rngs=rngs)
        self.attn = SEModule(mid, attn, act, rngs=rngs) if attn else None
        self.conv2 = _cna(mid, out_chs, 3, groups=groups, act=act, bn_weight_init=0.0, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.conv1(x)
        if self.attn is not None:
            y = self.attn(y)
        return self.drop_path(self.conv2(y)) + x


class EdgeBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, bottle_ratio, groups, act, attn, drop_path, *, rngs):
        mid = int(round(out_chs * bottle_ratio))
        self.conv1 = _cna(in_chs, mid, 3, groups=groups, act=act, rngs=rngs)
        self.attn = SEModule(mid, attn, act, rngs=rngs) if attn else None
        self.conv2 = _cna(mid, out_chs, act=act, bn_weight_init=0.0, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.conv1(x)
        if self.attn is not None:
            y = self.attn(y)
        return self.drop_path(self.conv2(y)) + x


_BLOCKS = {"bottle": BottleneckBlock, "dark": DarkBlock, "edge": EdgeBlock}


class AvgDown(nnx.Module):
    """2x2 average pooling (when strided), then a 1x1 conv."""

    def __init__(self, in_chs, out_chs, stride, groups, act, *, rngs):
        self.stride = stride
        self.conv = _cna(in_chs, out_chs, groups=groups, act=act, rngs=rngs)

    def __call__(self, x):
        if self.stride == 2:
            x = nnx.avg_pool(x, (2, 2), strides=(2, 2))
        return self.conv(x)


class CspStage(nnx.Module):
    """``stage_type`` "csp" (CrossStage), "cs3" (CrossStage3), or "dark" (DarkStage)."""

    def __init__(self, in_chs, cfg, i, act, drop_paths, *, rngs):
        self.stage_type = stage_type = cfg.stage_type
        out_chs, stride = cfg.out_chs[i], cfg.per_stage("stride", i)
        groups = cfg.per_stage("groups", i)
        block = _BLOCKS[cfg.block_type]
        block_out = int(round(out_chs * cfg.per_stage("block_ratio", i)))
        down_chs = out_chs if cfg.down_growth or stage_type == "dark" else in_chs
        if stage_type == "dark" or stride != 1:
            if cfg.avg_down:
                self.conv_down = AvgDown(in_chs, out_chs, stride, groups, act, rngs=rngs)
            else:
                self.conv_down = _cna(in_chs, down_chs, 3, stride, groups, act, rngs=rngs)
            prev = down_chs
        else:
            self.conv_down = None
            prev = in_chs
        if stage_type != "dark":
            self.expand_chs = exp = int(round(out_chs * cfg.per_stage("expand_ratio", i)))
            self.conv_exp = _cna(prev, exp, act=None if cfg.cross_linear else act, rngs=rngs)
            prev = exp // 2
        blocks = []
        for j in range(cfg.depth[i]):
            blocks.append(
                block(
                    prev,
                    block_out,
                    cfg.per_stage("bottle_ratio", i),
                    groups,
                    act,
                    cfg.attn_rd_ratio,
                    drop_paths[j],
                    rngs=rngs,
                )
            )
            prev = block_out
        self.blocks = nnx.List(blocks)
        if stage_type == "csp":
            self.conv_transition_b = _cna(prev, exp // 2, act=act, rngs=rngs)
        if stage_type != "dark":
            self.conv_transition = _cna(exp, out_chs, act=act, rngs=rngs)

    def __call__(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        if self.stage_type == "dark":
            for blk in self.blocks:
                x = blk(x)
            return x
        x = self.conv_exp(x)
        half = self.expand_chs // 2
        xs, xb = x[..., :half], x[..., half:]
        if self.stage_type == "csp":
            for blk in self.blocks:
                xb = blk(xb)
            xb = self.conv_transition_b(xb)
            return self.conv_transition(jnp.concatenate([xs, xb], axis=-1))
        for blk in self.blocks:
            xs = blk(xs)
        return self.conv_transition(jnp.concatenate([xs, xb], axis=-1))


class CspNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        cfg,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        act = _ACTS[cfg.act]
        stem, prev, last = [], in_chans, len(cfg.stem_chs) - 1
        for i, chs in enumerate(cfg.stem_chs):
            stride = (
                2
                if (i == 0 and cfg.stem_stride > 1)
                or (i == last and cfg.stem_stride > 2 and not cfg.stem_pool)
                else 1
            )
            kernel = cfg.stem_kernel
            conv = _cna(prev, chs, kernel, stride, act=act, rngs=rngs)
            if i == 0 and cfg.stem_pad is not None:  # the focus stem pads its 6x6 conv by 2
                conv.conv.padding = ((cfg.stem_pad,) * 2,) * 2
            stem.append(conv)
            prev = chs
        self.stem = nnx.List(stem)
        self.stem_pool = cfg.stem_pool
        stages_cfg = cfg.stages
        depths = stages_cfg.depth
        total = sum(depths)
        rates = [drop_path_rate * k / max(total - 1, 1) for k in range(total)]
        stages = []
        for i in range(len(depths)):
            start = sum(depths[:i])
            stages.append(
                CspStage(prev, stages_cfg, i, act, rates[start : start + depths[i]], rngs=rngs)
            )
            prev = stages_cfg.out_chs[i]
        self.stages = nnx.List(stages)
        self.num_features = prev
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(prev, num_classes, kernel_init=nnx.initializers.normal(0.01), rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        for conv in self.stem:
            x = conv(x)
        if self.stem_pool:
            x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        for stage in self.stages:
            x = stage(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _cs3(width=1.0, depth=1.0, focus=False, se=None, bottle_ratio=1.0, block="dark"):
    w = lambda c: make_divisible(c * width)  # noqa: E731
    stem = (
        dict(stem_chs=(w(64),), stem_kernel=6, stem_pad=2)
        if focus
        else dict(stem_chs=(w(32), w(64)))
    )
    return CspCfg(
        **stem,
        stem_stride=2,
        act="silu",
        stages=StageCfg(
            depth=tuple(int(d * depth) for d in (3, 6, 9, 3)),
            out_chs=tuple(w(c) for c in (128, 256, 512, 1024)),
            bottle_ratio=(bottle_ratio,),
            block_ratio=(0.5,),
            attn_rd_ratio=se,
            stage_type="cs3",
            block_type=block,
        ),
    )


def _dark(depth, se=None, avg_down=False):
    return CspCfg(
        stem_chs=(32,),
        stem_stride=1,
        stages=StageCfg(
            depth=depth,
            out_chs=(64, 128, 256, 512, 1024),
            bottle_ratio=(0.5,),
            avg_down=avg_down,
            attn_rd_ratio=se,
            stage_type="dark",
            block_type="dark",
        ),
    )


_RESNET_STEM = dict(stem_chs=(64,), stem_stride=4, stem_kernel=7, stem_pool=True)
_DEEP_STEM = dict(stem_chs=(32, 32, 64), stem_stride=4, stem_pool=True)
_CFGS = {
    "cspresnet50": CspCfg(
        **_RESNET_STEM,
        stages=StageCfg(
            depth=(3, 3, 5, 2),
            out_chs=(128, 256, 512, 1024),
            stride=(1, 2),
            expand_ratio=(2.0,),
            bottle_ratio=(0.5,),
            cross_linear=True,
        ),
    ),
    "cspresnet50d": CspCfg(
        **_DEEP_STEM,
        stages=StageCfg(
            depth=(3, 3, 5, 2),
            out_chs=(128, 256, 512, 1024),
            stride=(1, 2),
            expand_ratio=(2.0,),
            bottle_ratio=(0.5,),
            cross_linear=True,
        ),
    ),
    "cspresnet50w": CspCfg(
        **_DEEP_STEM,
        stages=StageCfg(
            depth=(3, 3, 5, 2),
            out_chs=(256, 512, 1024, 2048),
            stride=(1, 2),
            bottle_ratio=(0.25,),
            block_ratio=(0.5,),
            cross_linear=True,
        ),
    ),
    "cspresnext50": CspCfg(
        **_RESNET_STEM,
        stages=StageCfg(
            depth=(3, 3, 5, 2),
            out_chs=(256, 512, 1024, 2048),
            stride=(1, 2),
            groups=(32,),
            block_ratio=(0.5,),
            cross_linear=True,
        ),
    ),
    "cspdarknet53": CspCfg(
        stem_chs=(32,),
        stem_stride=1,
        stages=StageCfg(
            depth=(1, 2, 8, 8, 4),
            out_chs=(64, 128, 256, 512, 1024),
            expand_ratio=(2.0, 1.0),
            bottle_ratio=(0.5, 1.0),
            block_ratio=(1.0, 0.5),
            down_growth=True,
            block_type="dark",
        ),
    ),
    "darknet17": _dark((1, 1, 1, 1, 1)),
    "darknet21": _dark((1, 1, 1, 2, 2)),
    "sedarknet21": _dark((1, 1, 1, 2, 2), se=1 / 16),
    "darknet53": _dark((1, 2, 8, 8, 4)),
    "darknetaa53": _dark((1, 2, 8, 8, 4), avg_down=True),
    "cs3darknet_s": _cs3(0.5, 0.5),
    "cs3darknet_m": _cs3(0.75, 0.67),
    "cs3darknet_l": _cs3(),
    "cs3darknet_x": _cs3(1.25, 1.33),
    "cs3darknet_focus_s": _cs3(0.5, 0.5, focus=True),
    "cs3darknet_focus_m": _cs3(0.75, 0.67, focus=True),
    "cs3darknet_focus_l": _cs3(focus=True),
    "cs3darknet_focus_x": _cs3(1.25, 1.33, focus=True),
    "cs3sedarknet_l": _cs3(se=0.25),
    "cs3sedarknet_x": _cs3(1.25, 1.33, se=1 / 16),
    "cs3sedarknet_xdw": CspCfg(
        stem_chs=(32, 64),
        stem_stride=2,
        act="silu",
        stages=StageCfg(
            depth=(3, 6, 12, 4),
            out_chs=(256, 512, 1024, 2048),
            groups=(1, 1, 256, 512),
            bottle_ratio=(0.5,),
            block_ratio=(0.5,),
            attn_rd_ratio=1 / 16,
        ),
    ),
    "cs3edgenet_x": _cs3(1.25, 1.33, bottle_ratio=1.5, block="edge"),
    "cs3se_edgenet_x": _cs3(1.25, 1.33, se=0.25, bottle_ratio=1.5, block="edge"),
}


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = CspNet(cfg, **kwargs)
        bicubic = name.startswith("cs3") or name == "darknet53"
        focus_s = name == "cs3darknet_focus_s"
        model.default_cfg = _cfg(
            input_size=(3, 256, 256),
            crop_pct=0.95 if name in ("cs3darknet_x", "cs3se_edgenet_x") else 0.887,
            interpolation="bicubic" if bicubic else "bilinear",
            **(dict(mean=(0.5,) * 3, std=(0.5,) * 3) if focus_s else {}),
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
