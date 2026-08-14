"""HRNet in flax nnx, NHWC. Mirrors timm.models.hrnet / official HRNet-Image-Classification.

Classifier head: per-branch 4x channel increase + progressive downsample-merge (official scheme).
"""
import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, global_pool_nhwc
from ..registry import register_model, _cfg


class HRBottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_chs, chs, stride=1, *, rngs):
        out = chs * self.expansion
        self.conv1 = ConvBNAct(in_chs, chs, 1, rngs=rngs)
        self.conv2 = ConvBNAct(chs, chs, 3, stride, rngs=rngs)
        self.conv3 = ConvBNAct(chs, out, 1, act="identity", rngs=rngs)
        self.shortcut = ConvBNAct(in_chs, out, 1, stride, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out) else None

    def __call__(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(y + sc)


class HRBasicBlock(nnx.Module):
    def __init__(self, chs, *, rngs):
        self.conv1 = ConvBNAct(chs, chs, 3, rngs=rngs)
        self.conv2 = ConvBNAct(chs, chs, 3, act="identity", rngs=rngs)

    def __call__(self, x):
        return nnx.relu(self.conv2(self.conv1(x)) + x)


class FuseLayer(nnx.Module):
    """Multi-resolution fusion: from each input branch to each output branch.

    width grows as resolution halves: wi < wj -> downsample (3x3 s2 convs),
    wi > wj -> 1x1 conv + nearest upsample, wi == wj -> identity.
    """

    def __init__(self, in_widths, out_widths, *, rngs):
        projs, factors = [], []
        for ow in out_widths:
            row, frow = [], []
            for iw in in_widths:
                if iw == ow:
                    row.append(None)
                    frow.append(1)
                elif iw < ow:  # downsample by ow/iw (power of 2) via 3x3 s2 convs
                    steps, convs, ch = 0, [], iw
                    f = ow // iw
                    while f > 1:
                        steps += 1
                        f //= 2
                    for t in range(steps):
                        out_c = ow if t == steps - 1 else iw
                        convs.append(ConvBNAct(ch, out_c, 3, 2,
                                               act="relu" if t < steps - 1 else "identity", rngs=rngs))
                        ch = out_c
                    row.append(nnx.List(convs))
                    frow.append(1)
                else:  # 1x1 conv + nearest upsample by iw // ow
                    row.append(ConvBNAct(iw, ow, 1, act="identity", rngs=rngs))
                    frow.append(iw // ow)
            projs.append(nnx.List(row))
            factors.append(frow)
        self.projs = nnx.List(projs)
        self.factors = factors

    def __call__(self, xs):
        outs = []
        for row, frow in zip(self.projs, self.factors):
            zs = [self._apply(proj, x, f) for proj, x, f in zip(row, xs, frow)]
            y = zs[0]
            for z in zs[1:]:
                y = y + z
            outs.append(nnx.relu(y))
        return outs

    @staticmethod
    def _apply(proj, x, factor):
        if proj is None:
            return x
        if isinstance(proj, nnx.List):
            for c in proj:
                x = c(x)
            return x
        x = proj(x)
        b, h, w, c = x.shape
        return jax.image.resize(x, (b, h * factor, w * factor, c), "nearest")


def _run_branch(branch, x):
    for blk in branch:
        x = blk(x)
    return x


class HRNet(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, widths, s1_blocks=4, s2_blocks=4, s3_modules=4, s4_modules=3, bpm=4,
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List([ConvBNAct(in_chans, 64, 3, 2, rngs=rngs),
                              ConvBNAct(64, 64, 3, 2, rngs=rngs)])
        self.stage1 = nnx.List([HRBottleneck(64 if j == 0 else 256, 64, rngs=rngs)
                                for j in range(s1_blocks)])
        # transitions: 256 -> b0/b1 after stage1; then from top of previous fuse
        self.trans0 = nnx.List([ConvBNAct(256, widths[0], 3, 1, rngs=rngs),
                                ConvBNAct(256, widths[1], 3, 2, rngs=rngs)])
        self.trans1 = ConvBNAct(widths[1], widths[2], 3, 2, rngs=rngs)
        self.trans2 = ConvBNAct(widths[2], widths[3], 3, 2, rngs=rngs)
        self.s2 = nnx.List([nnx.List([HRBasicBlock(widths[0], rngs=rngs) for _ in range(s2_blocks)]),
                            nnx.List([HRBasicBlock(widths[1], rngs=rngs) for _ in range(s2_blocks)])])
        self.fuse2 = FuseLayer([widths[0], widths[1]], [widths[0], widths[1]], rngs=rngs)
        self.s3 = nnx.List([nnx.List([nnx.List([HRBasicBlock(w, rngs=rngs) for _ in range(bpm)])
                                      for w in widths[:3]]) for _ in range(s3_modules)])
        self.fuse3 = FuseLayer(widths[:3], widths[:3], rngs=rngs)
        self.s4 = nnx.List([nnx.List([nnx.List([HRBasicBlock(w, rngs=rngs) for _ in range(bpm)])
                                      for w in widths]) for _ in range(s4_modules)])
        self.fuse4 = FuseLayer(widths, widths, rngs=rngs)
        # official increased head: 4x channels per branch, merge down to 1/32 res
        self.incre = nnx.List([ConvBNAct(w, 4 * w, 1, rngs=rngs) for w in widths])
        self.downs = nnx.List([ConvBNAct(4 * widths[i], 4 * widths[i + 1], 3, 2,
                                         act="identity", rngs=rngs) for i in range(3)])
        self.num_features = 4 * widths[-1]
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for blk in self.stage1:
            x = blk(x)
        xs = [self.trans0[0](x), self.trans0[1](x)]
        xs = [_run_branch(b, x) for b, x in zip(self.s2, xs)]
        xs = self.fuse2(xs)
        xs.append(self.trans1(xs[1]))
        for module in self.s3:
            xs = [_run_branch(b, x) for b, x in zip(module, xs)]
            xs = self.fuse3(xs)
        xs.append(self.trans2(xs[2]))
        for module in self.s4:
            xs = [_run_branch(b, x) for b, x in zip(module, xs)]
            xs = self.fuse4(xs)
        # increased head merge
        y = self.downs[0](self.incre[0](xs[0])) + self.incre[1](xs[1])
        y = self.downs[1](y) + self.incre[2](xs[2])
        y = self.downs[2](y) + self.incre[3](xs[3])
        return nnx.relu(y)

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


_CFGS = {  # widths, s1_blocks, s2_blocks, s3_modules, s4_modules, blocks_per_module
    "hrnet_w18_small": ([16, 32, 64, 128], 2, 2, 3, 2, 2),
    "hrnet_w18": ([18, 36, 72, 144], 4, 4, 4, 3, 4),
    "hrnet_w32": ([32, 64, 128, 256], 4, 4, 4, 3, 4),
    "hrnet_w48": ([48, 96, 192, 384], 4, 4, 4, 3, 4),
}


def _make(name):
    widths, s1, s2, m3, m4, bpm = _CFGS[name]

    def entry(**kwargs):
        model = HRNet(widths, s1, s2, m3, m4, bpm, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
