"""PP-LCNetV2 in flax nnx, NHWC. Mirrors timm.models.lcnetv2.

Depthwise-separable blocks whose later-stage depthwise convs are trained as
parallel multi-scale conv-BN branches (folded into one conv for inference), with
split pointwise convolutions, squeeze-excite, and occasional shortcuts.
"""

from flax import nnx

from ..layers import ClassifierMixin, ConvBNAct, SqueezeExcite, global_pool_nhwc, make_divisible
from ..registry import _cfg, register_model


class RepDepthwiseSeparable(nnx.Module):
    """Depthwise (optionally multi-scale re-parameterizable) + pointwise block."""

    def __init__(
        self,
        in_chs,
        out_chs,
        stride=1,
        dw_kernel_size=3,
        split_pw=False,
        use_rep=False,
        use_se=False,
        use_shortcut=False,
        pw_ratio=0.5,
        se_ratio=0.25,
        inference_mode=False,
        *,
        rngs,
    ):
        self.use_shortcut = use_shortcut and stride == 1 and in_chs == out_chs
        self.dw_act = None
        if use_rep:
            # timm trains a 1x1 branch only for stride 1, and folds the branches
            # into a single convolution for inference.
            if inference_mode:
                self.dw_branches = None
                self.dw_conv = nnx.Conv(
                    in_chs,
                    in_chs,
                    (dw_kernel_size, dw_kernel_size),
                    strides=(stride, stride),
                    padding=((dw_kernel_size // 2, dw_kernel_size // 2),) * 2,
                    feature_group_count=in_chs,
                    use_bias=True,
                    rngs=rngs,
                )
            else:
                self.dw_conv = None
                self.dw_branches = nnx.List(
                    [
                        ConvBNAct(
                            in_chs,
                            in_chs,
                            k,
                            stride,
                            groups=in_chs,
                            act="identity",
                            rngs=rngs,
                        )
                        for k in range(dw_kernel_size, 0, -2)
                        if not (k == 1 and stride != 1)
                    ]
                )
            self.dw_act = nnx.relu
        else:
            self.dw_branches = None
            self.dw_conv = ConvBNAct(
                in_chs,
                in_chs,
                dw_kernel_size,
                stride,
                groups=in_chs,
                act="relu",
                rngs=rngs,
            )
        self.se = (
            SqueezeExcite(in_chs, rd_channels=int(in_chs * se_ratio), rngs=rngs) if use_se else None
        )
        if split_pw:
            mid = int(out_chs * pw_ratio)
            self.pw1 = ConvBNAct(in_chs, mid, 1, act="relu", rngs=rngs)
            self.pw2 = ConvBNAct(mid, out_chs, 1, act="relu", rngs=rngs)
            self.pw = None
        else:
            self.pw1 = self.pw2 = None
            self.pw = ConvBNAct(in_chs, out_chs, 1, act="relu", rngs=rngs)

    def __call__(self, x):
        if self.dw_conv is not None:
            x = self.dw_conv(x)
        else:
            y = self.dw_branches[0](x)
            for branch in self.dw_branches[1:]:
                y = y + branch(x)
            x = y
        if self.dw_act is not None:
            x = self.dw_act(x)
        if self.se is not None:
            x = self.se(x)
        if self.pw is not None:
            x = self.pw(x)
        else:
            x = self.pw2(self.pw1(x))
        return x


# in_chs, dw_kernel_size, split_pw, use_rep, use_se, use_shortcut
_STAGE_CFG = (
    (64, 3, False, False, False, False),
    (128, 3, False, False, False, False),
    (256, 5, True, True, True, False),
    (512, 5, False, True, False, True),
)


class LCNetV2(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        scale=1.0,
        depths=(2, 2, 6, 2),
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        head_hidden_size=1280,
        drop_rate=0.0,
        inference_mode=False,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem_chs = make_divisible(32 * scale)
        prev = make_divisible(64 * scale)
        self.stem_conv = ConvBNAct(in_chans, stem_chs, 3, 2, act="relu", rngs=rngs)
        self.stem_block = RepDepthwiseSeparable(stem_chs, prev, rngs=rngs)
        stages = []
        for stage_idx, (chs, k, split_pw, use_rep, use_se, use_shortcut) in enumerate(_STAGE_CFG):
            out = make_divisible(chs * 2 * scale)
            blocks = []
            for block_idx in range(depths[stage_idx]):
                blocks.append(
                    RepDepthwiseSeparable(
                        prev,
                        out,
                        stride=2 if block_idx == 0 else 1,
                        dw_kernel_size=k,
                        split_pw=split_pw,
                        use_rep=use_rep,
                        use_se=use_se,
                        use_shortcut=use_shortcut,
                        inference_mode=inference_mode,
                        rngs=rngs,
                    )
                )
                prev = out
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.head_hidden_size = head_hidden_size or prev
        # The pre-logits features come from the optional 1x1 head projection.
        self.num_features = self.head_hidden_size
        if head_hidden_size:
            self.conv_head = nnx.Linear(prev, head_hidden_size, use_bias=False, rngs=rngs)
        else:
            self.conv_head = None
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(self.head_hidden_size, num_classes, rngs=rngs) if num_classes > 0 else None
        )

    def forward_features(self, x):
        x = self.stem_conv(x)
        x = self.stem_block(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        if self.conv_head is not None:
            x = nnx.relu(self.conv_head(x))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _lcnetv2(scale, depths, **kwargs):
    model = LCNetV2(scale, depths, **kwargs)
    model.default_cfg = _cfg(interpolation="bicubic")
    return model


@register_model
def lcnetv2_small(**kwargs):
    return _lcnetv2(0.75, (2, 2, 4, 2), **kwargs)


@register_model
def lcnetv2_base(**kwargs):
    return _lcnetv2(1.0, (2, 2, 6, 2), **kwargs)


@register_model
def lcnetv2_large(**kwargs):
    return _lcnetv2(1.25, (2, 2, 8, 2), **kwargs)
