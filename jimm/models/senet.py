"""SENet-154 in flax nnx, NHWC. Mirrors timm's ``senet154`` (timm.models.resnet).

timm builds SENet-154 as a ResNet: a three-conv "deep" stem, bottlenecks with
64 groups of width 4 whose first 1x1 convolution halves the grouped width,
squeeze-excite after the last BatchNorm, and 3x3 strided shortcut
convolutions.
"""

from ..registry import _cfg, register_model
from .resnet import Bottleneck, ResNet


@register_model
def senet154(**kwargs):
    model = ResNet(
        Bottleneck,
        (3, 8, 36, 3),
        se=True,
        groups=64,
        base_width=4,
        deep_stem=True,
        reduce_first=2,
        down_kernel=3,
        **kwargs,
    )
    model.default_cfg = _cfg(interpolation="bicubic")
    return model
