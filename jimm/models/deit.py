"""DeiT, DeiT-III and distilled DeiT in flax nnx. Mirrors timm.models.deit.

timm builds these with its VisionTransformer (and ``VisionTransformerDistilled``, which adds a
distillation token and head); jimm registers them from the same configurable model.
"""

from ._vit_cfgs import VIT_CFGS
from .vision_transformer import register_all

register_all((n for n in VIT_CFGS if n.startswith("deit")), __name__)
