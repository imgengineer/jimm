"""ByoaNet (BoTNet, HaloNet, LambdaNet and hybrids) in flax nnx, NHWC. Mirrors timm.models.byoanet.

These are ByobNet configurations with self-attention blocks; see ``byobnet``.
"""

from .byobnet import _BYOANET, BYOB_CFGS, register_all

register_all([n for n in BYOB_CFGS if n in _BYOANET], __name__)
