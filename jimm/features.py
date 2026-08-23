"""Multi-scale feature extraction helpers for JAX/Flax NNX models (mirrors timm.models._features).

Wraps vision models to extract intermediate hierarchical feature representations for
downstream tasks like object detection, segmentation, and feature pyramid networks.
"""
from collections.abc import Sequence
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

__all__ = ["FeatureInfo", "FeatureExtractor", "create_feature_extractor"]


class FeatureInfo:
    """Metadata container describing intermediate feature map channel depths and reduction factors."""

    def __init__(self, info: list[dict[str, Any]], out_indices: Sequence[int]):
        self.info = info
        self.out_indices = tuple(out_indices)

    def channels(self) -> list[int]:
        """Return list of channel depths for extracted features."""
        return [self.info[i]["num_chs"] for i in self.out_indices]

    def reduction(self) -> list[int]:
        """Return list of spatial reduction ratios relative to input image."""
        return [self.info[i]["reduction"] for i in self.out_indices]

    def get(self, key: str, idx: int | None = None) -> Any:
        """Retrieve metadata field for all selected stages or a specific index."""
        if idx is None:
            return [self.info[i][key] for i in self.out_indices]
        return self.info[idx][key]

    def __len__(self) -> int:
        return len(self.out_indices)

    def __repr__(self) -> str:
        return f"FeatureInfo({self.info}, out_indices={self.out_indices})"


class FeatureExtractor(nnx.Module):
    """Wrapper that executes forward pass and returns multi-scale intermediate feature maps."""

    def __init__(self, model: nnx.Module, out_indices: Sequence[int] | None = None):
        self.model = model
        self.out_indices = tuple(out_indices) if out_indices is not None else None
        self.default_cfg = getattr(model, "default_cfg", {})
        self.feature_info: FeatureInfo | None = None

    def __call__(self, x: jax.Array) -> list[jax.Array]:
        """Extract multi-scale feature maps for input tensor (B, H, W, C)."""
        return self.forward_features(x)

    def forward_features(self, x: jax.Array) -> list[jax.Array]:
        """Forward pass extracting hierarchical feature stages across CNN & Transformer backbones."""
        m: Any = self.model
        if hasattr(m, "forward_intermediates"):
            return m.forward_intermediates(x, self.out_indices)

        feats: list[jax.Array] = []
        # ResNet-style architectures (stem + stages)
        if hasattr(m, "conv1") and hasattr(m, "bn1") and hasattr(m, "stages"):
            stem = nnx.max_pool(nnx.relu(m.bn1(m.conv1(x))), (3, 3), strides=(2, 2), padding="SAME")
            feats.append(stem)
            curr = stem
            for stage in m.stages:
                for blk in stage:
                    curr = blk(curr)
                feats.append(curr)
        # ConvNeXt-style architectures (stem + stages + downsamples)
        elif hasattr(m, "stem") and hasattr(m, "stages") and hasattr(m, "downsamples"):
            curr = m.stem_norm(m.stem(x)) if hasattr(m, "stem_norm") else m.stem(x)
            feats.append(curr)
            for i, stage in enumerate(m.stages):
                if i > 0 and i - 1 < len(m.downsamples):
                    curr = m.downsamples[i - 1](curr)
                for blk in stage:
                    curr = blk(curr)
                feats.append(curr)
        # Transformer-style architectures (patch_embed + blocks)
        elif hasattr(m, "patch_embed") and hasattr(m, "blocks"):
            batch_size = x.shape[0]
            curr = m.patch_embed(x)
            if curr.ndim == 4:
                curr = curr.reshape(batch_size, -1, curr.shape[-1])
            if hasattr(m, "cls_token"):
                cls = m.cls_token[...]
                curr = jnp.concatenate([jnp.broadcast_to(cls, (batch_size, 1, curr.shape[-1])), curr], axis=1)
            if hasattr(m, "pos_embed"):
                curr = curr + m.pos_embed[...]
            for blk in m.blocks:
                curr = blk(curr)
                feats.append(curr)
        else:
            feats.append(m.forward_features(x))

        if self.out_indices is not None:
            n = len(feats)
            selected = []
            for idx in self.out_indices:
                real_idx = n + idx if idx < 0 else idx
                if 0 <= real_idx < n:
                    selected.append(feats[real_idx])
            return selected
        return feats


def create_feature_extractor(
    model: nnx.Module,
    out_indices: Sequence[int] | None = None,
) -> FeatureExtractor:
    """Wrap a model to extract intermediate representations at specified stage indices."""
    return FeatureExtractor(model, out_indices=out_indices)
