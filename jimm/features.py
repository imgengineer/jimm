"""Feature extraction helpers for JAX/Flax NNX models (mirrors timm.models._features)."""
from typing import Any, Sequence
import jax
import jax.numpy as jnp
from flax import nnx


class FeatureInfo:
    """Metadata describing intermediate feature maps."""

    def __init__(self, info: list[dict], out_indices: Sequence[int]):
        self.info = info
        self.out_indices = tuple(out_indices)

    def channels(self) -> list[int]:
        return [self.info[i]["num_chs"] for i in self.out_indices]

    def reduction(self) -> list[int]:
        return [self.info[i]["reduction"] for i in self.out_indices]

    def get(self, key: str, idx: int | None = None):
        if idx is None:
            return [self.info[i][key] for i in self.out_indices]
        return self.info[idx][key]

    def __len__(self) -> int:
        return len(self.out_indices)

    def __repr__(self) -> str:
        return f"FeatureInfo({self.info}, out_indices={self.out_indices})"


class FeatureExtractor(nnx.Module):
    """Wrapper that runs model forward and returns multi-scale intermediate feature maps."""

    def __init__(self, model: nnx.Module, out_indices: Sequence[int] | None = None):
        self.model = model
        self.out_indices = tuple(out_indices) if out_indices is not None else None
        self.default_cfg = getattr(model, "default_cfg", {})
        self.feature_info = None

    def __call__(self, x: jax.Array) -> list[jax.Array]:
        return self.forward_features(x)

    def forward_features(self, x: jax.Array) -> list[jax.Array]:
        m: Any = self.model
        if hasattr(m, "forward_intermediates"):
            return m.forward_intermediates(x, self.out_indices)

        feats = []
        # ResNet style (stem + stages)
        if hasattr(m, "conv1") and hasattr(m, "bn1") and hasattr(m, "stages"):
            stem = nnx.max_pool(nnx.relu(m.bn1(m.conv1(x))), (3, 3), strides=(2, 2), padding="SAME")
            feats.append(stem)
            curr = stem
            for stage in m.stages:
                for blk in stage:
                    curr = blk(curr)
                feats.append(curr)
        # ConvNeXt style (stem + stages with downsamples)
        elif hasattr(m, "stem") and hasattr(m, "stages") and hasattr(m, "downsamples"):
            curr = m.stem_norm(m.stem(x)) if hasattr(m, "stem_norm") else m.stem(x)
            feats.append(curr)
            for i, stage in enumerate(m.stages):
                if i > 0 and i - 1 < len(m.downsamples):
                    curr = m.downsamples[i - 1](curr)
                for blk in stage:
                    if blk != "D":
                        curr = blk(curr)
                feats.append(curr)
        # Transformer style (patch_embed + blocks)
        elif hasattr(m, "patch_embed") and hasattr(m, "blocks"):
            B = x.shape[0]
            curr = m.patch_embed(x)
            if curr.ndim == 4:
                curr = curr.reshape(B, -1, curr.shape[-1])
            if hasattr(m, "cls_token"):
                cls = m.cls_token[...]
                curr = jnp.concatenate([jnp.broadcast_to(cls, (B, 1, curr.shape[-1])), curr], axis=1)
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


def create_feature_extractor(model: nnx.Module, out_indices: Sequence[int] | None = None) -> FeatureExtractor:
    """Create a feature extractor wrapper for extracting intermediate representations."""
    return FeatureExtractor(model, out_indices=out_indices)
