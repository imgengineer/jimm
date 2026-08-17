"""Optional AlbumentationsX pipeline for the Grain image transform.

AlbumentationsX is intentionally imported lazily: the default OpenCV backend
must not require its AGPL/PyTorch dependency profile.
"""
from __future__ import annotations

from typing import Any

import cv2  # pyright: ignore[reportMissingImports]


def _load_albumentations() -> Any:
    try:
        import albumentations as A  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "AlbumentationsX backend requires the optional dependency profile: "
            "uv sync --extra albumentationsx"
        ) from exc
    return A


def _as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _as_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _as_range(value: Any, *, name: str) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must be a scalar or 2-tuple")
        return _as_float(value[0], name=name), _as_float(value[1], name=name)
    value = _as_float(value, name=name)
    return max(0.0, 1.0 - value), 1.0 + value


def _policy_transforms(A: Any) -> list[Any]:
    """Use AlbumentationsX primitives for policy-style color transforms."""
    return [
        A.RandomBrightnessContrast(
            brightness_range=(-0.2, 0.2), contrast_range=(-0.2, 0.2), p=1.0),
        A.ColorJitter(
            brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2), hue_range=(-0.1, 0.1), p=1.0),
        A.ToGray(p=1.0),
        A.GaussianBlur(sigma_range=(0.5, 2.0), p=1.0),
    ]


def _auto_augment(A: Any, config: str | None) -> Any | None:
    if not config:
        return None
    transforms = _policy_transforms(A)
    name = str(config).split("-", 1)[0]
    if name == "3a":
        return A.OneOf(transforms[-2:], p=1.0)
    if name.startswith("rand") or name.startswith("augmix") or name.startswith("trivial"):
        return A.SomeOf(transforms, n=min(2, len(transforms)), replace=False, p=1.0)
    return A.OneOf(transforms, p=1.0)


def build_albumentationsx_transform(
        *,
        img_size: int,
        is_training: bool,
        crop_pct: float,
        scale: tuple[float, float],
        ratio: tuple[float, float],
        interpolation: int,
        train_crop_mode: str,
        hflip: float,
        vflip: float,
        color_jitter: Any,
        color_jitter_prob: float | None,
        hue: float,
        grayscale_prob: float,
        gaussian_blur_prob: float,
        auto_augment: str | None,
        force_color_jitter: bool,
        re_prob: float,
        re_mode: str,
        re_count: int,
        mean: Any,
        std: Any,
) -> Any:
    """Build an official AlbumentationsX ``Compose`` pipeline."""
    A = _load_albumentations()
    transforms: list[Any] = []
    if is_training:
        if train_crop_mode == "rrc":
            transforms.append(A.RandomResizedCrop(
                size=(img_size, img_size), scale=scale, ratio=ratio,
                interpolation=interpolation, p=1.0))
        else:
            transforms.extend([
                A.Resize(img_size, img_size, interpolation=interpolation, p=1.0),
                A.CenterCrop(img_size, img_size, p=1.0)
                if train_crop_mode == "rkrc" else A.RandomCrop(img_size, img_size, p=1.0),
            ])
        if hflip > 0:
            transforms.append(A.HorizontalFlip(p=hflip))
        if vflip > 0:
            transforms.append(A.VerticalFlip(p=vflip))
        policy = _auto_augment(A, auto_augment)
        if policy is not None:
            transforms.append(policy)
        if color_jitter is not None and (policy is None or force_color_jitter):
            values = (
                color_jitter if isinstance(color_jitter, (tuple, list))
                else (color_jitter,) * 3
            )
            if len(values) not in (3, 4):
                raise ValueError("color_jitter must have 3 or 4 values")
            transforms.append(A.ColorJitter(
                brightness_range=_as_range(values[0], name="brightness"),
                contrast_range=_as_range(values[1], name="contrast"),
                saturation_range=_as_range(values[2], name="saturation"),
                hue_range=(
                    -_as_float(values[3] if len(values) == 4 else hue, name="hue"),
                    _as_float(values[3] if len(values) == 4 else hue, name="hue"),
                ),
                p=1.0 if color_jitter_prob is None else color_jitter_prob,
            ))
        if grayscale_prob > 0:
            transforms.append(A.ToGray(p=grayscale_prob))
        if gaussian_blur_prob > 0:
            transforms.append(A.GaussianBlur(sigma_range=(0.1, 2.0), p=gaussian_blur_prob))
        if re_prob > 0:
            fill = "random" if re_mode in ("rand", "pixel") else 0
            transforms.append(A.CoarseDropout(
                num_holes_range=(
                    max(1, _as_int(re_count, name="re_count")),
                    max(1, _as_int(re_count, name="re_count")),
                ),
                hole_height_range=(0.02, 0.33), hole_width_range=(0.02, 0.33),
                fill=fill, p=re_prob,
            ))
    else:
        resize = _as_int(round(img_size / crop_pct), name="resize")
        transforms.extend([
            A.Resize(resize, resize, interpolation=interpolation, p=1.0),
            A.CenterCrop(img_size, img_size, p=1.0),
        ])
    transforms.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0))
    return A.Compose(transforms)
