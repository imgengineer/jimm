"""Grain data pipeline with timm-style OpenCV augmentations.

Images are decoded to RGB NumPy arrays with OpenCV and yielded as normalized
float32 NHWC batches. Grain handles sharding and batching.
"""
from pathlib import Path

import cv2  # pyright: ignore[reportMissingImports]
import grain.python as grain
import jax
import numpy as np

from .augment import (
    AugmentOp,
    AutoAugment,
    AugMixAugment,
    Mixup,
    auto_augment_policy,
    auto_augment_policy_3a,
    auto_augment_policy_original,
    auto_augment_policy_originalr,
    auto_augment_policy_v0,
    auto_augment_policy_v0r,
    augmix_ops,
    MixupCutmix,
    RandAugment,
    TrivialAugmentWide,
    augment_and_mix_transform,
    auto_augment_transform,
    build_auto_augment,
    center_crop_or_pad,
    color_jitter,
    gaussian_blur,
    random_crop_or_pad,
    random_erasing,
    random_flip_left_right,
    random_flip_up_down,
    random_grayscale,
    random_resized_crop,
    rand_augment_choices,
    rand_augment_ops,
    rand_augment_transform,
    resize_keep_ratio,
    resolve_interpolation,
    str_to_interp_mode,
    str_to_pil_interp,
    interp_mode_to_str,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_IMAGE_SUFFIXES = frozenset({
    ".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".jp2", ".png", ".tif", ".tiff", ".webp",
})

__all__ = [
    "AugmentOp", "AutoAugment", "AugMixAugment", "ImageFolder", "Loader",
    "Mixup", "MixupCutmix", "RandAugment", "TrivialAugmentWide",
    "IMAGENET_MEAN", "IMAGENET_STD", "_DecodeTransform", "auto_augment_policy",
    "auto_augment_policy_3a", "auto_augment_policy_original",
    "auto_augment_policy_originalr", "auto_augment_policy_v0",
    "auto_augment_policy_v0r", "auto_augment_transform", "augmix_ops",
    "augment_and_mix_transform", "build_auto_augment", "center_crop_or_pad",
    "color_jitter", "create_dataset", "create_loader", "gaussian_blur",
    "random_crop_or_pad", "random_erasing", "random_flip_left_right",
    "random_flip_up_down", "random_grayscale", "random_resized_crop",
    "rand_augment_choices", "rand_augment_ops", "rand_augment_transform",
    "resize_keep_ratio", "resolve_interpolation", "str_to_interp_mode",
    "str_to_pil_interp", "interp_mode_to_str",
]


def _is_within(root: Path, path: Path) -> bool:
    """Keep directory entries lexically under root, including trusted symlinks."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _decode_image(raw: bytes) -> np.ndarray:
    encoded = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unable to decode image bytes")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_image(path: Path) -> np.ndarray:
    """Read a file directly; OpenCV's file reader tolerates some JPEG truncation."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode image file {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ImageFolder(grain.RandomAccessDataSource):
    """Folder dataset: ``root/class_name/image`` -> image bytes and label."""

    def __init__(self, root, in_memory=False, img_size=None):
        self.root = Path(root).expanduser().resolve()
        try:
            classes = sorted(
                (path for path in self.root.iterdir()
                 if path.is_dir() and _is_within(self.root, path)),
                key=lambda path: path.name,
            )
        except OSError:
            classes = []
        if not classes:
            raise ValueError(f"no class subdirectories under {root!r}")
        self.class_to_idx = {path.name: i for i, path in enumerate(classes)}
        samples = []
        for class_dir in classes:
            try:
                files = sorted(
                    (path for path in class_dir.iterdir()
                     if path.is_file()
                     and path.suffix.lower() in _IMAGE_SUFFIXES
                     and _is_within(self.root, path)),
                    key=lambda path: path.name,
                )
            except OSError:
                files = []
            samples.extend((path, self.class_to_idx[class_dir.name]) for path in files)
        if not samples:
            raise ValueError(f"no image files under class directories in {root!r}")
        self.samples = samples
        self._cache = None
        if in_memory:
            self._cache = []
            for path, label in self.samples:
                try:
                    with path.open("rb") as file:
                        raw = file.read()
                    try:
                        image = _decode_image(raw)
                    except ValueError:
                        image = _read_image(path)
                    if img_size:
                        image = cv2.resize(
                            image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                    self._cache.append((image, label))
                except (OSError, ValueError) as exc:
                    raise ValueError(f"unable to cache image {path}") from exc

    def __len__(self):
        return len(self._cache) if self._cache is not None else len(self.samples)

    def __getitem__(self, index):
        if self._cache is not None:
            image, label = self._cache[index]
            return {"image": image, "label": label}
        path, label = self.samples[index]
        try:
            with path.open("rb") as file:
                content = file.read()
        except OSError as exc:
            raise OSError(f"unable to read image {path}") from exc
        return {"image": content, "label": label}


class _DecodeTransform(grain.RandomMapTransform):
    """Decode one sample, apply timm-style augmentation, normalize to NHWC."""

    def __init__(self, img_size=224, is_training=False, crop_pct=0.875,
                 scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation="random", train_crop_mode="rrc", hflip=0.5,
                 vflip=0.0, color_jitter=0.4, color_jitter_prob=None, hue=0.0,
                 grayscale_prob=0.0, gaussian_blur_prob=0.0, auto_augment=None,
                 force_color_jitter=False, re_prob=0.2, re_mode="const",
                 re_count=1, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.img_size = img_size
        self.is_training = is_training
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.train_crop_mode = train_crop_mode
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter_values = color_jitter
        self.color_jitter_prob = color_jitter_prob
        self.hue = hue
        self.grayscale_prob = grayscale_prob
        self.gaussian_blur_prob = gaussian_blur_prob
        if is_training and train_crop_mode not in ("rrc", "rkrc", "rkrr"):
            raise ValueError(f"unknown train_crop_mode: {train_crop_mode}")
        self.auto_augment = build_auto_augment(auto_augment)
        self.force_color_jitter = force_color_jitter
        self.re_prob = re_prob
        self.re_mode = re_mode
        self.re_count = re_count
        try:
            crop_pct = float(crop_pct)
            self.resize = int(round(img_size / crop_pct)) if crop_pct > 0 else 256
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            self.resize = 256
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("mean and std must each contain three channels")
        if np.any(self.std == 0):
            raise ValueError("std values must be non-zero")

    @staticmethod
    def _coerce_image(raw):
        if isinstance(raw, bytes):
            return _decode_image(raw)
        image = np.asarray(raw, dtype=np.uint8)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[-1] == 4:
            return image[..., :3]
        return image

    def map(self, element):  # type: ignore[override]
        """Apply the transform with a local RNG for direct callers and tests."""
        return self.random_map(element, np.random.default_rng())

    def random_map(self, element, rng):  # pyright: ignore[reportIncompatibleMethodOverride]
        image = self._coerce_image(element["image"])

        if self.is_training:
            if self.train_crop_mode == "rrc":
                image = random_resized_crop(
                    image, self.img_size, self.scale, self.ratio, self.interpolation, rng=rng)
            elif self.train_crop_mode in ("rkrc", "rkrr"):
                image = resize_keep_ratio(
                    image, self.img_size, self.scale, self.ratio, self.interpolation, rng=rng)
                if self.train_crop_mode == "rkrc":
                    image = center_crop_or_pad(image, self.img_size)
                else:
                    image = random_crop_or_pad(image, self.img_size, rng=rng)
            else:
                raise ValueError(f"unknown train_crop_mode: {self.train_crop_mode}")
            image = random_flip_left_right(image, self.hflip, rng=rng)
            image = random_flip_up_down(image, self.vflip, rng=rng)
            if self.auto_augment is not None:
                image = self.auto_augment(image, rng=rng)
            if self.color_jitter_values is not None and (
                    self.auto_augment is None or self.force_color_jitter):
                values = self.color_jitter_values
                if isinstance(values, (tuple, list)):
                    if len(values) not in (3, 4):
                        raise ValueError("color_jitter must have 3 or 4 values")
                    brightness, contrast, saturation = values[:3]
                    hue = values[3] if len(values) == 4 else self.hue
                else:
                    brightness = contrast = saturation = values
                    hue = self.hue
                image = color_jitter(
                    image, brightness, contrast, saturation, hue,
                    prob=self.color_jitter_prob, rng=rng)
            image = random_grayscale(image, self.grayscale_prob, rng=rng)
            image = gaussian_blur(image, self.gaussian_blur_prob, rng=rng)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = random_erasing(
                array, self.re_prob, mode=self.re_mode, count=self.re_count, rng=rng)
        else:
            image = cv2.resize(
                image, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
            image = center_crop_or_pad(image, self.img_size)
            array = np.asarray(image, dtype=np.float32) / 255.0

        return {
            "image": (array - self.mean) / self.std,
            "label": np.int32(element["label"]),
        }


def create_dataset(root, in_memory=False, **kwargs):
    """Return a Grain source and transform for one folder split."""
    source = ImageFolder(root, in_memory=in_memory)
    return source, _DecodeTransform(**kwargs)


class Loader:
    """Thin wrapper over ``grain.DataLoader`` with a timm-style ``len``."""

    def __init__(self, loader, num_records, batch_size, drop_remainder):
        self._loader = loader
        self.num_records = num_records
        self.batch_size = batch_size
        self._drop = drop_remainder

    def __iter__(self):
        iterator = iter(self._loader)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                return

    def __len__(self):
        quotient, remainder = divmod(self.num_records, self.batch_size)
        return quotient if self._drop or remainder == 0 else quotient + 1


def create_loader(
        root, batch_size, img_size=224, is_training=False, crop_pct=0.875,
        scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="random", train_crop_mode="rrc", hflip=0.5,
        vflip=0.0, color_jitter=0.4, color_jitter_prob=None, hue=0.0,
        grayscale_prob=0.0, gaussian_blur_prob=0.0, auto_augment=None,
        force_color_jitter=False, re_prob=0.2, re_mode="const", re_count=1,
        mean=IMAGENET_MEAN, std=IMAGENET_STD, num_workers=4,
        worker_buffer_size=1, enable_profiling=False, seed=0, shuffle=None,
        shard_options=None, in_memory=False):
    """Create a Grain loader with timm-compatible augmentation options."""
    for name, value in (("batch_size", batch_size), ("img_size", img_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("num_workers", num_workers), ("worker_buffer_size", worker_buffer_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if worker_buffer_size == 0:
        raise ValueError("worker_buffer_size must be positive")
    source, transform = create_dataset(
        root,
        in_memory=in_memory,
        img_size=img_size,
        is_training=is_training,
        crop_pct=crop_pct,
        scale=scale,
        ratio=ratio,
        interpolation=interpolation,
        train_crop_mode=train_crop_mode,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        color_jitter_prob=color_jitter_prob,
        hue=hue,
        grayscale_prob=grayscale_prob,
        gaussian_blur_prob=gaussian_blur_prob,
        auto_augment=auto_augment,
        force_color_jitter=force_color_jitter,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        mean=mean,
        std=std,
    )
    shuffle = is_training if shuffle is None else shuffle
    if shard_options is None:
        shard_options = grain.ShardOptions(
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
            drop_remainder=is_training,
        )
    sampler = grain.IndexSampler(
        num_records=len(source),
        shard_options=shard_options,
        shuffle=shuffle,
        seed=seed,
        num_epochs=None if is_training else 1,
    )
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[transform, grain.Batch(batch_size, drop_remainder=is_training)],
        worker_count=num_workers,
        worker_buffer_size=worker_buffer_size,
        enable_profiling=enable_profiling,
    )
    records = len(source)
    shard_count = shard_options.shard_count
    local_records = records // shard_count if is_training else -(-records // shard_count)
    return Loader(loader, local_records, batch_size, is_training)
