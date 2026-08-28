"""Grain data pipeline with timm-style OpenCV augmentations.

Images are decoded to RGB NumPy arrays with OpenCV and yielded as normalized
float32 NHWC batches. Grain handles sharding and batching.
"""
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

from absl import flags
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
_INV_255 = np.float32(1.0 / 255.0)
_IMAGE_SUFFIXES = frozenset({
    ".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".jp2", ".png", ".tif", ".tiff", ".webp",
})
_IMAGE_CACHE_ROOT = Path(os.environ.get("JIMM_CACHE_DIR", "~/.cache/jimm/image-cache")).expanduser()


def _ensure_absl_flags_parsed():
    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()


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
    if not raw:
        raise ValueError("unable to decode empty image bytes")
    encoded = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        # Fallback: Truncated JPEG recovery (append missing EOI marker \xff\xd9)
        if raw.startswith(b"\xff\xd8"):
            image = cv2.imdecode(np.frombuffer(raw + b"\xff\xd9", dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unable to decode image bytes")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_image(path: Path) -> np.ndarray:
    """Read a file directly; OpenCV's file reader tolerates some JPEG truncation."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode image file {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _decode_file(path: Path) -> np.ndarray:
    try:
        with path.open("rb") as file:
            raw = file.read()
        try:
            return _decode_image(raw)
        except ValueError:
            return _read_image(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to cache image {path}") from exc


def _cache_key(root: Path, samples) -> str:
    digest = hashlib.sha256()
    digest.update(b"full-resolution-v1\0")
    digest.update(str(root).encode())
    digest.update(b"\0")
    for path, label in samples:
        try:
            stat = path.stat()
        except OSError as exc:
            raise ValueError(f"unable to stat image {path}") from exc
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(f"{label}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        digest.update(b"\0")
    return digest.hexdigest()


class _MemmapImageCache:
    """Read-only decoded images backed by one file shared across Grain workers."""

    def __init__(self, data_path, records, total_bytes):
        self.data_path = str(data_path)
        self.records = records
        self.total_bytes = total_bytes
        self._mmap = None

    def _data(self):
        if self._mmap is None:
            self._mmap = np.memmap(
                self.data_path,
                mode="r",
                dtype=np.uint8,
                shape=(self.total_bytes,),
            )
        return self._mmap

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        offset, size, shape, label = self.records[index]
        image = self._data()[offset:offset + size].reshape(shape)
        return image, label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mmap"] = None
        return state


def _load_memmap_cache(data_path: Path, metadata_path: Path):
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if data_path.stat().st_size != metadata["total_bytes"]:
            raise ValueError("cache size mismatch")
        records = [
            (int(offset), int(size), tuple(shape), int(label))
            for offset, size, shape, label in metadata["records"]
        ]
        return _MemmapImageCache(data_path, records, metadata["total_bytes"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _build_memmap_cache(root: Path, samples):
    key = _cache_key(root, samples)
    cache_dir = _IMAGE_CACHE_ROOT
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"{key}.bin"
    metadata_path = cache_dir / f"{key}.json"
    cache = _load_memmap_cache(data_path, metadata_path)
    if cache is not None:
        return cache

    lock_path = cache_dir / f"{key}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache = _load_memmap_cache(data_path, metadata_path)
        if cache is not None:
            return cache
        data_tmp = metadata_tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=cache_dir, prefix=f"{key}.", suffix=".bin.tmp", delete=False
            ) as data_file:
                data_tmp = Path(data_file.name)
                offset = 0
                records = []
                for path, label in samples:
                    image = _decode_file(path)
                    image = np.ascontiguousarray(image, dtype=np.uint8)
                    size = image.nbytes
                    data_file.write(memoryview(image).cast("B"))
                    records.append((offset, size, list(image.shape), label))
                    offset += size
                data_file.flush()
                os.fsync(data_file.fileno())
            metadata = {"version": 1, "total_bytes": offset, "records": records}
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=cache_dir,
                prefix=f"{key}.", suffix=".json.tmp", delete=False
            ) as metadata_file:
                metadata_tmp = Path(metadata_file.name)
                json.dump(metadata, metadata_file, separators=(",", ":"))
                metadata_file.flush()
                os.fsync(metadata_file.fileno())
            os.replace(data_tmp, data_path)
            os.replace(metadata_tmp, metadata_path)
            cache = _load_memmap_cache(data_path, metadata_path)
            if cache is None:
                raise ValueError(f"unable to load image cache {data_path}")
            return cache
        finally:
            for path in (data_tmp, metadata_tmp):
                if path is not None:
                    path.unlink(missing_ok=True)


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
        except OSError as exc:
            raise ValueError(f"unable to scan dataset root {root!r}") from exc
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
            except OSError as exc:
                raise OSError(f"unable to scan class directory {class_dir}") from exc
            samples.extend((path, self.class_to_idx[class_dir.name]) for path in files)
        if not samples:
            raise ValueError(f"no image files under class directories in {root!r}")
        self.samples = samples
        self._cache = _build_memmap_cache(self.root, self.samples) if in_memory else None

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
        self.re_prob = re_prob
        for name in (
                "hflip", "vflip", "color_jitter_prob", "grayscale_prob",
                "gaussian_blur_prob", "re_prob"):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be between 0 and 1") from exc
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
            setattr(self, name, value)
        if is_training and train_crop_mode not in ("rrc", "rkrc", "rkrr"):
            raise ValueError(f"unknown train_crop_mode: {train_crop_mode}")
        self.auto_augment = build_auto_augment(auto_augment)
        self.force_color_jitter = force_color_jitter
        self.re_mode = re_mode
        self.re_count = re_count
        try:
            crop_pct = float(crop_pct)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("crop_pct must be a positive finite number") from exc
        if not np.isfinite(crop_pct) or crop_pct <= 0:
            raise ValueError("crop_pct must be a positive finite number")
        self.resize = int(round(img_size / crop_pct))
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("mean and std must each contain three channels")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)):
            raise ValueError("mean and std values must be finite")
        if np.any(self.std <= 0):
            raise ValueError("std values must be positive")
        # Fused normalization: (x / 255 - mean) / std == x * inv_std * (1/255) + shift,
        # computed with in-place multiply/add instead of full-image temporaries.
        self._inv_std = (1.0 / self.std).astype(np.float32)
        self._shift = (-self.mean * self._inv_std).astype(np.float32)

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

    def random_map(self, element, rng):  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
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
            array = image.astype(np.float32)
            array *= _INV_255  # in-place; astype above already copied
            array = random_erasing(
                array, self.re_prob, mode=self.re_mode, count=self.re_count, rng=rng)
        else:
            image = cv2.resize(
                image, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
            image = center_crop_or_pad(image, self.img_size)
            array = image.astype(np.float32)
            array *= _INV_255

        np.multiply(array, self._inv_std, out=array)
        np.add(array, self._shift, out=array)
        result = {
            "image": array,
            "label": np.int32(element["label"]),
        }
        if "valid" in element:
            result["valid"] = np.bool_(element["valid"])
        return result


def create_dataset(root, in_memory=False, **kwargs):
    """Return a Grain source and transform for one folder split.

    The optional decoded-image cache preserves source resolution so training
    augmentations see the same input with and without caching.
    """
    source = ImageFolder(root, in_memory=in_memory)
    return source, _DecodeTransform(**kwargs)


class _PaddedDataSource(grain.RandomAccessDataSource):
    """Pad a source to a batch-and-shard multiple and mark duplicate records."""

    def __init__(self, source, multiple):
        self._source = source
        self._num_records = len(source)
        self._length = -(-self._num_records // multiple) * multiple

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if not 0 <= index < self._length:
            raise IndexError(index)
        valid = index < self._num_records
        element = dict(self._source[index if valid else 0])
        element["valid"] = np.bool_(valid)
        return element


class _OffsetSampler:
    """Apply a mutable global-record offset to an infinite training sampler."""

    def __init__(self, sampler):
        self._sampler = sampler
        self.offset = 0

    def __len__(self):
        return len(self._sampler)

    def __getitem__(self, index):
        return self._sampler[index + self.offset]

    def __repr__(self):
        return f"_OffsetSampler({self._sampler!r}, offset={self.offset})"


class _SamplerWithLength:
    """Expose the finite per-shard length expected by Grain DataLoader."""

    def __init__(self, sampler, length):
        self._sampler = sampler
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._sampler[index]

    def __repr__(self):
        return f"_SamplerWithLength({self._sampler!r}, length={self._length})"


class Loader:
    """Grain loader with a timm-style ``len`` and explicit worker cleanup."""

    def __init__(self, loader, num_records, batch_size, drop_remainder,
                 sampler=None, shard_count=1):
        self._loader = loader
        self._prefetched_iterator = None
        self._active_iterator = None
        self.num_records = num_records
        self.batch_size = batch_size
        self._drop = drop_remainder
        self._sampler = sampler
        self._shard_count = shard_count

    @staticmethod
    def _close_iterator(iterator):
        close = getattr(iterator, "close", None)
        if close is None:
            close = getattr(getattr(iterator, "_iterator", None), "close", None)
        if close is not None:
            close()

    def start_prefetch(self):
        if self._prefetched_iterator is not None or self._active_iterator is not None:
            return
        iterator = iter(self._loader)
        start = getattr(iterator, "start_prefetch", None)
        if start is not None:
            start()
        self._prefetched_iterator = iterator

    def set_start_step(self, step):
        """Start an infinite training stream at process-local batch ``step``."""
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if self._sampler is None:
            raise ValueError("set_start_step is only available for training loaders")
        if self._prefetched_iterator is not None or self._active_iterator is not None:
            raise RuntimeError("set_start_step must be called before iterating the loader")
        self._sampler.offset = int(step) * self.batch_size * self._shard_count

    def __iter__(self):
        iterator = self._prefetched_iterator
        self._prefetched_iterator = None
        if iterator is None:
            iterator = iter(self._loader)
        self._active_iterator = iterator
        try:
            while True:
                yield next(iterator)
        except StopIteration:
            return
        finally:
            if self._active_iterator is iterator:
                self._active_iterator = None
            self._close_iterator(iterator)

    def close(self):
        iterators = (self._prefetched_iterator, self._active_iterator)
        self._prefetched_iterator = None
        self._active_iterator = None
        for iterator in iterators:
            if iterator is not None:
                self._close_iterator(iterator)

    def __del__(self):
        try:
            self.close()
        except Exception:
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
        shard_options=None, in_memory=False, drop_remainder=None,
        pad_remainder=False):
    """Create a Grain loader with timm-compatible augmentation options.

    Args:
      drop_remainder: Drop the final partial batch (and per-shard record
        remainder). Defaults to ``is_training``.
      pad_remainder: Pad the source across batches and shards, adding a boolean
        ``valid`` field so distributed evaluation can retain every record while
        using equal, full batches on every host.
    """
    _ensure_absl_flags_parsed()
    for name, value in (("batch_size", batch_size), ("img_size", img_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("num_workers", num_workers), ("worker_buffer_size", worker_buffer_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if worker_buffer_size == 0:
        raise ValueError("worker_buffer_size must be positive")
    if drop_remainder is not None and not isinstance(drop_remainder, bool):
        raise ValueError("drop_remainder must be a boolean or None")
    if not isinstance(pad_remainder, bool):
        raise ValueError("pad_remainder must be a boolean")
    drop = is_training if drop_remainder is None else drop_remainder
    shuffle = is_training if shuffle is None else shuffle
    if shard_options is None:
        shard_options = grain.ShardOptions(
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
            drop_remainder=drop,
        )
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
    if pad_remainder:
        source = _PaddedDataSource(
            source, int(batch_size) * shard_options.shard_count)
    records = len(source)
    shard_count = shard_options.shard_count
    local_records, record_remainder = divmod(records, shard_count)
    if (not shard_options.drop_remainder
            and shard_options.shard_index < record_remainder):
        local_records += 1
    sampler = grain.IndexSampler(
        num_records=records,
        shard_options=shard_options,
        shuffle=shuffle,
        seed=seed,
        num_epochs=None if is_training else 1,
    )
    if not is_training:
        # DataLoader divides sampler length evenly across shards. Advertise this
        # shard's actual length so non-divisible record remainders are retained.
        sampler = _SamplerWithLength(sampler, local_records * shard_count)
    offset_sampler = _OffsetSampler(sampler) if is_training else None
    loader_sampler = offset_sampler if offset_sampler is not None else sampler
    batch_drop = drop or pad_remainder
    loader = grain.DataLoader(
        data_source=source,
        sampler=loader_sampler,
        operations=[transform, grain.Batch(batch_size, drop_remainder=batch_drop)],
        worker_count=num_workers,
        worker_buffer_size=worker_buffer_size,
        shard_options=shard_options,
        enable_profiling=enable_profiling,
    )
    return Loader(
        loader, local_records, batch_size, batch_drop,
        sampler=offset_sampler, shard_count=shard_count)
