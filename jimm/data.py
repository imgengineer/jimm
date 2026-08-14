"""grain-based data pipeline. Mirrors timm.data.create_loader / create_dataset.

Yields dicts {'image': float32 NHWC batch, 'label': int32 batch}.
Supports automatic multi-host / multi-process sharding via JAX process indices.
"""
import io
import os

import grain.python as grain
import jax
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

__all__ = ["ImageFolder", "create_loader", "create_dataset", "IMAGENET_MEAN", "IMAGENET_STD"]


class ImageFolder(grain.RandomAccessDataSource):
    """torchvision-style folder dataset: root/class_x/yyy.img -> (bytes, label)."""

    def __init__(self, root):
        classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if not classes:
            raise ValueError(f"no class subdirectories under {root!r}")
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = [(os.path.join(root, c, f), self.class_to_idx[c])
                        for c in classes for f in sorted(os.listdir(os.path.join(root, c)))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with open(path, "rb") as f:
            return {"image": f.read(), "label": label}


class _DecodeTransform(grain.MapTransform):
    """Decode bytes -> uint8 RGB, resize/crop, float32 normalized NHWC."""

    def __init__(self, img_size=224, is_training=False, crop_pct=0.875,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.img_size, self.is_training = img_size, is_training
        self.resize = int(round(img_size / crop_pct))
        self.mean, self.std = mean, std

    def map(self, element):  # pyright: ignore[reportIncompatibleMethodOverride]
        img = Image.open(io.BytesIO(element["image"])).convert("RGB")
        if self.is_training:
            # ponytail: random-resized-crop approximated by random scale square crop; add aspect jitter if needed
            w, h = img.size
            scale = np.random.uniform(0.6, 1.0) * min(w, h)
            side = int(scale)
            left, top = np.random.randint(0, w - side + 1), np.random.randint(0, h - side + 1)
            img = img.crop((left, top, left + side, top + side)).resize(
                (self.img_size, self.img_size), Image.Resampling.BILINEAR)
            if np.random.rand() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            img = img.resize((self.resize, self.resize), Image.Resampling.BILINEAR)
            left = (self.resize - self.img_size) // 2
            img = img.crop((left, left, left + self.img_size, left + self.img_size))
        a = np.asarray(img, np.float32) / 255.0
        return {"image": (a - self.mean) / self.std, "label": np.int32(element["label"])}


def create_dataset(root, **kwargs):
    """Returns (grain_data_source, transform) for the given split root."""
    return ImageFolder(root), _DecodeTransform(**kwargs)


class Loader:
    """Thin wrapper over grain.DataLoader with timm-style len()."""

    def __init__(self, loader, num_records, batch_size, drop_remainder):
        self._loader = loader
        self.num_records = num_records
        self.batch_size = batch_size
        self._drop = drop_remainder

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        q, r = divmod(self.num_records, self.batch_size)
        return q if self._drop or r == 0 else q + 1


def create_loader(root, batch_size, img_size=224, is_training=False, crop_pct=0.875,
                  mean=IMAGENET_MEAN, std=IMAGENET_STD, num_workers=4, seed=0, shuffle=None,
                  shard_options=None):
    """grain DataLoader over an ImageFolder split with automatic multi-host sharding.

    Args:
        batch_size: Process-local batch size (each host feeds `batch_size` samples).
    """
    source, transform = create_dataset(root, img_size=img_size, is_training=is_training,
                                       crop_pct=crop_pct, mean=mean, std=std)
    shuffle = is_training if shuffle is None else shuffle
    
    # Auto-shard across JAX distributed processes if not explicitly provided
    if shard_options is None:
        shard_options = grain.ShardOptions(
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
            drop_remainder=is_training
        )

    sampler = grain.IndexSampler(num_records=len(source), shard_options=shard_options,
                                 shuffle=shuffle, seed=seed,
                                 num_epochs=None if is_training else 1)
    loader = grain.DataLoader(data_source=source, sampler=sampler,
                              operations=[transform, grain.Batch(batch_size, drop_remainder=is_training)],
                              worker_count=num_workers)
    
    # Calculate process-local record count
    local_records = len(source) // jax.process_count()
    return Loader(loader, local_records, batch_size, is_training)
