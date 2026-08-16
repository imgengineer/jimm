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
    """torchvision-style folder dataset: root/class_x/yyy.img -> (bytes/image, label)."""

    def __init__(self, root, in_memory=False, img_size=224):
        try:
            classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        except Exception:
            classes = []
        if not classes:
            raise ValueError(f"no class subdirectories under {root!r}")
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        samples = []
        for c in classes:
            c_dir = os.path.join(root, c)
            try:
                files = sorted(os.listdir(c_dir))
            except Exception:
                files = []
            for f in files:
                samples.append((os.path.join(c_dir, f), self.class_to_idx[c]))
        self.samples = samples
        self.in_memory = in_memory
        self._cache = None
        if in_memory:
            self._cache = []
            for path, label in self.samples:
                try:
                    with open(path, "rb") as f:
                        im = Image.open(f).convert("RGB")
                        if img_size:
                            im = im.resize((img_size, img_size), Image.Resampling.BILINEAR)
                        self._cache.append((np.asarray(im, dtype=np.uint8), label))
                except Exception:
                    pass

    def __len__(self):
        return len(self._cache) if self._cache is not None else len(self.samples)

    def __getitem__(self, i):
        if self._cache is not None:
            img_arr, label = self._cache[i]
            return {"image": img_arr, "label": label}
        path, label = self.samples[i]
        try:
            with open(path, "rb") as f:
                content = f.read()
        except Exception:
            content = b""
        return {"image": content, "label": label}


class _DecodeTransform(grain.MapTransform):
    """Decode bytes/array -> uint8 RGB, resize/crop, float32 normalized NHWC."""

    def __init__(self, img_size=224, is_training=False, crop_pct=0.875,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.img_size, self.is_training = img_size, is_training
        try:
            self.resize = int(round(img_size / crop_pct))
        except Exception:
            self.resize = 256
        self.mean, self.std = mean, std

    def map(self, element):  # pyright: ignore[reportIncompatibleMethodOverride]
        raw = element["image"]
        if isinstance(raw, np.ndarray):
            img = Image.fromarray(raw)
        elif isinstance(raw, bytes):
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            img = raw
        if self.is_training:
            w, h = img.size
            scale = np.random.uniform(0.6, 1.0) * min(w, h)
            try:
                side = int(scale)
            except Exception:
                side = min(w, h)
            left = np.random.randint(0, max(1, w - side + 1))
            top = np.random.randint(0, max(1, h - side + 1))
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


def create_dataset(root, in_memory=False, **kwargs):
    """Returns (grain_data_source, transform) for the given split root."""
    return ImageFolder(root, in_memory=in_memory, img_size=kwargs.get("img_size", 224)), _DecodeTransform(**kwargs)


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
                  shard_options=None, in_memory=False):
    """grain DataLoader over an ImageFolder split with automatic multi-host sharding.

    Args:
        batch_size: Process-local batch size (each host feeds `batch_size` samples).
        in_memory: If True, preloads and decodes images in RAM for ultra-fast training.
    """
    source, transform = create_dataset(root, in_memory=in_memory, img_size=img_size,
                                       is_training=is_training, crop_pct=crop_pct,
                                       mean=mean, std=std)
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
    
    # Process-local record count (eval keeps remainders per shard)
    n = len(source)
    sc = shard_options.shard_count
    local_records = n // sc if is_training else -(-n // sc)
    return Loader(loader, local_records, batch_size, is_training)
