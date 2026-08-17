"""grain-based data pipeline with comprehensive augmentations. Mirrors timm.data.

Yields dicts {'image': float32 NHWC batch, 'label': int32 batch}.
Supports automatic multi-host / multi-process sharding via JAX process indices.
"""
import math
from pathlib import Path

import grain.python as grain
import jax
import numpy as np
from PIL import Image, ImageEnhance, ImageFile

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

__all__ = [
    "ImageFolder", "create_loader", "create_dataset",
    "IMAGENET_MEAN", "IMAGENET_STD", "MixupCutmix",
    "random_resized_crop", "color_jitter", "random_erasing"
]


def _is_within(root: Path, path: Path) -> bool:
    """Return whether a resolved path stays inside the dataset root."""
    try:
        path.resolve().relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _decode_image(raw: bytes) -> Image.Image:
    parser = ImageFile.Parser()
    parser.feed(raw)
    return parser.close().convert("RGB")


def random_resized_crop(img: Image.Image, size: int = 224, scale=(0.08, 1.0),
                        ratio=(3.0 / 4.0, 4.0 / 3.0)) -> Image.Image:
    """Random resized crop with area and aspect ratio scaling (mirrors timm)."""
    width, height = img.size
    area = width * height
    for _ in range(10):
        target_area = np.random.uniform(scale[0], scale[1]) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
        try:
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            w, h = width, height
        if 0 < w <= width and 0 < h <= height:
            i = np.random.randint(0, max(1, height - h + 1))
            j = np.random.randint(0, max(1, width - w + 1))
            return img.crop((j, i, j + w, i + h)).resize((size, size), Image.Resampling.BILINEAR)
    # Fallback
    return img.resize((size, size), Image.Resampling.BILINEAR)


def color_jitter(img: Image.Image, brightness=0.2, contrast=0.2, saturation=0.2) -> Image.Image:
    """Photometric jitter on PIL image."""
    if brightness > 0 and np.random.rand() < 0.8:
        try:
            factor = float(np.random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness))
        except (TypeError, ValueError, OverflowError):
            factor = 1.0
        img = ImageEnhance.Brightness(img).enhance(factor)
    if contrast > 0 and np.random.rand() < 0.8:
        try:
            factor = float(np.random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast))
        except (TypeError, ValueError, OverflowError):
            factor = 1.0
        img = ImageEnhance.Contrast(img).enhance(factor)
    if saturation > 0 and np.random.rand() < 0.8:
        try:
            factor = float(np.random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation))
        except (TypeError, ValueError, OverflowError):
            factor = 1.0
        img = ImageEnhance.Color(img).enhance(factor)
    return img


def random_erasing(arr: np.ndarray, prob=0.2, sl=0.02, sh=0.33, r1=0.3) -> np.ndarray:
    """Random erasing / cutout on (H, W, C) float array."""
    if np.random.rand() > prob:
        return arr
    h, w, c = arr.shape
    area = h * w
    for _ in range(10):
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1.0 / r1)
        try:
            h_e = int(round(math.sqrt(target_area * aspect_ratio)))
            w_e = int(round(math.sqrt(target_area / aspect_ratio)))
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            h_e, w_e = 0, 0
        if 0 < h_e < h and 0 < w_e < w:
            x1 = np.random.randint(0, h - h_e)
            y1 = np.random.randint(0, w - w_e)
            arr = arr.copy()
            arr[x1:x1 + h_e, y1:y1 + w_e, :] = 0.0
            return arr
    return arr


class ImageFolder(grain.RandomAccessDataSource):
    """torchvision-style folder dataset: root/class_x/yyy.img -> (bytes/image, label)."""

    def __init__(self, root, in_memory=False, img_size=224):
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
                     if path.is_file() and _is_within(self.root, path)),
                    key=lambda path: path.name,
                )
            except OSError:
                files = []
            samples.extend((path, self.class_to_idx[class_dir.name]) for path in files)
        self.samples = samples
        self.in_memory = in_memory
        self._cache = None
        if in_memory:
            self._cache = []
            for path, label in self.samples:
                try:
                    with path.open("rb") as file:
                        im = _decode_image(file.read())
                    if img_size:
                        im = im.resize((img_size, img_size), Image.Resampling.BILINEAR)
                    self._cache.append((np.asarray(im, dtype=np.uint8), label))
                except (OSError, ValueError):
                    continue

    def __len__(self):
        return len(self._cache) if self._cache is not None else len(self.samples)

    def __getitem__(self, i):
        if self._cache is not None:
            img_arr, label = self._cache[i]
            return {"image": img_arr, "label": label}
        path, label = self.samples[i]
        try:
            with path.open("rb") as file:
                content = file.read()
        except OSError:
            content = b""
        return {"image": content, "label": label}


class _DecodeTransform(grain.MapTransform):
    """Decode bytes/array -> uint8 RGB, full augmentations, float32 normalized NHWC."""

    def __init__(self, img_size=224, is_training=False, crop_pct=0.875,
                 scale=(0.08, 1.0), hflip=0.5, color_jitter_prob=0.2, re_prob=0.2,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.img_size, self.is_training = img_size, is_training
        self.scale = scale
        self.hflip = hflip
        self.color_jitter_prob = color_jitter_prob
        self.re_prob = re_prob
        try:
            self.resize = int(round(img_size / crop_pct))
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            self.resize = 256
        self.mean, self.std = mean, std

    def map(self, element):  # pyright: ignore[reportIncompatibleMethodOverride]
        raw = element["image"]
        if isinstance(raw, np.ndarray):
            img = Image.fromarray(raw)
        elif isinstance(raw, bytes):
            img = _decode_image(raw)
        else:
            img = raw

        if self.is_training:
            # 1. Random Resized Crop with Aspect Ratio Scaling
            img = random_resized_crop(img, size=self.img_size, scale=self.scale)
            # 2. Random Horizontal Flip
            if self.hflip > 0 and np.random.rand() < self.hflip:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            # 3. Color Jitter
            if self.color_jitter_prob > 0:
                img = color_jitter(img, brightness=self.color_jitter_prob,
                                   contrast=self.color_jitter_prob,
                                   saturation=self.color_jitter_prob)
            a = np.asarray(img, np.float32) / 255.0
            # 4. Random Erasing (Cutout)
            if self.re_prob > 0:
                a = random_erasing(a, prob=self.re_prob)
        else:
            img = img.resize((self.resize, self.resize), Image.Resampling.BILINEAR)
            left = (self.resize - self.img_size) // 2
            img = img.crop((left, left, left + self.img_size, left + self.img_size))
            a = np.asarray(img, np.float32) / 255.0

        return {"image": (a - self.mean) / self.std, "label": np.int32(element["label"])}


class MixupCutmix:
    """Batch-level Mixup and CutMix augmentation for JAX/Flax tensors."""

    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, images: np.ndarray, labels: np.ndarray):
        if np.random.rand() > self.prob:
            return images, labels
        B, H, W, _ = images.shape
        rand_index = np.random.permutation(B)

        # Mixup
        if self.mixup_alpha > 0 and np.random.rand() < 0.5:
            try:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            except (TypeError, ValueError, OverflowError):
                lam = 1.0
            mixed_images = lam * images + (1.0 - lam) * images[rand_index]
            return mixed_images, labels
        # CutMix
        elif self.cutmix_alpha > 0:
            try:
                lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            except (TypeError, ValueError, OverflowError):
                lam = 1.0
            cut_rat = math.sqrt(1.0 - lam)
            try:
                cut_w = int(round(W * cut_rat))
                cut_h = int(round(H * cut_rat))
            except (TypeError, ValueError, OverflowError):
                cut_w, cut_h = 0, 0
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            mixed_images = images.copy()
            mixed_images[:, bby1:bby2, bbx1:bbx2, :] = images[rand_index, bby1:bby2, bbx1:bbx2, :]
            return mixed_images, labels
        return images, labels


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
                  scale=(0.08, 1.0), hflip=0.5, color_jitter_prob=0.2, re_prob=0.2,
                  mean=IMAGENET_MEAN, std=IMAGENET_STD, num_workers=0, seed=0, shuffle=None,
                  shard_options=None, in_memory=False):
    """grain DataLoader over an ImageFolder split with automatic multi-host sharding.

    Args:
        batch_size: Process-local batch size (each host feeds `batch_size` samples).
        in_memory: If True, preloads and decodes images in RAM for ultra-fast training.
    """
    source, transform = create_dataset(root, in_memory=in_memory, img_size=img_size,
                                       is_training=is_training, crop_pct=crop_pct,
                                       scale=scale, hflip=hflip,
                                       color_jitter_prob=color_jitter_prob, re_prob=re_prob,
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
