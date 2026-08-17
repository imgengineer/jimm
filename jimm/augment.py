"""Timm-style image augmentation using OpenCV and dm_pix.

Images are RGB ``uint8`` NumPy arrays at the Grain boundary. Geometric and
codec operations use OpenCV; differentiable/JAX-native color, crop, flip,
blur, rotation, hue, saturation, and solarization use dm_pix. Batch
Mixup/CutMix remains NumPy/JAX friendly because it runs after Grain batching.
"""
import math

import cv2  # pyright: ignore[reportMissingImports]
import dm_pix as pix  # pyright: ignore[reportMissingImports]
import jax
import jax.numpy as jnp
import numpy as np

_INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "box": cv2.INTER_AREA,
    "hamming": cv2.INTER_LINEAR,
    "lanczos": cv2.INTER_LANCZOS4,
}
_RANDOM_INTERPOLATIONS = ("bilinear", "bicubic")


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid integer value: {value!r}") from exc


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid float value: {value!r}") from exc


__all__ = [
    "AugmentOp", "AutoAugment", "AugMixAugment", "Mixup", "MixupCutmix",
    "RandAugment", "TrivialAugmentWide", "auto_augment_transform",
    "augment_and_mix_transform", "build_auto_augment", "center_crop_or_pad",
    "color_jitter", "gaussian_blur", "random_crop_or_pad", "random_erasing",
    "random_flip_left_right", "random_flip_up_down", "random_grayscale",
    "random_resized_crop", "rand_augment_transform", "resize_keep_ratio",
    "resolve_interpolation",
]


def resolve_interpolation(interpolation="random"):
    if isinstance(interpolation, (tuple, list)):
        interpolation = interpolation[np.random.randint(len(interpolation))]
    if interpolation is None or interpolation == "random":
        interpolation = _RANDOM_INTERPOLATIONS[np.random.randint(len(_RANDOM_INTERPOLATIONS))]
    if isinstance(interpolation, str):
        try:
            return _INTERPOLATIONS[interpolation.lower()]
        except KeyError as exc:
            raise ValueError(f"unknown interpolation mode: {interpolation!r}") from exc
    return interpolation


def _size(size):
    if isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError("size must be an int or (height, width)")
        return _as_int(size[0]), _as_int(size[1])
    value = _as_int(size)
    return value, value


def _key():
    return jax.random.PRNGKey(np.int32(np.random.randint(0, 2**31 - 1)))


def _rgb(image):
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("image must be HWC RGB/RGBA or HW grayscale")
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.asarray(array, dtype=np.uint8)


def _float_image(image):
    return _rgb(image).astype(np.float32) / 255.0


def _uint8_image(array):
    return np.clip(np.asarray(array) * 255.0, 0, 255).astype(np.uint8)


def _dm_pix_image(image, operation, *args, **kwargs):
    array = jnp.asarray(_float_image(image), dtype=jnp.float32)
    return _uint8_image(operation(array, *args, **kwargs))


def random_resized_crop(image, size=224, scale=(0.08, 1.0),
                        ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation="random"):
    """Timm-compatible random area/aspect-ratio crop using OpenCV resize."""
    image = _rgb(image)
    height, width = image.shape[:2]
    area = height * width
    out_h, out_w = _size(size)
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = np.random.uniform(scale[0], scale[1]) * area
        aspect = math.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
        try:
            crop_w = _as_int(round(math.sqrt(target_area * aspect)))
            crop_h = _as_int(round(math.sqrt(target_area / aspect)))
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            crop_w, crop_h = width, height
        if 0 < crop_w <= width and 0 < crop_h <= height:
            top = np.random.randint(0, max(1, height - crop_h + 1))
            left = np.random.randint(0, max(1, width - crop_w + 1))
            crop = image[top:top + crop_h, left:left + crop_w]
            return cv2.resize(
                crop, (out_w, out_h), interpolation=resolve_interpolation(interpolation))

    input_ratio = width / height
    if input_ratio < ratio[0]:
        crop_w, crop_h = width, _as_int(round(width / ratio[0]))
    elif input_ratio > ratio[1]:
        crop_w, crop_h = _as_int(round(height * ratio[1])), height
    else:
        crop_w, crop_h = width, height
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    crop = image[top:top + crop_h, left:left + crop_w]
    return cv2.resize(
        crop, (out_w, out_h), interpolation=resolve_interpolation(interpolation))


def resize_keep_ratio(image, size=224, scale=(0.8, 1.0),
                      ratio=(0.9, 1.0 / 0.9), interpolation="random"):
    image = _rgb(image)
    target = min(_size(size)) * np.random.uniform(scale[0], scale[1])
    aspect = math.exp(np.random.uniform(math.log(ratio[0]), math.log(ratio[1])))
    width = max(1, _as_int(round(target * math.sqrt(aspect))))
    height = max(1, _as_int(round(target / math.sqrt(aspect))))
    return cv2.resize(image, (width, height), interpolation=resolve_interpolation(interpolation))


def center_crop_or_pad(image, size=224):
    height, width = _size(size)
    return _dm_pix_image(image, pix.resize_with_crop_or_pad, height, width)


def random_crop_or_pad(image, size=224):
    image = _rgb(image)
    height, width = _size(size)
    array = jnp.asarray(_float_image(image), dtype=jnp.float32)
    if image.shape[0] >= height and image.shape[1] >= width:
        array = pix.random_crop(_key(), array, (height, width, 3))
    else:
        array = pix.resize_with_crop_or_pad(array, height, width)
    return _uint8_image(array)


def _range(value, name):
    if value is None:
        return 0.0, 0.0
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must be a scalar or 2-tuple")
        return _as_float(value[0]), _as_float(value[1])
    value = _as_float(value)
    return max(0.0, 1.0 - value), 1.0 + value


def _hue_range(value):
    if value is None:
        return 0.0, 0.0
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("hue must be a scalar or 2-tuple")
        return _as_float(value[0]), _as_float(value[1])
    value = _as_float(value)
    return -value, value


def color_jitter(image, brightness=0.0, contrast=0.0, saturation=0.0,
                 hue=0.0, prob=None, random_order=True):
    """Color jitter using dm_pix's JAX-native brightness/contrast/HSL ops."""
    if prob is not None and np.random.rand() >= prob:
        return _rgb(image)
    operations = [
        ("brightness", _range(brightness, "brightness")),
        ("contrast", _range(contrast, "contrast")),
        ("saturation", _range(saturation, "saturation")),
        ("hue", _hue_range(hue)),
    ]
    operations = [item for item in operations if item[1] != (0.0, 0.0)]
    if random_order:
        np.random.shuffle(operations)
    result = _rgb(image)
    for name, bounds in operations:
        array = jnp.asarray(_float_image(result), dtype=jnp.float32)
        if name == "brightness":
            delta = np.random.uniform(-_as_float(brightness), _as_float(brightness))
            array = pix.adjust_brightness(array, delta)
        elif name == "contrast":
            array = pix.adjust_contrast(array, np.random.uniform(bounds[0], bounds[1]))
        elif name == "saturation":
            array = pix.adjust_saturation(array, np.random.uniform(bounds[0], bounds[1]))
        else:
            array = pix.adjust_hue(array, np.random.uniform(bounds[0], bounds[1]))
        result = _uint8_image(array)
    return result


def random_flip_left_right(image, prob=0.5):
    image = _rgb(image)
    if prob <= 0 or np.random.rand() >= prob:
        return image
    return cv2.flip(image, 1)


def random_flip_up_down(image, prob=0.0):
    image = _rgb(image)
    if prob <= 0 or np.random.rand() >= prob:
        return image
    return cv2.flip(image, 0)


def random_grayscale(image, prob=0.0):
    image = _rgb(image)
    if prob <= 0 or np.random.rand() >= prob:
        return image
    array = pix.rgb_to_grayscale(
        jnp.asarray(_float_image(image), dtype=jnp.float32), keep_dims=True)
    return _uint8_image(array)


def gaussian_blur(image, prob=0.0, sigma=(0.1, 2.0)):
    image = _rgb(image)
    if prob <= 0 or np.random.rand() >= prob:
        return image
    radius = _as_float(np.random.uniform(sigma[0], sigma[1]))
    kernel = max(3, _as_int(round(radius * 6)))
    if kernel % 2 == 0:
        kernel += 1
    return _dm_pix_image(image, pix.gaussian_blur, radius, kernel)


def random_erasing(array, prob=0.0, sl=0.02, sh=0.33, r1=0.3,
                   mode="const", count=1, value=0.0):
    if np.random.rand() >= prob:
        return array
    height, width, channels = array.shape
    result = array.copy()
    for _ in range(max(1, _as_int(count))):
        for _ in range(10):
            target_area = np.random.uniform(sl, sh) * height * width
            aspect = np.random.uniform(r1, 1.0 / r1)
            try:
                erase_h = _as_int(round(math.sqrt(target_area * aspect)))
                erase_w = _as_int(round(math.sqrt(target_area / aspect)))
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                erase_h, erase_w = 0, 0
            if 0 < erase_h < height and 0 < erase_w < width:
                top = np.random.randint(0, height - erase_h)
                left = np.random.randint(0, width - erase_w)
                if mode in ("rand", "pixel"):
                    fill = np.random.uniform(0.0, 1.0, (erase_h, erase_w, channels))
                elif mode == "mean":
                    fill = result.mean(axis=(0, 1), keepdims=True)
                else:
                    fill = value
                result[top:top + erase_h, left:left + erase_w] = fill
                break
    return result


def _auto_op(image, name, magnitude, hparams):
    image = _rgb(image)
    strength = max(0.0, min(_as_float(magnitude), _as_float(hparams.get("magnitude_max", 10)))) / 10.0
    if name == "AutoContrast":
        array = jnp.asarray(_float_image(image), dtype=jnp.float32)
        return _uint8_image(pix.adjust_contrast(array, 1.0 + strength))
    if name == "Equalize":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    if name == "Invert":
        return 255 - image
    if name in ("Solarize", "SolarizeIncreasing"):
        array = jnp.asarray(_float_image(image), dtype=jnp.float32)
        return _uint8_image(pix.solarize(array, 1.0 - strength))
    if name == "SolarizeAdd":
        array = _float_image(image)
        mask = array < 0.5
        array[mask] = np.clip(array[mask] + 0.43 * strength, 0.0, 1.0)
        return _uint8_image(array)
    if name in ("Color", "ColorIncreasing"):
        return _dm_pix_image(image, pix.adjust_saturation, 1.0 + _random_sign(0.9 * strength))
    if name in ("Contrast", "ContrastIncreasing"):
        return _dm_pix_image(image, pix.adjust_contrast, 1.0 + _random_sign(0.9 * strength))
    if name in ("Brightness", "BrightnessIncreasing"):
        return _dm_pix_image(image, pix.adjust_brightness, _random_sign(0.9 * strength))
    if name in ("Sharpness", "SharpnessIncreasing"):
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        return cv2.addWeighted(image, 1.0 + 0.9 * strength, blur, -0.9 * strength, 0)
    if name == "Desaturate":
        return random_grayscale(image, 1.0 if strength else 0.0)
    if name in ("GaussianBlur", "GaussianBlurRand"):
        return gaussian_blur(image, 1.0, (0.1, max(0.2, 2.0 * strength)))
    if name == "Rotate":
        return _dm_pix_image(image, pix.rotate, math.radians(_random_sign(30.0 * strength)))
    if name in ("Posterize", "PosterizeOriginal", "PosterizeIncreasing"):
        bits = max(1, 8 - _as_int(round(4 * strength)))
        return image & np.uint8((0xFF << (8 - bits)) & 0xFF)
    if name in ("ShearX", "ShearY", "TranslateXRel", "TranslateYRel"):
        shear = _random_sign(0.3 * strength) if name.startswith("Shear") else 0.0
        tx = _random_sign(0.45 * strength) if name == "TranslateXRel" else 0.0
        ty = _random_sign(0.45 * strength) if name == "TranslateYRel" else 0.0
        matrix = jnp.array([[1.0, shear, tx], [shear, 1.0, ty], [0.0, 0.0, 1.0]])
        return _dm_pix_image(image, pix.affine_transform, matrix)
    raise ValueError(f"unknown augmentation operation: {name}")


def _random_sign(value):
    return -value if np.random.rand() < 0.5 else value


class AugmentOp:
    def __init__(self, name, prob=0.5, magnitude=10.0, hparams=None):
        self.name = name
        self.prob = _as_float(prob)
        self.magnitude = _as_float(magnitude)
        self.hparams = dict(hparams or {})

    def __call__(self, image):
        if self.prob < 1.0 and np.random.rand() >= self.prob:
            return image
        magnitude = self.magnitude
        std = self.hparams.get("magnitude_std", 0.0)
        if std == _as_float("inf"):
            magnitude = np.random.uniform(0.0, magnitude)
        elif std:
            magnitude = np.random.normal(magnitude, std)
        return _auto_op(image, self.name, magnitude, self.hparams)


_V0_POLICY = [
    [("Equalize", .8, 1), ("ShearY", .8, 4)], [("Color", .4, 9), ("Equalize", .6, 3)],
    [("Color", .4, 1), ("Rotate", .6, 8)], [("Solarize", .8, 3), ("Equalize", .4, 7)],
    [("Solarize", .4, 2), ("Solarize", .6, 2)], [("Color", .2, 0), ("Equalize", .8, 8)],
    [("Equalize", .4, 8), ("SolarizeAdd", .8, 3)], [("ShearX", .2, 9), ("Rotate", .6, 8)],
    [("Color", .6, 1), ("Equalize", 1.0, 2)], [("Invert", .4, 9), ("Rotate", .6, 0)],
    [("Equalize", 1.0, 9), ("ShearY", .6, 3)], [("Color", .4, 7), ("Equalize", .6, 0)],
    [("Posterize", .4, 6), ("AutoContrast", .4, 7)], [("Solarize", .6, 8), ("Color", .6, 9)],
    [("Solarize", .2, 4), ("Rotate", .8, 9)], [("Rotate", 1.0, 7), ("TranslateYRel", .8, 9)],
    [("ShearX", .0, 0), ("Solarize", .8, 4)], [("ShearY", .8, 0), ("Color", .6, 4)],
    [("Color", 1.0, 0), ("Rotate", .6, 2)], [("Equalize", .8, 4), ("Equalize", .0, 8)],
    [("Equalize", 1.0, 4), ("AutoContrast", .6, 2)], [("ShearY", .4, 7), ("SolarizeAdd", .6, 7)],
    [("Posterize", .8, 2), ("Solarize", .6, 10)], [("Solarize", .6, 8), ("Equalize", .6, 1)],
    [("Color", .8, 6), ("Rotate", .4, 5)],
]
_ORIGINAL_POLICY = [
    [("PosterizeOriginal", .4, 8), ("Rotate", .6, 9)], [("Solarize", .6, 5), ("AutoContrast", .6, 5)],
    [("Equalize", .8, 8), ("Equalize", .6, 3)], [("PosterizeOriginal", .6, 7), ("PosterizeOriginal", .6, 6)],
    [("Equalize", .4, 7), ("Solarize", .2, 4)], [("Equalize", .4, 4), ("Rotate", .8, 8)],
    [("Solarize", .6, 3), ("Equalize", .6, 7)], [("PosterizeOriginal", .8, 5), ("Equalize", 1.0, 2)],
    [("Rotate", .2, 3), ("Solarize", .6, 8)], [("Equalize", .6, 8), ("PosterizeOriginal", .4, 6)],
    [("Rotate", .8, 8), ("Color", .4, 0)], [("Rotate", .4, 9), ("Equalize", .6, 2)],
    [("Equalize", .0, 7), ("Equalize", .8, 8)], [("Invert", .6, 4), ("Equalize", 1.0, 8)],
    [("Color", .6, 4), ("Contrast", 1.0, 8)], [("Rotate", .8, 8), ("Color", 1.0, 2)],
    [("Color", .8, 8), ("Solarize", .8, 7)], [("Sharpness", .4, 7), ("Invert", .6, 8)],
    [("ShearX", .6, 5), ("Equalize", 1.0, 9)], [("Color", .4, 0), ("Equalize", .6, 3)],
    [("Equalize", .4, 7), ("Solarize", .2, 4)], [("Solarize", .6, 5), ("AutoContrast", .6, 5)],
    [("Invert", .6, 4), ("Equalize", 1.0, 8)], [("Color", .6, 4), ("Contrast", 1.0, 8)],
    [("Equalize", .8, 8), ("Equalize", .6, 3)],
]


class AutoAugment:
    def __init__(self, policy, hparams=None):
        self.policy = policy
        self.hparams = dict(hparams or {})

    def __call__(self, image):
        row = self.policy[np.random.randint(len(self.policy))]
        for name, probability, magnitude in row:
            image = AugmentOp(name, probability, magnitude, self.hparams)(image)
        return image


def auto_augment_transform(config_str="v0", hparams=None):
    name = config_str.split("-")[0]
    if name in ("v0", "v0r"):
        policy = _V0_POLICY
    elif name in ("original", "originalr"):
        policy = _ORIGINAL_POLICY
    elif name == "3a":
        policy = [[("Solarize", 1.0, 5)], [("Desaturate", 1.0, 10)], [("GaussianBlurRand", 1.0, 10)]]
    else:
        raise ValueError(f"unknown AutoAugment policy: {name}")
    if name.endswith("r"):
        policy = [[
            (operation.replace("Posterize", "PosterizeIncreasing"), probability, magnitude)
            for operation, probability, magnitude in row
        ] for row in policy]
    config = dict(hparams or {})
    for part in config_str.split("-")[1:]:
        if part.startswith("mstd"):
            config["magnitude_std"] = _as_float(part[4:])
    return AutoAugment(policy, config)


_RAND_OPS = ["AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "Color", "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY", "TranslateXRel", "TranslateYRel"]


class RandAugment:
    def __init__(self, ops, num_layers=2):
        self.ops = ops
        self.num_layers = _as_int(num_layers)

    def __call__(self, image):
        count = min(self.num_layers, len(self.ops))
        for index in np.random.choice(len(self.ops), count, replace=False):
            image = self.ops[_as_int(index)](image)
        return image


def rand_augment_transform(config_str="rand-m9-n2", hparams=None):
    magnitude, layers = 10, 2
    config = dict(hparams or {})
    for part in config_str.split("-")[1:]:
        if part.startswith("mstd"):
            config["magnitude_std"] = _as_float(part[4:])
        elif part.startswith("m"):
            magnitude = _as_float(part[1:])
        elif part.startswith("n"):
            layers = _as_int(part[1:])
    ops = [AugmentOp(name, magnitude=magnitude, hparams=config) for name in _RAND_OPS]
    return RandAugment(ops, layers)


class TrivialAugmentWide:
    def __init__(self, hparams=None):
        self.hparams = dict(hparams or {})

    def __call__(self, image):
        name = _RAND_OPS[np.random.randint(len(_RAND_OPS))]
        magnitude = _as_float(np.random.uniform(0, 10))
        return AugmentOp(name, prob=1.0, magnitude=magnitude, hparams=self.hparams)(image)


class AugMixAugment:
    def __init__(self, ops, alpha=1.0, width=3, depth=-1):
        self.ops = ops
        self.alpha = _as_float(alpha)
        self.width = _as_int(width)
        self.depth = _as_int(depth)

    def __call__(self, image):
        weights = np.random.dirichlet([self.alpha] * self.width)
        mix = _as_float(np.random.beta(self.alpha, self.alpha))
        mixed = np.zeros_like(_float_image(image))
        for weight in weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            result = image
            for index in np.random.choice(len(self.ops), depth, replace=True):
                result = self.ops[_as_int(index)](result)
            mixed += weight * _float_image(result)
        return _uint8_image((1.0 - mix) * _float_image(image) + mix * mixed)


def augment_and_mix_transform(config_str="augmix-m3-w3", hparams=None):
    magnitude, width, depth = 3, 3, -1
    for part in config_str.split("-")[1:]:
        if part.startswith("m"):
            magnitude = _as_float(part[1:])
        elif part.startswith("w"):
            width = _as_int(part[1:])
        elif part.startswith("d"):
            depth = _as_int(part[1:])
    ops = [AugmentOp(name, prob=1.0, magnitude=magnitude, hparams=hparams)
           for name in _RAND_OPS]
    return AugMixAugment(ops, width=width, depth=depth)


def build_auto_augment(config, hparams=None):
    if not config or str(config).lower() in ("none", "off"):
        return None
    config = str(config)
    if config.startswith("rand"):
        return rand_augment_transform(config, hparams)
    if config.startswith("augmix"):
        return augment_and_mix_transform(config, hparams)
    if config.startswith("ta") or config.startswith("trivial"):
        return TrivialAugmentWide(hparams)
    return auto_augment_transform(config, hparams)


def _one_hot(labels, num_classes):
    values = np.asarray(labels)
    if values.ndim == 2:
        return values.astype(np.float32, copy=False)
    result = np.zeros((values.shape[0], num_classes), dtype=np.float32)
    result[np.arange(values.shape[0]), values.astype(np.int64)] = 1.0
    return result


def _cutmix_box(height, width, lam, minmax=None):
    if minmax is None:
        ratio = math.sqrt(max(0.0, 1.0 - lam))
        box_h, box_w = _as_int(round(height * ratio)), _as_int(round(width * ratio))
    else:
        box_h = _as_int(round(height * np.random.uniform(minmax[0], minmax[1])))
        box_w = _as_int(round(width * np.random.uniform(minmax[0], minmax[1])))
    center_y, center_x = np.random.randint(height), np.random.randint(width)
    top, left = max(0, center_y - box_h // 2), max(0, center_x - box_w // 2)
    bottom, right = min(height, center_y + box_h // 2), min(width, center_x + box_w // 2)
    actual = 1.0 - (bottom - top) * (right - left) / _as_float(height * width)
    return top, left, bottom, right, actual


class MixupCutmix:
    """Timm-style Mixup/CutMix returning soft labels."""

    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0,
                 switch_prob=0.5, mode="batch", label_smoothing=0.0,
                 num_classes=1000, cutmix_minmax=None):
        if mode not in ("batch", "pair", "elem"):
            raise ValueError("mode must be batch, pair, or elem")
        self.mixup_alpha = _as_float(mixup_alpha)
        self.cutmix_alpha = _as_float(cutmix_alpha)
        self.prob = _as_float(prob)
        self.switch_prob = _as_float(switch_prob)
        self.mode = mode
        self.label_smoothing = _as_float(label_smoothing)
        self.num_classes = _as_int(num_classes)
        self.cutmix_minmax = cutmix_minmax

    def __call__(self, images, labels):
        images = np.asarray(images)
        if np.random.rand() >= self.prob:
            return images, labels
        batch, height, width, _ = images.shape
        use_cutmix = self.cutmix_alpha > 0 and (
            self.mixup_alpha <= 0 or np.random.rand() < self.switch_prob)
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        if alpha <= 0:
            return images, labels
        targets = _one_hot(labels, self.num_classes)
        if self.label_smoothing:
            targets = targets * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
        indices = np.arange(batch - 1, -1, -1) if self.mode == "pair" else np.random.permutation(batch)
        if self.mode == "elem":
            mixed = images.copy()
            lambdas = np.empty(batch, dtype=np.float32)
            for index in range(batch):
                lam = _as_float(np.random.beta(alpha, alpha))
                if use_cutmix:
                    top, left, bottom, right, lam = _cutmix_box(height, width, lam, self.cutmix_minmax)
                    mixed[index, top:bottom, left:right] = images[indices[index], top:bottom, left:right]
                else:
                    mixed[index] = lam * images[index] + (1.0 - lam) * images[indices[index]]
                lambdas[index] = lam
            return mixed, lambdas[:, None] * targets + (1.0 - lambdas[:, None]) * targets[indices]
        lam = _as_float(np.random.beta(alpha, alpha))
        mixed = images.copy()
        if use_cutmix:
            top, left, bottom, right, lam = _cutmix_box(height, width, lam, self.cutmix_minmax)
            mixed[:, top:bottom, left:right] = images[indices, top:bottom, left:right]
        else:
            mixed = lam * images + (1.0 - lam) * images[indices]
        return mixed, lam * targets + (1.0 - lam) * targets[indices]


Mixup = MixupCutmix
