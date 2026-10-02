# JAX Image Models (jimm)

- [What's New](#whats-new)
- [Introduction](#introduction)
- [Models](#models)
- [Features](#features)
- [Getting Started](#getting-started)
- [Training, Validation, and Inference](#training-validation-and-inference)
- [Verification](#verification)
- [Repository Layout](#repository-layout)
- [Licenses and Acknowledgments](#licenses-and-acknowledgments)

## What's New

### October 2, 2026

- Add **37 model variants** from five families following [timm 1.0.30](https://github.com/huggingface/pytorch-image-models/releases/tag/v1.0.30): LowFormer, iFormer, EfficientViM, Qwen3 ViT, and DeepSeek ViT. The registry now contains **420 models across 99 families**.
- Support Qwen3 spatial-merger and DeepSeek aligner classifiers, native VLM encoders, and distilled iFormer/EfficientViM heads.
- Update dependencies to the latest stable releases checked on this date, including **JAX 0.11.2 with CUDA 13** and **Flax 0.12.10**. The resolved environment is recorded in [uv.lock](uv.lock).
- Use **Tokamax 0.0.14** fused attention by default for GPU bfloat16 execution, with automatic kernel tuning, and use `nnx.jit_partial` for cached training/evaluation steps. Fix LowFormer transposed convolutions for mixed-precision training. See [attention performance](#attention-performance) for measurements.
- Improve ImageFolder scanning and color jitter, and fix Mixup and throughput reporting.

## Introduction

**jimm** provides image classification models, feature extractors, data augmentation, and training utilities built with JAX and Flax NNX. Its model registry and common APIs follow [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models), with native JAX implementations and NHWC image tensors.

Models use Flax NNX modules; training uses Optax, data loading uses Grain, and checkpointing uses Orbax. PyTorch is not a runtime dependency. The project implements a selected set of timm architectures; model names and weight formats should be checked against the local registry and documentation.

## Models

Use the registry to discover the exact supported names:

```python
import jimm

print(len(jimm.list_models()))  # 420
print(len(jimm.list_modules()))  # 99
print(jimm.list_models("resnet*"))
print(jimm.list_models(module="qwen3_vit"))
print(jimm.get_default_cfg("qwen3_vit_88m"))
```

Representative architectures are listed below. Each name is a registered entry; the registry contains additional variants.

| Architecture group | Example model names |
| --- | --- |
| Residual and attention CNNs | `resnet18`, `resnet50`, `resnext50_32x4d`, `seresnet50`, `resnetv2_50`, `res2net50_26w_4s`, `resnest50d`, `skresnet50` |
| Modern convolutional models | `convnext_tiny`, `convnextv2_tiny`, `regnety_008`, `rdnet_tiny`, `inception_next_tiny` |
| Mobile and efficient CNNs | `efficientnet_b0`, `mobilenetv2_100`, `mobilenetv3_large_100`, `mobilenetv5_300m`, `mnasnet_100`, `fasternet_t0`, `starnet_s050` |
| Classic CNNs and feature backbones | `densenet121`, `vgg16_bn`, `darknet53`, `cspdarknet53`, `hrnet_w18`, `xception`, `dla34`, `dpn68` |
| Vision transformers | `vit_tiny_patch16_224`, `deit_tiny_patch16_224`, `deit3_small_patch16_224`, `beit_base_patch16_224`, `eva_small_patch16_224` |
| Hierarchical transformers | `swin_tiny_patch4_window7_224`, `hiera_tiny_224`, `sam2_hiera_tiny`, `pvt_v2_b0`, `twins_svt_small`, `davit_tiny` |
| Hybrid and mobile transformers | `maxvit_tiny_rw_224`, `coatnet_0_rw_224`, `mobilevit_xxs`, `efficientvit_b0`, `fastvit_t8`, `repvit_m0_9`, `tiny_vit_5m_224` |
| Token and spatial mixers | `mixer_b16_224`, `resmlp_12_224`, `poolformer_s12`, `convmixer_768_32`, `caformer_s18`, `mambaout_tiny` |
| Additional vision towers | `gemma4_vit_167m`, `gemma4_vit_167m_enc`, `vit_sam_base_patch16_224`, `vitamin_small_224` |

### Additions from timm 1.0.30

The following families retain the upstream configurations and parameter counts. See the linked timm implementations for architecture references and original sources.

| Family | Variants | Architecture |
| --- | --- | --- |
| [LowFormer](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/lowformer.py) | 8: `lowformer_b0`, B1, B1.5 (`lowformer_b15`), B2, B3, E1–E3 | Strided convolutional attention with learned upsampling and distinct edge configurations |
| [iFormer](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/iformer.py) | 9: `iformer_t`, S, M, L, L2, H; M/L/L2 with `_distilled` | Single-head modulation attention, convolutional positional encoding, and convolution/FFN stages |
| [EfficientViM](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/efficientvim.py) | 8: `efficientvim_m1`–M4, each with a `_dist` variant | HSM-SSD hidden-state mixing and learned fusion of four classification heads |
| [Qwen3 ViT](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/qwen3_vit.py) | 9: `qwen3_vit_88m`, 306M, 416M; each with `_merge` and `_enc` variants | Learned positions, axial RoPE, and a native 2×2 spatial merger |
| [DeepSeek ViT](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/deepseek_vit.py) | 3: `deepseek_vit_412m`, `deepseek_vit_412m_align`, `deepseek_vit_412m_enc` | FP32 RMSNorm, SwiGLU, axial RoPE, and a channel-major 3×3 aligner |

### Weights

Models initialize randomly by default. There are currently no registered downloadable pretrained weights, so `jimm.list_models(pretrained=True)` returns an empty list and `pretrained=True` raises `NotImplementedError`.

Restore trained jimm models with the Orbax checkpoint helpers. `pretrained="/path/to/weights.npz"` and array state dictionaries are also supported through [jimm/weights.py](jimm/weights.py), provided their names and shapes match the converter. Automatic import of timm checkpoints is not implemented; architecture verification does not imply checkpoint compatibility for every family.

## Features

- **Model APIs:** `create_model`, filtered `list_models`, `list_modules`, and `get_default_cfg`; classification models expose `forward_features`, `forward_head`, and `reset_classifier`.
- **Feature extraction:** unpooled feature maps or tokens, pooled embeddings with `num_classes=0`, and `features_only=True` for supported intermediate stages.
- **NNX transformations:** models work with `nnx.jit` and `nnx.grad`; pass the model as an explicit argument to transformed functions. Use `model.train()` and `model.eval()` to control dropout and batch normalization.
- **Attention:** Tokamax fused kernels for GPU bfloat16 self/cross-attention, including relative-position bias. New GPU/kernel shapes automatically tune and reuse the fastest measured configuration in the process. CPU, float32, and float16 use Flax attention; active attention dropout retains its existing implementation.
- **Data and augmentation:** Grain ImageFolder loading, OpenCV decoding, random crops, color jitter, AutoAugment, RandAugment, AugMix, TrivialAugment, random erasing, Mixup, and CutMix.
- **Training:** AdamW with cosine scheduling and warmup, label smoothing, gradient clipping, optional bfloat16 computation, and JAX SPMD data parallelism or FSDP.
- **Checkpointing:** asynchronous Orbax model/optimizer checkpoints, retention settings, and epoch resume with restored data position.

Images have shape **`(batch, height, width, channels)`**. `default_cfg["input_size"]` keeps timm's metadata convention `(channels, height, width)`; use its size and normalization settings when preparing model inputs.

## Getting Started

### Installation

The project requires **Python 3.12 or newer**. Its dependency set includes `jax[cuda13]` for NVIDIA GPU execution. With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/imgengineer/jimm.git
cd jimm
uv sync --locked --python 3.12

# Inspect the JAX backend and available devices.
uv run python -c "import jax; print(jax.__version__, jax.devices())"
```

`uv sync` installs the project in editable mode and includes development tools. To force CPU execution with this environment, prefix commands with `JAX_PLATFORMS=cpu`.

Tokamax and Triton are installed by default. GPU bfloat16 attention uses [Tokamax's automatic backend selection](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/attention/api.py) and its native `autotune` cache-miss policy for both forward and backward kernels. First compilation for a new device/shape takes longer while candidate configurations are benchmarked; later calls in the same process reuse the fastest measured configuration. Unsupported shapes fall back to XLA. CPU, float32, and float16 keep Flax attention. No extra installation flags or model configuration changes are required.

### Classification and embeddings

```python
import jax.numpy as jnp
from flax import nnx
import jimm

model = jimm.create_model("resnet18", num_classes=10, rngs=nnx.Rngs(0))
model.eval()
images = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)


@nnx.jit
def infer(model, images):
    return model(images)


@nnx.jit
def extract(model, images):
    return model.forward_features(images)


print(infer(model, images).shape)  # (1, 10)
print(extract(model, images).shape)  # (1, 7, 7, 512)

model.reset_classifier(0)
print(infer(model, images).shape)  # (1, 512)
```

### Intermediate features

```python
import jax.numpy as jnp
from flax import nnx
import jimm

backbone = jimm.create_model("resnet18", features_only=True, out_indices=(1, 2, 3, 4))
backbone.eval()


@nnx.jit
def extract_stages(model, images):
    return model(images)


features = extract_stages(backbone, jnp.zeros((1, 224, 224, 3)))
print([x.shape for x in features])
# [(1, 56, 56, 64), (1, 28, 28, 128), (1, 14, 14, 256), (1, 7, 7, 512)]
```

The available stage indices depend on the architecture. For transformers, intermediate outputs may be token sequences or spatial patch grids.

### Native VLM tokens

Use an `_enc` entry for projected vision tokens; `out_features` controls the projector width independently of the encoder embedding width.

```python
import jax.numpy as jnp
from flax import nnx
import jimm

encoder = jimm.create_model("qwen3_vit_88m_enc", out_features=1024, rngs=nnx.Rngs(0))
encoder.eval()


@nnx.jit
def encode(model, images):
    return model(images)


tokens = encode(encoder, jnp.zeros((1, 224, 224, 3)))
print(tokens.shape)  # (1, 49, 1024): 14×14 patches merged into a 7×7 grid
```

Qwen3 accepts rectangular images whose dimensions are divisible by its patch size (16 by default); its spatial merger also requires even patch-grid dimensions. DeepSeek uses patch size 14 and supports `dynamic_img_pad=True` for other image sizes. Its aligner pads the patch grid before grouping 3×3 patches. The `_merge` and `_align` classifier entries accept `num_classes`.

### ImageFolder data

Arrange data as class directories with matching class names in both splits:

```text
dataset/
  train/
    class_a/001.jpg
    class_b/002.jpg
  val/
    class_a/101.jpg
    class_b/102.jpg
```

```python
from jimm.data import create_loader

loader = create_loader(
    root="/path/to/dataset/train",
    batch_size=32,
    img_size=224,
    is_training=True,
    auto_augment="rand-m9-n2",
    num_workers=4,
)
try:
    batch = next(iter(loader))
    print(batch["image"].shape)  # (32, 224, 224, 3), float32
    print(batch["label"].shape)  # (32,), int32
finally:
    loader.close()
```

Training loaders repeat across epochs and drop incomplete batches by default. Evaluation loaders run once and keep the remainder. For distributed evaluation outside the training CLI, pass `pad_remainder=True` and exclude samples where `batch["valid"]` is false.

`in_memory=True` enables a shared decoded-image cache under `~/.cache/jimm/image-cache`, configurable with `JIMM_CACHE_DIR`. It preserves source resolution so random crops and other augmentation still run on each read.

## Training, Validation, and Inference

### Training and validation

The training entry point evaluates the `val/` split after each epoch and saves model and optimizer checkpoints under `<output>/<model>`:

```bash
uv run python -m jimm.train \
    --model convnext_tiny \
    --data-dir /path/to/dataset \
    --num-classes 1000 \
    --img-size 224 \
    --epochs 90 \
    --batch-size 128 \
    --lr 5e-4 \
    --auto-augment rand-m9-n2 \
    --mixup-alpha 0.8 \
    --cutmix-alpha 1.0 \
    --output ./output
```

Set `--num-classes` to your dataset's class count. The CLI defaults to 224×224 inputs and ImageNet normalization; model-specific preprocessing settings must be supplied when constructing a custom input pipeline. Use `uv run python -m jimm.train --help` for all options.

The default compute mode uses bfloat16; `--no-amp` selects float32. `--clip-grad`, `--drop-path`, `--smoothing`, and `--mixup-mode` control regularization. Append `--resume` to the same command to restore the latest checkpoint, keeping the data, batch, sharding, and steps-per-epoch settings consistent. `--max-to-keep N` limits checkpoint retention.

The CLI creates cached `nnx.jit_partial` train/eval functions after setting their modes, following the fixed-structure approach in the [Flax NNX performance guide](https://flax.readthedocs.io/en/latest/guides/performance.html). In custom loops, set `model.train()` or `model.eval()` before constructing the corresponding cached step, and recreate it after changing static configuration or calling `reset_classifier`. Parameters, optimizer state, batch statistics, and RNG values continue to update on each call.

### Multiple devices and hosts

The CLI uses all available local devices. Parameters are replicated by default; add `--fsdp` to shard parameters and optimizer state. `--batch-size` is **per process** and must be divisible by the number of local devices. The global batch size is the process batch size multiplied by the number of processes.

For a two-host run, start the following command on host 0 and use `--dist-process-id 1` on host 1. Both hosts need the same dataset and training configuration:

```bash
uv run python -m jimm.train \
    --model resnet50 \
    --data-dir /path/to/dataset \
    --batch-size 128 \
    --dist-coordinator-address 192.168.1.100:12345 \
    --dist-num-processes 2 \
    --dist-process-id 0
```

The CLI pads distributed validation batches and applies a validity mask to retain every real sample. All processes participate in checkpoint synchronization; process 0 prints metrics.

### Saving and restoring a model

```python
from flax import nnx
import jimm
from jimm.checkpoint import load_checkpoint, save_checkpoint

model = jimm.create_model("resnet18", num_classes=10, rngs=nnx.Rngs(0))
save_checkpoint("./checkpoints/example", model, epoch=1)

restored = jimm.create_model("resnet18", num_classes=10, rngs=nnx.Rngs(1))
epoch = load_checkpoint("./checkpoints/example", restored)
print(epoch)  # 1
```

The helpers also accept an optimizer for training-state restoration. For asynchronous saves, use `wait=False` and call `wait_for_checkpoints()` before exit. Switch a restored model to evaluation mode before inference, then use the classification or encoder examples above with your preprocessed NHWC images.

## Verification

Run the core regression suite and static checks with the locked environment:

```bash
JAX_PLATFORMS=cpu uv run pytest tests/ --ignore=tests/test_models.py -q
JAX_PLATFORMS=cpu uv run pytest tests/test_models.py::test_all_registered_model_entrypoints_instantiation -q
uv run ruff check .
uv run ruff format --check .
```

Standalone scripts check representative forward passes and gradients:

```bash
uv run python scripts/test_jimm.py
uv run python scripts/test_backprop.py
```

Use `uv run pytest tests/` for the full suite, including representative forward/backward tests across model families. `scripts/test_jimm.py --all` checks every registered model at its configured input size; large variants require substantial memory and compilation time.

The architecture update passed construction checks for **all 420 models** and native-resolution CUDA 13 inference checks for one model from each new family on an RTX 5090. All **37 new variants** match timm 1.0.30 parameter counts. In a separate comparison environment, **13 reduced models** across those five families matched timm outputs with identical weights (maximum absolute error below `5e-8`). ImageNet accuracy has not been evaluated for jimm.

The attention/training update passed **195 core regression tests** (four GPU-only cases skipped on CPU) and **9 GPU attention checks**. Coverage includes automatic backend selection, the native autotuning policy, Flax output and gradient parity, shared dropout RNGs, optimizer and batch-statistic updates, mixed-precision master weights, and checkpoint resume. Run the GPU attention tests with:

```bash
uv run pytest tests/test_attention.py -q
```

### Attention performance

Measurements below use an **RTX 5090**, JAX 0.11.2/CUDA 13, Flax 0.12.10, Tokamax 0.0.14, and bfloat16 with native kernel autotuning. Times are median GPU device execution times after compilation and tuning, measured with `tokamax.benchmark`; compilation, tuning, and host-to-device transfers are excluded. The shared attention path follows the [JAX attention conventions](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html) and [Tokamax attention implementation](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/attention/api.py).

| Workload | Flax attention | jimm + Tokamax | Speedup |
| --- | ---: | ---: | ---: |
| Attention forward, `(1, 2304, 12, 64)` BTHD | 0.481 ms | 0.104 ms | 4.64× |
| Attention forward + backward, same shape | 1.287 ms | 0.419 ms | 3.07× |
| Full `qwen3_vit_88m` forward, batch 1, 768×768, 5 classes | 8.452 ms | 3.941 ms | 2.14× |

For the attention forward/backward workload, XLA's compiled temporary-buffer estimate fell from **607.5 MiB to 3.69 MiB**. The full Qwen3 forward estimate fell from **372.96 MiB to 67.46 MiB**. These estimates exclude input/output and parameter buffers and do not measure peak GPU allocation. Performance depends on model shapes and hardware.

Reproduce the attention benchmark and export its timings, numerical difference, and temporary-buffer estimates:

```bash
uv run python scripts/benchmark_attention.py \
    --seq-len 2304 --iterations 10 --output attention-benchmark.json
```

Native-resolution bfloat16 forward comparisons passed for ViT, Swin, LowFormer, Qwen3, and DeepSeek with identical weights, using mixed-precision tolerances. Native Qwen3 training with default autotuning produced a finite loss and preserved FP32 master weights; ViT and Swin also passed native bfloat16 training checks. Float32 attention outputs and gradients remain identical to Flax.

## Repository Layout

```text
jimm/
  registry.py       Model creation, registration, and configuration
  models/           Architecture implementations and shared helpers
  layers.py         Common NNX layers and classifier utilities
  attention.py      Shared attention and autotuned Tokamax dispatch
  features.py       Intermediate feature extraction
  data.py           ImageFolder datasets and Grain loaders
  augment.py        Image augmentation and policies
  train.py          Optimizer, train/eval steps, and training CLI
  checkpoint.py     Orbax model and optimizer checkpointing
  weights.py        Array state-dict conversion and NPZ loading
tests/              Regression and model tests
scripts/            Forward/backpropagation checks and attention benchmark
pyproject.toml      Project metadata and dependency requirements
uv.lock             Resolved dependency versions
```

## Licenses and Acknowledgments

Code is licensed under [Apache License 2.0](LICENSE). [NOTICE](NOTICE) retains attribution and applicable MIT notices for the adapted implementations.

Thanks to [Ross Wightman and the timm contributors](https://github.com/huggingface/pytorch-image-models) and the original architecture authors. The five new model families follow the implementations in timm 1.0.30, and this document follows the organization of [its README](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/README.md). External model weights retain their own licenses; consult their original model cards before use.
