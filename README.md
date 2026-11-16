# JAX Image Models (jimm)

- [What's New](#whats-new)
- [Introduction](#introduction)
- [Models](#models)
- [Features](#features)
- [Getting Started](#getting-started)
- [Training, Validation, and Inference](#training-validation-and-inference)
- [Testing](#testing)
- [Performance](#performance)
- [Repository Layout](#repository-layout)
- [Licenses and Acknowledgments](#licenses-and-acknowledgments)

## What's New

### November 16, 2026

- Normalize the codebase against a stricter lint profile: `pyproject.toml` now enables whitespace (W), modern-syntax (UP), bug-prone-construct (B), simplification (SIM), misc (PIE), return (RET), raise (RSE), string-concatenation (ISC), static-join (FLY) and ruff-specific (RUF) rules on top of the default E/F/I set, with `ruff format` still the single owner of layout. The resulting 62-file diff only touches syntax: `__all__` lists and imports are sorted, `from typing import Callable` moves to `collections.abc`, `getattr`/`setattr` on constant names become attribute access, `[*a, b]` replaces list concatenation, unused unpacked shapes and loop counters become `_`-prefixed (or disappear with their `enumerate`), `int(round(...))` drops the redundant cast, `if`/`else` assignments become expressions, three assigned lambdas become `def`s (removing their `# noqa: E731`), and 21 misplaced `# fmt: skip` comments (no-ops on closing brackets) are removed. `B905` stays ignored because `zip(..., strict=True)` would change the runtime behaviour of the ported model code, and the public `jimm.__all__` keeps its area grouping with a documented `# noqa: RUF022` instead of an alphabetical flat list.

### November 15, 2026

- Add an optional JAX persistent compilation cache (`--compilation-cache DIR`, or `JAX_COMPILATION_CACHE_DIR`) to the training, validation, inference and benchmark entry points.
- Rotate Orbax checkpoints with background deletion so retention cleanup no longer blocks the save path; async checkpointing remains the default.
- Order checkpoint overwrites against that background deletion. Re-saving an existing step (`save(force=True)`, used when a run resumes past its last checkpoint or two runs share an output directory) now waits for the queued removal of the old step, which previously could fail the save or delete the checkpoint it had just written while Orbax still listed the step. `CheckpointManager.close()` also waits for pending deletions and reports deletion failures instead of leaving them on Orbax's daemon thread.
- Parametrize the per-model instantiation test: it is now one case per registered model instead of a single loop, so `pytest -n 8` spreads the ~1400 abstract constructions over the workers (39 s instead of a 354 s single-core test) and a failure names the offending model. The testing docs now record the measured scaling: eight workers saturate a 14-core CPU, so more workers, thread-count limits and a shared compilation cache all fail to shorten the suite.
- Test that a crash only pauses training instead of changing it: `tests/test_train.py` restarts a run from the epoch-1 checkpoint and asserts that the next epoch's parameters and AdamW moments match the uninterrupted run bitwise, which covers optimizer state, the model's drop-path RNG streams, and the data-stream offset replayed by `set_start_step`.
- Let `load_checkpoint` read `CheckpointManager` step directories, which the evaluation scripts already document as accepted `--checkpoint` values: manager steps keep their metadata at the manager root, so the previous behaviour was a bare `FileNotFoundError` about a missing `_METADATA` file. Passing the manager root now raises a `ValueError` naming the step directory to use instead, and `tests/test_scripts.py` runs `validate.py --checkpoint <manager>/<step>` end to end. `CheckpointManager.save` also documents that Orbax serializes writes — only the commit stage is asynchronous — so there is no growing queue of pending saves to throttle.

- Split `jimm/train.py` (1290 lines) along its responsibilities without changing behaviour: argument parsing now lives in `jimm/train_cli.py`, device-side normalization/random erasing/Mixup-CutMix in `jimm/preprocess.py`, and the double-buffered host-to-device prefetch loop in `jimm/prefetch.py`, which the validation helpers in `jimm/scripts_common.py` now reuse instead of keeping a second copy. `main()` reads as a sequence of named phases (runtime setup, model/loader/sharding construction, resume, per-epoch validation), and `jimm/train.py` re-exports the moved names, so existing `jimm.train` imports keep working.

### November 14, 2026

- Close the remaining gap to timm 1.0.30: the registry now contains every timm 1.0.30 model name (1351) plus 9 additional jimm-only names. Model module grouping and the default `input_size`, `crop_pct`, `interpolation`, `mean` and `std` metadata now match timm for all 1351 shared names. New families are MobileViTv2 (seven widths), GhostNetV2/V3, the PP-LCNetV2 small/base/large models, MambaOut's re-trained `_rw` variants, EfficientViT-MSRA (M0–M5), EfficientViT-MIT B3 and L1–L3, PNASNet-5-Large, the legacy Cadene SE-ResNet/SE-ResNeXt/SENet-154/Xception registrations, Sapiens2 (B/L/H/g/5B), and the pruned EfficientNet-B1/B2/B3, EfficientNet-ES/EL and ECA-ResNet-50d/101d models. Parameter counts match timm for every new entry.
- Update the locked environment to JAX 0.12.0.1, Flax 0.12.12, Grain 0.2.19, NumPy 2.5.4, Optax 0.2.9, Orbax Checkpoint 0.12.10, Tokamax 0.0.17, Triton 3.9.0, and pytest 9.2.0.
- Install CPU JAX on macOS and `jax[cuda13]` on Linux, so the project keeps NVIDIA GPU support as its default while remaining installable on Apple silicon.
- Add the timm-style top-level scripts `train.py`, `validate.py`, `inference.py`, `benchmark.py`, `avg_checkpoints.py`, `clean_checkpoint.py`, `bulk_runner.py` and `distributed_train.sh`, with shared argument/data/metric helpers in `jimm/scripts_common.py`.

### October 4, 2026

- Complete timm's BEiT (BEiT v2, `beit_large_patch16_512`), Hiera (`abswin` models with windowed absolute position embeddings and layer scale), HRNet (W18-small-v2, W30/W40/W44/W64 and the SSLD models) and PVTv2 (`pvt_v2_b2_li` with linear spatial-reduction attention). PVTv2's MLP now applies GELU only after the depthwise conv, and its attention and patch-embedding norms use timm's epsilon.
- Complete timm's Res2Net (Res2NeXt, 6s/8s/48w and `d` variants), ResNeSt (radix 1/4 and grouped split attention), DenseNet-161/264d and the blur-pooled DenseNet, DLA Res2Net/Res2NeXt, and the ECA and EvoNorm VoVNets. Res2Net bottlenecks now treat projecting stride-1 blocks as stage-first blocks, as timm does.
- Port timm's EfficientNet builder: every `efficientnet`, `mobilenetv3` and `hardcorenas` name is rebuilt from its recorded block configuration, including EfficientNet-V2, EfficientNet-X/H, CondConv, TF-padded `tf_*` ports, MixNet, MNASNet, FBNet, LCNet, MobileNet V1/V2/V3 and MobileNetV4 (universal inverted residual and multi-query attention blocks), with GroupNorm, LayerNorm, EvoNorm and anti-aliased variants. The `_pruned` EfficientNets are not included.
- Complete timm's NFNet module: NF-RegNet B0–B5, NF-ResNet-26/50/101 and their SE and ECA variants, and `test_nfnet`.
- Complete timm's MLP-Mixer module (gMLP, gMixer, `resmlp_big_24_224` and every Mixer size), Swin V2 CR (384×384, `_ns` and window-scaled variants), the Swin V2 window-transfer fine-tunes (`window12to16`, `window12to24`) and the Swin-S3 models.
- Port timm's VisionTransformer as one configurable model and register every timm `vision_transformer` and `deit` name: AugReg, CLIP/OpenCLIP/MetaCLIP/DFN, SigLIP/SigLIP 2 (MAP heads), DINOv2 (with registers), AIMv2, BEiT3, InternViT, TIPSv2, ViT-22B-style parallel-scaling and parallel blocks, residual-post-norm, differential attention, register/GAP variants and the distilled DeiT models.
- Complete timm's ConvNeXt family in one configurable model: overlapping stems (`_ols`), RMSNorm and SimpleNorm variants, the norm-first head (`convnext_tiny_hnf`), the CLIP MLP head (`convnext_large_mlp`), `convnext_xlarge`/`convnext_xxlarge`, and ConvNeXt-V2 under the same `convnext` module.
- Complete timm's LeViT family: `levit_128`, `levit_384`, the SiLU stride-8 stem, wide and deep variants, and the `levit_conv_*` names. The weight loader now loads 1×1 convolution kernels into linear layers.
- Complete timm's RegNet family: every RegNetX/RegNetY width (through the SEER `regnety_2560`), the torchvision-rounded `_tv` variants, GroupNorm RegNetY, pre-activation RegNetV and inverted-bottleneck RegNetZ.
- Port timm's ByobNet and ByoaNet modules: GENet, RepVGG, MobileOne, RegNetZ (including EvoNorm), the ts/q ResNets with SE/ECA/global-context/BAT attention, OpenAI CLIP ResNets, BoTNet, HaloNet and LambdaNet.
- Port timm's EVA module: EVA/EVA-02 (including the CLIP towers), the axial and mixed RoPE ViTs, DINOv3, LingBot and Perception Encoder (core, language and spatial) towers.
- Complete timm's MaxViT/CoAtNet family: rmlp and TensorFlow-ported MaxViT, CoAtNet (rw, rmlp, BatchNorm and TF-style), CoAtNeXt, MaxxViT and MaxxViT-V2, with learned, TF-style and MLP relative position biases.
- Register timm's remaining CaiT variants and timm's name for the 1024-wide ConvMixer, replacing jimm-only names.
- Port timm's TNT (tnt_s, tnt_s_legacy, tnt_b) and CrossViT (all eleven variants, including the dagger stems).
- Complete timm's ResNet family: deep and tiered stems (c/d/s/t variants), average-pool shortcuts, ECA and SE attention, anti-aliased (average and blur pool) downsampling, ResNet-RS, wide ResNets, GroupNorm ResNet-50, and the remaining SE-ResNet/ResNeXt variants.
- Complete timm's ResNet V2 family: pre-activation basic and bottleneck blocks, deep/tiered stems with average-pool shortcuts, GroupNorm, EvoNorm-S0 and Filter Response Norm variants, and the BiT models (weight-standardized convolutions, width multipliers).
- Port timm's hybrid ViTs (BiT ResNetV2 R26/R50 and ResNet-D trunks, and MobileCLIP's vit_base_mci_224).
- Port timm's Segment Anything ViT encoders (samvit_base/large/huge_patch16 and samvit_base_patch16_224).

### October 3, 2026

- Port timm's RegNet, PiT, EfficientFormer, SENet-154, aligned Xception, SelecSLS, HardCoReNAS, RepGhostNet, VoVNet, HGNet/HGNetV2, TinyViT, EdgeNeXt, XCiT, FasterNet, SwiftFormer, ConViT, Twins, NesT, DaViT (including the Florence-2 image towers), FocalNet, GCViT, MobileViT, MViTv2, EfficientFormer-V2, SHViT, Sequencer2D, NFNet, CSPNet/DarkNet (including the CS3 variants), CoaT, Next-ViT, VOLO, ViT-RelPos, ViTamin, FastViT (including the MobileCLIP image towers), CPUBone, CSATv2, MobileNetV5 (including the Gemma 3n encoder), NASNet-A Large, the Gemma 4 vision towers, NaFlexViT, and the MetaFormer baselines (CAFormer, ConvFormer, PoolFormerV2), and register their remaining timm variants.
- Port timm's StarNet, ReXNet, InceptionNeXt, MambaOut, RepViT, DLA, DPN, Hiera, and SAM2 Hiera (HieraDet).
- Pad strided convolutions and pooling symmetrically, as PyTorch does, instead of Flax's SAME padding; TinyNet follows timm's width and depth scaling.
- Freeze long-lived Python objects after the first training step to reduce garbage-collection pauses in the training loop.

### October 2, 2026

- Port timm's BEiT, ResNeSt, Swin V2, Swin V2-CR, DeiT-III, CaiT, PoolFormer, TResNet, RDNet, SK-ResNet, LeViT, MaxViT, CoAtNet, EfficientViT (MIT), Visformer, and HRNet, and align SE-ResNet/SE-ResNeXt, MobileNetV3, MNASNet, SPNASNet, MLP-Mixer, ResMLP, ConvMixer, GhostNet, PVTv2, and VGG-BN with timm.
- Fix the Swin shifted-window attention mask and LayerNorm epsilon.
- Compute grouped-convolution kernel gradients with `jimm.layers.conv_general_dilated`, which avoids slow XLA lowerings of large depthwise and dilated grouped kernels on GPU.
- Use Tokamax fused attention with heuristic kernel configurations for GPU bfloat16; `--attn-autotune` enables per-shape autotuning.
- Use PyTorch-equivalent BatchNorm momentum.
- Follow timm for color jitter, AutoAugment, RandAugment, AugMix, and random erasing, and resize the shorter edge before center cropping at evaluation.
- Keep training images as uint8 from loader workers to the device, where normalization and random erasing run inside the compiled step.
- Add LowFormer, iFormer, EfficientViM, Qwen3 ViT, and DeepSeek ViT from timm 1.0.30, with Qwen3 spatial-merger and DeepSeek aligner classifiers and native VLM encoders.
- Update to JAX 0.11.2 (CUDA 13), Flax 0.12.10, and Tokamax 0.0.14.
- Pass regularization and pooling overrides to models only when requested, and fix ImageFolder scanning, Mixup, and throughput reporting.

## Introduction

**jimm** provides image classification models, feature extractors, data augmentation, and training utilities built with JAX and Flax NNX. Its model registry and common APIs follow [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models), with native JAX implementations and NHWC image tensors.

Models use Flax NNX modules; training uses Optax, data loading uses Grain, and checkpointing uses Orbax. PyTorch is not a runtime dependency. The project implements a selected set of timm architectures; model names and weight formats should be checked against the local registry and documentation.

## Models

Use the registry to discover the exact supported names:

```python
import jimm

print(len(jimm.list_models()))  # 1360
print(len(jimm.list_modules()))  # 99
print(jimm.list_models("resnet*"))
print(jimm.list_models(module="qwen3_vit"))
print(jimm.get_default_cfg("qwen3_vit_88m"))
```

Representative architectures are listed below. Each name is a registered entry; the registry contains additional variants.

| Architecture group | Example model names |
| --- | --- |
| Residual and attention CNNs | `resnet18`, `resnet50`, `resnext50_32x4d`, `seresnet50`, `resnetv2_50`, `res2net50_26w_4s`, `resnest50d`, `skresnet50`, `tresnet_m` |
| Modern convolutional models | `convnext_tiny`, `convnextv2_tiny`, `regnety_008`, `rdnet_tiny`, `inception_next_tiny`, `ghostnetv2_100` |
| Mobile and efficient CNNs | `efficientnet_b0`, `tf_efficientnetv2_s`, `efficientnet_b1_pruned`, `mobilenetv2_100`, `mobilenetv3_large_100`, `mobilenetv4_conv_medium`, `lcnet_100`, `lcnetv2_base`, `mobilenetv5_300m`, `mnasnet_100`, `fasternet_t0`, `starnet_s050` |
| Classic CNNs and feature backbones | `densenet121`, `vgg16_bn`, `darknet53`, `cspdarknet53`, `hrnet_w18`, `pnasnet5large`, `xception`, `dla34`, `dpn68` |
| Vision transformers | `vit_tiny_patch16_224`, `vit_base_patch16_clip_224`, `vit_small_patch14_reg4_dinov2`, `vit_so400m_patch14_siglip_224`, `vit_base_patch16_sapiens2`, `deit_tiny_distilled_patch16_224`, `deit3_small_patch16_224`, `beit_base_patch16_224`, `eva02_small_patch14_224` |
| Hierarchical transformers | `swin_tiny_patch4_window7_224`, `hiera_tiny_224`, `sam2_hiera_tiny`, `pvt_v2_b0`, `twins_svt_small`, `davit_tiny` |
| Hybrid and mobile transformers | `maxvit_tiny_rw_224`, `coatnet_0_rw_224`, `mobilevit_xxs`, `mobilevitv2_100`, `efficientvit_b0`, `efficientvit_m0`, `fastvit_t8`, `repvit_m0_9`, `tiny_vit_5m_224` |
| Token and spatial mixers | `mixer_b16_224`, `resmlp_12_224`, `poolformer_s12`, `convmixer_768_32`, `caformer_s18`, `mambaout_tiny` |
| Additional vision towers | `gemma4_vit_167m`, `gemma4_vit_167m_enc`, `samvit_base_patch16_224`, `vitamin_small_224` |

The architectures listed in [tests/test_timm_parity.py](tests/test_timm_parity.py) reproduce their timm 1.0.30 counterparts: parameter counts match. Automatic import of timm checkpoints is not implemented, so output-level equality is not asserted by the test suite. They include the ResNet, ResNeXt, SE-ResNet, SENet-154, SK-ResNet, TResNet, ConvNeXt, RegNet (X/Y/V/Z), NFNet (including NF-ResNet and NF-RegNet), RDNet, EfficientNet (including EfficientNet-V2, MixNet, MNASNet, FBNet, LCNet, the pruned variants and PP-LCNetV2), ViT (including CLIP, SigLIP, DINOv2, AIMv2 and BEiT3 towers)/DeiT/DeiT-III, BEiT, CaiT, PiT, ConViT, Twins, NesT, Swin/Swin V2, DaViT, FocalNet, GCViT, MViTv2, Hiera/SAM2 Hiera, InceptionNeXt, MambaOut (including the `_rw` variants), RepViT, StarNet, ReXNet, FasterNet, SwiftFormer, PoolFormer/PoolFormerV2, CAFormer, ConvFormer, ResNeSt, DenseNet, CSPNet/DarkNet, CoaT, Next-ViT, VOLO, ViT-RelPos, ViTamin, FastViT, CPUBone, CSATv2, MobileNetV5, NASNet, PNASNet, Gemma 4 ViT, NaFlexViT, TNT, CrossViT, SAM ViT, hybrid ViT, MaxxViT, EVA (including EVA-02, RoPE ViT, DINOv3, Perception Encoder and Sapiens2), ByobNet/ByoaNet, VoVNet, HGNet, DLA, DPN, Xception, SelecSLS, HRNet, MobileNet V1–V4, MNASNet, GhostNet/GhostNetV2/V3, RepGhostNet, HardCoReNAS, MLP-Mixer/gMLP/ResMLP, Sequencer2D, PVTv2, VGG, LeViT, SHViT, EfficientFormer/EfficientFormer-V2, MobileViT/MobileViTv2, TinyViT, EdgeNeXt, XCiT, MaxViT, CoAtNet, EfficientViT-MIT, EfficientViT-MSRA, and Visformer families. Every registered name that timm 1.0.30 also defines is listed there. The remaining entries have no timm 1.0.30 model of the same name; some approximate related timm architectures and can differ from them in structure, width, and cost.

### Additions from timm 1.0.30

The following families retain the upstream configurations and parameter counts. See the linked timm implementations for architecture references and original sources.

| Family | Variants | Architecture |
| --- | --- | --- |
| [LowFormer](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/lowformer.py) | 8: `lowformer_b0`, B1, B1.5 (`lowformer_b15`), B2, B3, E1–E3 | Strided convolutional attention with learned upsampling and distinct edge configurations |
| [iFormer](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/iformer.py) | 9: `iformer_t`, S, M, L, L2, H; M/L/L2 with `_distilled` | Single-head modulation attention, convolutional positional encoding, and convolution/FFN stages |
| [EfficientViM](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/efficientvim.py) | 8: `efficientvim_m1`–M4, each with a `_dist` variant | HSM-SSD hidden-state mixing and learned fusion of four classification heads |
| [Qwen3 ViT](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/qwen3_vit.py) | 9: `qwen3_vit_88m`, 306M, 416M; each with `_merge` and `_enc` variants | Learned positions, axial RoPE, and a native 2×2 spatial merger |
| [DeepSeek ViT](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/deepseek_vit.py) | 3: `deepseek_vit_412m`, `deepseek_vit_412m_align`, `deepseek_vit_412m_enc` | FP32 RMSNorm, SwiGLU, axial RoPE, and a channel-major 3×3 aligner |
| [Sapiens2](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/eva.py) | 5: `vit_base`, `vit_large`, `vit_huge`, `vit_giant` and `vit_5b`, all `patch16_sapiens2` | RMSNorm, grouped-query attention in the middle blocks, DINOv3 RoPE, SwiGLU, registers, and attention-only layer scale |

### Weights

Models initialize randomly by default. There are currently no registered downloadable pretrained weights, so `jimm.list_models(pretrained=True)` returns an empty list and `pretrained=True` raises `NotImplementedError`.

Restore trained jimm models with the Orbax checkpoint helpers. `pretrained="/path/to/weights.npz"` and array state dictionaries are also supported through [jimm/weights.py](jimm/weights.py), provided their names and shapes match the converter. Automatic import of timm checkpoints is not implemented; architecture verification does not imply checkpoint compatibility for every family. ImageNet accuracy has not been evaluated for jimm models.

## Features

- **Model APIs:** `create_model`, filtered `list_models`, `list_modules`, and `get_default_cfg`; classification models expose `forward_features`, `forward_head`, and `reset_classifier`.
- **Feature extraction:** unpooled feature maps or tokens, pooled embeddings with `num_classes=0`, and `features_only=True` for supported intermediate stages.
- **NNX transformations:** models work with `nnx.jit` and `nnx.grad`; pass the model as an explicit argument to transformed functions. Use `model.train()` and `model.eval()` to control dropout and batch normalization.
- **Attention:** Tokamax fused kernels for GPU bfloat16 self/cross-attention, including relative-position bias, with heuristic kernel configurations or optional per-shape autotuning. Value heads may differ in width from query/key heads. CPU, float32, and float16 use Flax attention; active attention dropout retains its existing implementation.
- **Grouped convolutions:** `create_model` routes grouped `nnx.Conv` layers through `jimm.layers.conv_general_dilated`, a drop-in for `jax.lax.conv_general_dilated` that computes two kernel gradients XLA lowers poorly on GPU: depthwise kernels as large as their feature map, and dilated grouped convolutions. Apply it to other models with `jimm.layers.use_fast_grouped_conv_grads(model)`.
- **Data and augmentation:** Grain ImageFolder loading, OpenCV decoding, random crops, color jitter, AutoAugment, RandAugment, AugMix, TrivialAugment, random erasing, Mixup, and CutMix. Photometric and AutoAugment operations follow timm's PIL semantics with OpenCV lookup tables, blends, and affine warps (OpenCV rounding can differ from Pillow by a few intensity levels); the training CLI normalizes uint8 batches and applies random erasing, Mixup, and CutMix on device.
- **Training:** AdamW with cosine scheduling and warmup, label smoothing, gradient clipping, optional bfloat16 computation, and JAX SPMD data parallelism or FSDP.
- **Checkpointing:** asynchronous Orbax model/optimizer checkpoints, background deletion of rotated checkpoints, retention settings, and epoch resume with restored data position.
- **Scripts:** timm-style `train.py`, `validate.py`, `inference.py`, `benchmark.py`, checkpoint averaging/cleanup, per-model bulk runs and a multi-host launcher (see [Scripts](#scripts)).

Images have shape **`(batch, height, width, channels)`**. `default_cfg["input_size"]` keeps timm's metadata convention `(channels, height, width)`; use its size and normalization settings when preparing model inputs.

## Getting Started

### Installation

The project requires **Python 3.12 or newer**. On Linux its dependency set includes `jax[cuda13]` for NVIDIA GPU execution; on macOS it installs CPU JAX. With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/imgengineer/jimm.git
cd jimm
uv sync --locked --python 3.12

# Inspect the JAX backend and available devices.
uv run python -c "import jax; print(jax.__version__, jax.devices())"
```

`uv sync` installs the project in editable mode and includes development tools. To force CPU execution with this environment, prefix commands with `JAX_PLATFORMS=cpu`.

Tokamax is installed by default; Triton is installed on Linux with `jax[cuda13]`, while macOS installs CPU JAX without Triton. GPU bfloat16 attention uses [Tokamax's automatic backend selection](https://github.com/openxla/tokamax/blob/main/tokamax/_src/ops/attention/api.py) with heuristic kernel configurations. Pass `--attn-autotune` to the training CLI, or call `jimm.attention.set_attention_autotuning(True)`, to benchmark candidate forward and backward kernels for each new shape instead; later calls in the same process reuse the fastest configuration. Autotuning adds substantial compilation time for every new shape and process and typically improves step time only slightly. Unsupported shapes fall back to XLA. CPU, float32, and float16 keep Flax attention. No extra installation flags or model configuration changes are required.

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

`normalize=False` yields uint8 images and skips normalization and random erasing, reducing worker, inter-process, and host-to-device traffic fourfold. Pass the matching `jimm.preprocess.ImagePreprocess(mean, std, re_prob, re_mode, re_count)` to `make_cached_train_step` or `make_cached_eval_step` to normalize inside the compiled step; training steps then also apply timm random erasing and need an `rng`. The training CLI uses this path.

`in_memory=True` enables a shared decoded-image cache under `~/.cache/jimm/image-cache`, configurable with `JIMM_CACHE_DIR`. It preserves source resolution so random crops and other augmentation still run on each read.

## Training, Validation, and Inference

### Scripts

The repository provides timm-style top-level entry points:

| Script | Purpose |
| --- | --- |
| [train.py](train.py) | Training CLI; equivalent to `python -m jimm.train` |
| [validate.py](validate.py) | Top-1/top-5 validation on an ImageFolder split |
| [inference.py](inference.py) | Top-k predictions written to CSV and/or JSON |
| [benchmark.py](benchmark.py) | Inference and compiled train-step throughput in samples/s |
| [avg_checkpoints.py](avg_checkpoints.py) | Average weights across a run's `CheckpointManager` steps |
| [clean_checkpoint.py](clean_checkpoint.py) | Strip optimizer state into a model-only checkpoint |
| [bulk_runner.py](bulk_runner.py) | Run a script once per model and collect CSV results |
| [distributed_train.sh](distributed_train.sh) | Launch one host of a multi-host JAX run |

```bash
# Validation of a trained checkpoint
uv run python validate.py --data-dir /path/to/dataset --split val \
    --model convnext_tiny --num-classes 1000 --checkpoint ./output/convnext_demo -b 256 --amp

# Top-5 predictions for every validation image
uv run python inference.py --data-dir /path/to/dataset --split val \
    --model convnext_tiny --num-classes 1000 --checkpoint ./output/convnext_demo \
    --topk 5 --results-dir ./results --results-file predictions --results-format csv json

# Inference and train-step throughput
uv run python benchmark.py --model convnext_tiny -b 128 --img-size 224 --bench both --amp

# Weight averaging and model-only checkpoint export
uv run python avg_checkpoints.py --input ./output/convnext_demo --model convnext_tiny \
    --output ./output/convnext_demo-avg -n 5
uv run python clean_checkpoint.py --checkpoint ./output/convnext_demo/10 \
    --model convnext_tiny --output ./output/convnext_demo-clean

# Sweep a wildcard over any script that prints a --result summary
uv run python bulk_runner.py --model-list 'resnet*' --results-file bench.csv \
    benchmark.py --amp -b 64
```

`--checkpoint` accepts either an Orbax checkpoint directory (including a `CheckpointManager` step) or a local `.npz` file. `validate.py` and `inference.py` accept `--prefetch N` (default 2) to queue batches on device and overlap host-to-device copies with compute; `benchmark.py` dispatches its timing iterations asynchronously and synchronizes once. `train.py`, `validate.py`, `inference.py` and `benchmark.py` accept `--compilation-cache DIR` to reuse XLA compilation across runs and processes. `--pretrained` is accepted for timm CLI compatibility but raises an error because jimm registers no downloadable weights. `avg_checkpoints.py` ranks steps by `--metric-key` when the training run saved validation metrics, and falls back to the latest steps otherwise. ONNX export and timm/Hugging Face checkpoint conversion scripts are not provided because jimm checkpoints and models are native JAX/Flax.

### Training and validation

The CLI follows the argument groups and common option names in [timm's `train.py`](https://github.com/huggingface/pytorch-image-models/blob/main/train.py). It uses JAX/Flax NNX training, Grain data loading, AdamW, and a cosine learning-rate schedule. The default `train/` and `val/` ImageFolder splits can be changed with `--train-split` and `--val-split`.

```bash
uv run python -m jimm.train \
    --model convnext_tiny \
    --data-dir /path/to/dataset \
    --num-classes 1000 \
    --img-size 224 \
    --epochs 90 \
    -b 128 -j 4 \
    --opt adamw --sched cosine \
    --lr 5e-4 \
    --aa rand-m9-n2 \
    --mixup 0.8 --cutmix 1.0 \
    --experiment convnext_demo \
    --output ./output
```

Set `--num-classes` to your dataset's class count. The dataset root also works as a positional argument. Inputs default to 224×224 with ImageNet normalization; `--input-size 3 H H`, `--img-size`, `--mean`, `--std`, `--crop-pct`, and the training/validation interpolation options configure preprocessing. Inputs must be square RGB images and match the selected architecture's supported resolution. Use `uv run python -m jimm.train --help` for all options.

Each host defaults to **4 Grain workers** and **bfloat16** compute with Tokamax attention; `--attn-autotune` tunes attention kernels for each new shape. Pass `--compilation-cache DIR` to reuse compiled programs across runs; on multi-host jobs the directory must be shared, and it should be writable only by you because JAX treats cached code as trusted. Workers send uint8 images; normalization and `--reprob` random erasing run on device. Evaluation resizes the shorter edge to `floor(img_size / crop_pct)` and center crops, as in timm. `--no-amp` selects float32. `--validation-batch-size` sets the validation batch size independently; it otherwise follows `--batch-size`.

| timm-style option | Long-form / legacy equivalent |
| --- | --- |
| `-b`, `-vb`, `-j` | `--batch-size`, `--validation-batch-size`, `--workers` |
| `--aa` | `--auto-augment` |
| `--mixup`, `--cutmix` | `--mixup-alpha`, `--cutmix-alpha` |
| `--checkpoint-hist` | `--max-to-keep` |
| `--gp` | `--global-pool` |

`--opt-eps` and `--opt-betas` configure AdamW; `--warmup-epochs`, `--warmup-lr`, and `--min-lr` configure its schedule. By default, warmup takes 10% of training steps, capped at five epochs and 10,000 steps, and the minimum LR is 1% of `--lr`. A one-step run uses a constant LR. Set `--warmup-epochs 0` to disable warmup. `--clip-grad`, `--drop`, `--drop-path`, `--smoothing`, and the Mixup/CutMix options control regularization. `--no-aug` disables image augmentation while retaining the repeating training sampler. Pass additional constructor arguments as `--model-kwargs KEY=VALUE`, for example `--model-kwargs mlp_ratio=3.0` with ViT.

Use `-c` / `--config` for YAML defaults, with explicit CLI arguments taking precedence. For example, save this as `train.yaml`:

```yaml
data_dir: /path/to/dataset
model: convnext_tiny
num_classes: 1000
batch_size: 128
workers: 4
epochs: 90
opt: adamw
sched: cosine
lr: 5e-4
warmup_epochs: 5
amp: true
aa: rand-m9-n2
mixup: 0.8
cutmix: 1.0
output: ./output
experiment: convnext_demo
```

```bash
uv run python -m jimm.train -c train.yaml -b 64 --lr 1e-3
```

Configuration keys accept argument names with underscores or hyphens, including the aliases above. Unknown keys and invalid values are rejected before model or data initialization. Each run saves the resolved configuration to `<output>/<experiment>/args.yaml`; an omitted `--experiment` uses the model name. Epoch checkpoints in the same directory contain both model and optimizer state.

Append `--resume` to restore the latest checkpoint from the output directory, or supply an Orbax manager directory to `--resume PATH`. Keep model, data, batch, sharding, and steps-per-epoch settings consistent when resuming:

```bash
uv run python -m jimm.train -c ./output/convnext_demo/args.yaml --resume
uv run python -m jimm.train -c ./output/convnext_demo/args.yaml \
    --experiment continued --resume ./output/convnext_demo
```

`--checkpoint-hist N` limits checkpoint retention while preserving the best validation checkpoint. `--initial-checkpoint weights.npz` initializes model weights from a jimm NPZ file without restoring the optimizer or epoch.

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

## Testing

Run the regression suite and static checks with the locked environment. Constructing models dominates the suite, so run it in parallel with `pytest-xdist`, which is part of the dev dependencies:

```bash
JAX_PLATFORMS=cpu uv run pytest tests/ -q -n 8
uv run ruff check .
uv run ruff format --check .
```

`ruff format` owns layout and `ruff check` owns everything else: the lint profile in `pyproject.toml` adds the W/UP/B/SIM/PIE/RET/RSE/ISC/FLY/RUF rule families to ruff's default E/F/I set, and `ruff check --fix` applies the safe fixes for all of them. The configured exceptions are deliberate and documented there: B905 is off because `zip(..., strict=True)` would change the runtime behaviour of the ported model code, and `jimm/__init__.py` keeps its area-grouped `__all__` behind a `# noqa: RUF022` instead of an alphabetical flat list. The model modules also keep timm's per-family duplication instead of sharing layer helpers.

Each worker builds real models, so size `-n` by cores *and* memory; on a 14-core CPU eight workers need about 25 minutes, while a serial run takes hours. Eight workers already saturate such a machine: the suite spends roughly eight CPU-seconds per wall-clock second, raising `-n` to 12 shortens the heaviest subset (`-k architecture_family_forward_and_backward`, 6.5 minutes at `-n 8`) by only 3%, and `OMP_NUM_THREADS=1` changes nothing. A shared JAX compilation cache does not help either: JAX caches only programs that take more than a second to compile, and the per-test programs stay below that. `tests/test_models.py` instantiates every registered model (one parametrized case per model name, so the check spreads across workers) and runs a forward and backward pass for one model of each family; those per-family passes are the long poles, since each compiles separate training, gradient, evaluation and feature programs. `tests/test_scripts.py` smoke-tests the top-level scripts against a tiny synthetic ImageFolder. `tests/test_attention.py` covers the Tokamax attention path and requires a GPU:

```bash
uv run pytest tests/test_attention.py -q
```

Standalone scripts check forward passes and gradients; `scripts/test_jimm.py --all` covers every registered model at its configured input size:

```bash
uv run python scripts/test_jimm.py
uv run python scripts/test_backprop.py
```

## Performance

### Training throughput

The table compares compiled jimm training steps with timm 1.0.29 in eager PyTorch 2.10 on an RTX 5090: batch size 64, bfloat16 autocast, AdamW, and label smoothing, at 224×224 unless noted. PyTorch uses channels-last tensors, fused AdamW, and cuDNN benchmarking without `torch.compile`; jimm uses its defaults. Throughput is the median over 20 steps after warmup.

| Model | jimm | timm (eager) | Ratio |
| --- | ---: | ---: | ---: |
| `resnet50` | 2,804 img/s | 2,595 img/s | 1.08× |
| `convnext_tiny` | 2,920 img/s | 2,044 img/s | 1.43× |
| `convnextv2_tiny` | 2,518 img/s | 835 img/s | 3.02× |
| `convnext_nano_ols` | 4,484 img/s | 2,850 img/s | 1.57× |
| `efficientnet_b0` | 6,100 img/s | 3,889 img/s | 1.57× |
| `tf_efficientnetv2_s` | 2,609 img/s | 2,075 img/s | 1.26× |
| `mobilenetv3_large_100` | 9,361 img/s | 6,876 img/s | 1.36× |
| `mobilenetv4_hybrid_medium` | 4,292 img/s | 3,660 img/s | 1.17× |
| `mixnet_m` | 2,557 img/s | 1,999 img/s | 1.28× |
| `lcnet_100` | 16,490 img/s | 12,680 img/s | 1.30× |
| `densenet121` | 2,256 img/s | 1,989 img/s | 1.13× |
| `vit_small_patch16_224` | 4,019 img/s | 3,280 img/s | 1.23× |
| `vit_base_patch16_224` | 1,420 img/s | 1,233 img/s | 1.15× |
| `deit_base_distilled_patch16_224` | 1,394 img/s | 1,235 img/s | 1.13× |
| `vit_base_patch16_clip_224` | 1,419 img/s | 1,235 img/s | 1.15× |
| `hrnet_w18_small` | 4,713 img/s | 3,824 img/s | 1.23× |
| `hrnet_w18` | 914 img/s | 944 img/s | 0.97× |
| `levit_128s` | 9,968 img/s | 7,954 img/s | 1.25× |
| `levit_384` | 3,916 img/s | 3,550 img/s | 1.10× |
| `efficientvit_b1` | 6,116 img/s | 3,969 img/s | 1.54× |
| `visformer_small` | 3,185 img/s | 2,401 img/s | 1.33× |
| `coatnet_0_rw_224` | 2,718 img/s | 1,475 img/s | 1.84× |
| `maxvit_tiny_rw_224` | 1,693 img/s | 979 img/s | 1.73× |
| `tresnet_m` | 3,045 img/s | 2,448 img/s | 1.24× |
| `rdnet_tiny` | 2,275 img/s | 1,536 img/s | 1.48× |
| `skresnet50` | 2,176 img/s | 1,667 img/s | 1.31× |
| `skresnext50_32x4d` | 1,916 img/s | 1,087 img/s | 1.76× |
| `swin_tiny_patch4_window7_224` | 2,850 img/s | 1,677 img/s | 1.70× |
| `swin_s3_tiny_224` | 2,729 img/s | 1,429 img/s | 1.91× |
| `swinv2_cr_tiny_ns_224` | 2,656 img/s | 1,230 img/s | 2.16× |
| `mixer_s16_224` | 5,395 img/s | 3,833 img/s | 1.41× |
| `gmlp_s16_224` | 2,721 img/s | 1,455 img/s | 1.87× |
| `swinv2_tiny_window8_256` | 2,055 img/s | 869 img/s | 2.36× |
| `deit3_small_patch16_224` | 4,009 img/s | 3,111 img/s | 1.29× |
| `cait_xxs24_224` | 3,250 img/s | 1,232 img/s | 2.64× |
| `poolformer_s12` | 5,758 img/s | 2,798 img/s | 2.06× |
| `beit_base_patch16_224` | 1,341 img/s | 1,070 img/s | 1.25× |
| `resnest50d` | 2,158 img/s | 1,255 img/s | 1.72× |
| `hiera_tiny_224` | 2,934 img/s | 1,657 img/s | 1.77× |
| `hieradet_small` (256×256) | 1,701 img/s | 1,175 img/s | 1.45× |
| `dla34` | 4,538 img/s | 3,544 img/s | 1.28× |
| `dpn92` | 912 img/s | 640 img/s | 1.43× |
| `inception_next_tiny` | 2,270 img/s | 1,964 img/s | 1.16× |
| `mambaout_tiny` | 2,882 img/s | 1,661 img/s | 1.74× |
| `repvit_m0_9` | 4,945 img/s | 4,277 img/s | 1.16× |
| `starnet_s1` | 8,684 img/s | 6,476 img/s | 1.34× |
| `rexnet_100` | 5,570 img/s | 3,547 img/s | 1.57× |
| `regnety_008` | 7,046 img/s | 5,293 img/s | 1.33× |
| `pit_s_224` | 4,718 img/s | 3,249 img/s | 1.45× |
| `efficientformer_l1` | 5,446 img/s | 3,588 img/s | 1.52× |
| `senet154` | 482 img/s | 393 img/s | 1.23× |
| `xception41` (299×299) | 1,153 img/s | 897 img/s | 1.29× |
| `selecsls42` | 5,152 img/s | 4,506 img/s | 1.14× |
| `hardcorenas_a` | 10,524 img/s | 6,607 img/s | 1.59× |
| `repghostnet_100` | 6,131 img/s | 5,720 img/s | 1.07× |
| `vovnet39a` | 2,655 img/s | 2,512 img/s | 1.06× |
| `hgnet_tiny` | 3,457 img/s | 2,676 img/s | 1.29× |
| `hgnetv2_b0` | 11,581 img/s | 6,316 img/s | 1.83× |
| `tiny_vit_5m_224` | 4,012 img/s | 3,239 img/s | 1.24× |
| `edgenext_small` (256×256) | 4,170 img/s | 2,625 img/s | 1.59× |
| `xcit_small_12_p16_224` | 2,946 img/s | 1,687 img/s | 1.75× |
| `fasternet_s` | 4,449 img/s | 3,665 img/s | 1.21× |
| `swiftformer_xs` | 7,003 img/s | 4,305 img/s | 1.63× |
| `convit_tiny` | 6,488 img/s | 2,274 img/s | 2.85× |
| `twins_svt_small` | 3,613 img/s | 2,076 img/s | 1.74× |
| `nest_tiny` | 2,280 img/s | 943 img/s | 2.42× |
| `davit_tiny` | 2,744 img/s | 1,721 img/s | 1.59× |
| `focalnet_tiny_srf` | 2,670 img/s | 1,533 img/s | 1.74× |
| `gcvit_tiny` | 1,363 img/s | 885 img/s | 1.54× |
| `mobilevit_s` (256×256) | 2,779 img/s | 1,635 img/s | 1.70× |
| `mvitv2_tiny` | 1,836 img/s | 964 img/s | 1.90× |
| `caformer_s18` | 3,297 img/s | 1,109 img/s | 2.97× |
| `convformer_s18` | 3,173 img/s | 1,074 img/s | 2.95× |
| `poolformerv2_s12` | 6,965 img/s | 1,883 img/s | 3.70× |
| `efficientformerv2_s0` | 4,852 img/s | 1,829 img/s | 2.65× |
| `shvit_s1` | 7,815 img/s | 6,254 img/s | 1.25× |
| `sequencer2d_s` | 884 img/s | 1,224 img/s | 0.72× |
| `nfnet_f0` (192×192) | 2,613 img/s | 1,986 img/s | 1.32× |
| `eca_nfnet_l0` | 3,693 img/s | 2,155 img/s | 1.71× |
| `cspresnet50` (256×256) | 2,806 img/s | 2,182 img/s | 1.29× |
| `cspdarknet53` (256×256) | 2,067 img/s | 1,383 img/s | 1.50× |
| `cs3darknet_l` (256×256) | 3,342 img/s | 2,782 img/s | 1.20× |
| `coat_tiny` | 1,211 img/s | 734 img/s | 1.65× |
| `coat_lite_small` | 1,842 img/s | 1,323 img/s | 1.39× |
| `nextvit_small` | 2,318 img/s | 1,938 img/s | 1.20× |
| `volo_d1_224` | 2,278 img/s | 1,328 img/s | 1.72× |
| `vit_relpos_base_patch16_224` | 1,277 img/s | 1,083 img/s | 1.18× |
| `vitamin_small_224` | 2,022 img/s | 1,401 img/s | 1.44× |
| `fastvit_t8` (256×256) | 2,331 img/s | 2,219 img/s | 1.05× |
| `fastvit_sa12` (256×256) | 1,695 img/s | 1,546 img/s | 1.10× |
| `cpubone_t0` | 12,037 img/s | 8,638 img/s | 1.39× |
| `cpubone_b3` | 1,830 img/s | 1,545 img/s | 1.18× |
| `csatv2` (512×512) | 4,329 img/s | 1,438 img/s | 3.01× |
| `mobilenetv5_base` (256×256, batch 32) | 945 img/s | 244 img/s | 3.87× |
| `nasnetalarge` (331×331, batch 32) | 349 img/s | 229 img/s | 1.53× |
| `gemma4_vit_167m` | 638 img/s | 320 img/s | 1.99× |
| `naflexvit_base_patch16_gap` | 1,221 img/s | 1,165 img/s | 1.05× |
| `tnt_s_patch16_224` | 1,701 img/s | 1,125 img/s | 1.51× |
| `crossvit_15_240` (240×240) | 2,695 img/s | 1,949 img/s | 1.38× |
| `samvit_base_patch16_224` | 1,298 img/s | 759 img/s | 1.71× |
| `resnet50d` | 2,731 img/s | 2,355 img/s | 1.16× |
| `resnetaa50d` | 2,591 img/s | 2,133 img/s | 1.21× |
| `vit_base_r50_s16_224` | 752 img/s | 616 img/s | 1.22× |
| `vit_small_r26_s32_224` | 1,929 img/s | 1,452 img/s | 1.33× |
| `resnetv2_50d` | 3,339 img/s | 2,525 img/s | 1.32× |
| `resnetv2_50x1_bit` | 1,788 img/s | 1,429 img/s | 1.25× |
| `maxvit_tiny_tf_224` | 1,516 img/s | 899 img/s | 1.69× |
| `coatnet_0_224` | 1,538 img/s | 1,209 img/s | 1.27× |
| `eva02_small_patch14_224` | 2,945 img/s | 1,850 img/s | 1.59× |
| `vit_base_patch16_rope_224` | 1,398 img/s | 995 img/s | 1.40× |
| `repvgg_b0` | 4,509 img/s | 3,691 img/s | 1.22× |
| `mobileone_s1` | 3,958 img/s | 3,285 img/s | 1.20× |
| `regnetz_b16` | 3,941 img/s | 2,598 img/s | 1.52× |
| `resnet50_clip` | 2,398 img/s | 1,974 img/s | 1.21× |
| `resnet33ts` (256×256) | 2,904 img/s | 2,338 img/s | 1.24× |
| `gcresnext26ts` (256×256) | 3,418 img/s | 2,570 img/s | 1.33× |
| `halonet26t` (256×256) | 3,302 img/s | 2,368 img/s | 1.39× |
| `botnet26t_256` (256×256) | 3,405 img/s | 2,461 img/s | 1.38× |
| `regnety_040` | 2,986 img/s | 2,029 img/s | 1.47× |
| `regnetv_040` | 2,885 img/s | 1,952 img/s | 1.48× |
| `regnety_040_sgn` | 2,795 img/s | 704 img/s | 3.97× |
| `regnetz_040` | 1,467 img/s | 773 img/s | 1.90× |

### Startup and compilation

A cold `create_model` is dominated by XLA compilation rather than by the initializers themselves: every unique parameter shape compiles its own RNG-init kernel on first use. On a fresh CPU process:

| Model | cold `create_model` | warm persistent cache | second create, same process |
| --- | ---: | ---: | ---: |
| `resnet18` | 1.78 s | 0.24 s | 0.06 s |
| `convnext_tiny` | 2.54 s | 0.44 s | 0.21 s |
| `nfnet_f0` | 4.32 s | 0.64 s | 0.33 s |
| `mobilevit_s` | 6.54 s | 0.74 s | 0.08 s |

The cache is keyed by the compiled program, so it warms per model configuration (`num_classes`, input size, dtype, sharding). Point it at a directory with `--compilation-cache DIR` or `JAX_COMPILATION_CACHE_DIR` and reuse it across runs; the helper also sets `jax_persistent_cache_min_compile_time_secs=0` so even short compilations are stored. Fusing all initializers into a single compiled program was measured 2–2.8× faster for small and medium models but 1.7–8.5× slower for large ones (`sequencer2d_s`, `mobilenetv5_300m`), because XLA compile time grows faster than linearly with the number of operations in one program, so jimm keeps eager initialization and relies on the persistent cache instead.

### Attention

GPU bfloat16 attention dispatches to Tokamax fused kernels, which reduce attention time and compiled temporary memory relative to Flax attention; results depend on shapes and hardware. To measure them on your hardware:

```bash
uv run python scripts/benchmark_attention.py \
    --seq-len 2304 --iterations 10 --autotune --output attention-benchmark.json
```

## Repository Layout

```text
jimm/
  registry.py       Model creation, registration, and configuration
  models/           Architecture implementations and shared helpers
  layers.py         Common NNX layers and classifier utilities
  attention.py      Shared attention and Tokamax dispatch
  features.py       Intermediate feature extraction
  data.py           ImageFolder datasets and Grain loaders
  augment.py        Image augmentation and policies
  train.py          Optimizer, train/eval steps, and the training loop
  train_cli.py      Training command-line parsing and configuration files
  preprocess.py     Device-side normalization, random erasing, and Mixup/CutMix
  prefetch.py       Double-buffered host-to-device batch prefetching
  checkpoint.py     Orbax model and optimizer checkpointing
  weights.py        Array state-dict conversion and NPZ loading
  scripts_common.py Shared argument, data, model and metric helpers
train.py            timm-style training entry point
validate.py         Standalone validation script
inference.py        Top-k inference script
benchmark.py        Throughput benchmark script
avg_checkpoints.py  Checkpoint weight averaging
clean_checkpoint.py Model-only checkpoint export
bulk_runner.py      Per-model script launcher
distributed_train.sh Multi-host JAX launcher
tests/              Regression and model tests
scripts/            Forward/backpropagation checks and attention benchmark
pyproject.toml      Project metadata and dependency requirements
uv.lock             Resolved dependency versions
```

## Licenses and Acknowledgments

Code is licensed under [Apache License 2.0](LICENSE). [NOTICE](NOTICE) retains attribution and applicable MIT notices for the adapted implementations.

Thanks to [Ross Wightman and the timm contributors](https://github.com/huggingface/pytorch-image-models) and the original architecture authors. The model implementations follow timm, and this document follows the organization of [its README](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/README.md). External model weights retain their own licenses; consult their original model cards before use.
