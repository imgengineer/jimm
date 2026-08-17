# jimm — JAX Image Models

**jimm** is a comprehensive JAX / Flax NNX implementation of the popular [`timm` (pytorch-image-models)](https://github.com/huggingface/pytorch-image-models) library.

It provides **1,344 registered model architectures across 94 model families** (100% coverage of timm's 1,309 entrypoints), mirroring `timm`'s API surface with a pure JAX native design.

---

## Key Highlights

- **Flax NNX Native**: Built on modern [Flax NNX](https://flax.readthedocs.io/), supporting both Python object-oriented model definitions and functional JAX transformations (`nnx.jit`, `nnx.grad`, `nnx.split`, etc.).
- **NHWC Layout Throughout**: Native NHWC (channels-last) tensor layout optimized for JAX, XLA, NVIDIA Tensor Cores, and TPUs.
- **Consistent `timm` API**:
  - `jimm.create_model(name, num_classes=..., drop_rate=..., features_only=..., pretrained=...)`
  - `jimm.list_models(filter=..., module=...)`
  - `jimm.list_modules()`
  - `jimm.get_default_cfg(name)`
  - `model.forward_features(x)`: unpooled feature map extraction (or multi-scale features with `features_only=True`)
  - `model.forward_head(feats)`: global pooling + classifier
  - `model.reset_classifier(num_classes, global_pool=...)`: in-place feature extractor conversion
  - `model.default_cfg`: input size, normalization mean/std, interpolation settings
- **Modern Ecosystem Integration**:
  - **Grain**: High-throughput multi-worker data loading pipeline (`jimm.data.create_loader`).
  - **Optax**: Scalable optimization with cosine schedules, label smoothing, and weight decay (`jimm.train`).
  - **Orbax-Checkpoint**: Robust asynchronous model and optimizer state checkpointing (`jimm.checkpoint`).

---

## Installation & Environment Setup

Using `uv` with proxy support for fast package downloads:

```bash
# Set up proxy (optional)
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890

# Create virtual environment and sync dependencies
uv venv --python 3.13 .venv
source .venv/bin/activate
uv sync
```

Dependencies in `pyproject.toml`:

- `jax[cuda13]`
- `flax >= 0.12.0`
- `grain`
- `optax`
- `orbax-checkpoint`
- `pillow`
- `numpy`

---

## Quickstart

### 1. Create and Inspect Models

```python
import jax.numpy as jnp
from flax import nnx
import jimm

# List available architectures
print("Total models:", len(jimm.list_models()))
print("Modules:", jimm.list_modules())
print("ResNet models:", jimm.list_models("resnet*"))

# Create any model (e.g. ConvNeXt, Swin, ViT, EfficientNet, MaxViT, EVA, Hiera, etc.)
model = jimm.create_model(
    "convnext_tiny",
    num_classes=1000,
    drop_path_rate=0.1,
    rngs=nnx.Rngs(0)
)
model.eval()

# Forward pass (NHWC: Batch, Height, Width, Channels)
x = jnp.zeros((2, 224, 224, 3), jnp.float32)
logits = model(x)
print("Logits shape:", logits.shape) # (2, 1000)

# Unpooled feature extraction
features = model.forward_features(x)
print("Features shape:", features.shape) # (2, 7, 7, 768)

# Convert to feature extractor
model.reset_classifier(0)
pooled_feats = model(x)
print("Pooled feature dimension:", pooled_feats.shape[-1]) # 768
```

---

### 2. High-Performance Data Pipeline (Grain)

`jimm.data` provides a Grain pipeline reading standard `ImageFolder` directories. Images are decoded and augmented with OpenCV/NumPy, avoiding per-image JAX dispatch overhead:

```python
from jimm.data import create_loader

# Create train & val loaders
train_loader = create_loader(
    root="/path/to/dataset/train",
    batch_size=128,
    img_size=224,
    is_training=True,
    auto_augment="rand-m9-n2",
    grayscale_prob=0.1,
    gaussian_blur_prob=0.1,
    num_workers=8
)

val_loader = create_loader(
    root="/path/to/dataset/val",
    batch_size=128,
    img_size=224,
    is_training=False,
    num_workers=4
)

# Fetch a batch
batch = next(iter(train_loader))
print(batch["image"].shape)  # (128, 224, 224, 3) float32
print(batch["label"].shape)  # (128,) int32
```

For the official AlbumentationsX backend, install a matching PyTorch build
first, then the optional profile. AlbumentationsX is AGPL-3.0-only:

```bash
# Install the PyTorch build appropriate for the host, then:
uv sync --extra albumentationsx
```

```python
train_loader = create_loader(
    root="/path/to/dataset/train",
    batch_size=128,
    img_size=224,
    is_training=True,
    augmentation_backend="albumentationsx",
)
```

The default remains the lightweight OpenCV backend when the optional profile is
not installed.

---

### 3. Distributed Training with SPMD & Optax

`jimm.train` natively supports single-GPU, single-node multi-GPU, and multi-node multi-GPU training with JAX SPMD Data Mesh and Grain process sharding.

#### Single-Device or Single-Node Multi-GPU

JAX automatically detects all available GPUs on the node and partitions data across them:

```bash
# Standard DDP (Replicated Parameters across all local GPUs):
python -m jimm.train \
    --model convnext_tiny \
    --data-dir /path/to/imagenet \
    --epochs 90 \
    --batch-size 128 \
    --lr 1e-3 \
    --smoothing 0.1 \
    --output ./checkpoints

# FSDP Mode (ZeRO-3: shards parameters & optimizer state across all GPUs to save memory):
python -m jimm.train \
    --model eva_large_patch16_224 \
    --data-dir /path/to/imagenet \
    --fsdp \
    --batch-size 128
```

#### Multi-Node Multi-GPU (e.g. 2 Nodes, 8 GPUs each)

On Master Node (Rank 0, IP `192.168.1.100`):

```bash
python -m jimm.train \
    --model convnext_tiny \
    --data-dir /path/to/imagenet \
    --dist-coordinator-address 192.168.1.100:12345 \
    --dist-num-processes 2 \
    --dist-process-id 0 \
    --batch-size 128
```

On Worker Node (Rank 1):

```bash
python -m jimm.train \
    --model convnext_tiny \
    --data-dir /path/to/imagenet \
    --dist-coordinator-address 192.168.1.100:12345 \
    --dist-num-processes 2 \
    --dist-process-id 1 \
    --batch-size 128
```

Or via standard SLURM / MPI cluster managers (JAX auto-detects `SLURM_JOB_ID` / `JAX_COORDINATOR_ADDRESS`):

```bash
srun -N 4 --ntasks-per-node=1 python -m jimm.train --model resnet50 --data-dir /path/to/imagenet
```

#### Programmatic Distributed Training (DDP & FSDP)

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import jimm
from jimm.train import make_optimizer, train_step, fsdp_shard_model

# Setup 1D Data-Parallel Device Mesh
mesh = jax.sharding.Mesh(jax.devices(), ('data',))
P = jax.sharding.PartitionSpec
data_sharding = jax.sharding.NamedSharding(mesh, P('data', None, None, None))
label_sharding = jax.sharding.NamedSharding(mesh, P('data',))

model = jimm.create_model("resnet50", num_classes=10, rngs=nnx.Rngs(0))
model.train()
optimizer = make_optimizer(model, lr=1e-3, weight_decay=0.05, epochs=90, steps_per_epoch=1000)

# --- DDP (default): weights replicated on every device, nothing to do ---
# --- FSDP (ZeRO-3): shard parameters & optimizer state across the mesh ---
fsdp_shard_model(model, mesh)
fsdp_shard_model(optimizer, mesh)

# Feed process-local batch slices
images = jax.make_array_from_process_local_data(data_sharding, batch["image"])
labels = jax.make_array_from_process_local_data(label_sharding, batch["label"])

loss, acc = train_step(model, optimizer, images, labels, smoothing=0.1)
```

---

### 4. Checkpoint Save and Restore (Orbax)

```python
from jimm.checkpoint import save_checkpoint, load_checkpoint

# Save model and optimizer
save_checkpoint("./checkpoints/epoch_1", model, optimizer, epoch=1)

# In-place restore into a fresh model
new_model = jimm.create_model("resnet50", num_classes=10, rngs=nnx.Rngs(0))
epoch = load_checkpoint("./checkpoints/epoch_1", new_model)
print("Restored epoch:", epoch)
```

---

## Supported Architectures (341 Models across 94 Families)

| Family | Key Variants | Description |
| --- | --- | --- |
| **ResNet / ResNeXt** | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d` | Deep Residual Networks |
| **ResNetV2** | `resnetv2_50`, `resnetv2_101`, `resnetv2_152` | Pre-Activation ResNet |
| **SE-ResNet / SENet** | `seresnet50`, `seresnext50_32x4d`, `senet154` | Squeeze-and-Excitation Networks |
| **Res2Net** | `res2net50_26w_4s`, `res2net50_14w_8s`, `res2net101_26w_4s` | Multi-scale hierarchical residual representations |
| **ResNeSt** | `resnest14d`, `resnest50d`, `resnest101e` | Split-Attention Networks |
| **SKNet** | `skresnet50`, `skresnet101` | Selective Kernel Networks |
| **RegNet** | `regnetx_002`..`032`, `regnety_004`..`032` | Designed with exact RegNet design space algorithm |
| **ConvNeXt / ConvNeXt-V2** | `convnext_atto`..`large`, `convnextv2_atto`..`huge` | Modern pure ConvNets with Global Response Normalization |
| **Vision Transformer (ViT)** | `vit_tiny`..`large_patch16_224` | Original Vision Transformer |
| **DeiT / DeiT-III / BEiT** | `deit_tiny`..`base`, `deit3_small`..`large`, `beit_base`..`large` | Distilled & Masked Pretrained ViT variants |
| **Swin / Swin-V2 / Swin-V2-CR** | `swin_tiny`..`base`, `swinv2_tiny`..`small_256`, `swinv2_cr_tiny`..`giant` | Hierarchical Vision Transformer using Shifted Windows |
| **Hiera / HieraDet / SAM-2** | `hiera_tiny`..`huge_224`, `sam2_hiera_tiny`..`large`, `hieradet_small` | Hierarchical Vision Transformer with MAE & SAM2 support |
| **VOLO** | `volo_d1_224`..`volo_d5_224` | Vision Outlooker (Outlook Attention + Transformer) |
| **EVA** | `eva_small_patch16_224`..`eva_large` | ViT with SwiGLU, LayerScale, and 2D RoPE |
| **CaiT** | `cait_xxs24_224`..`cait_m36_224` | Class-Attention in Image Transformers |
| **XCiT** | `xcit_tiny`..`medium_24_p16_224` | Cross-Covariance Image Transformers |
| **MaxViT / CoAtNet** | `maxvit_tiny`..`base_rw_224`, `coatnet_0`..`2_rw_224` | Multi-Axis Vision Transformer (MBConv + Window/Grid Attn) |
| **PVT-v2** | `pvt_v2_b0`..`pvt_v2_b5` | Pyramid Vision Transformer with Linear Spatial Reduction |
| **Twins** | `twins_svt_small`, `twins_svt_base`, `twins_svt_large` | Spatially Viable Transformer (Local + Global Attention) |
| **LeViT / PiT / TNT / ConViT** | `levit_128s`..`256`, `pit_ti`..`b_224`, `tnt_tiny`..`small`, `convit_tiny`..`base` | Hybrid & Gated Positional Attention Transformers |
| **DaViT / GCViT / TinyViT / CoaT** | `davit_tiny`..`base`, `gcvit_tiny`..`base`, `tiny_vit_5m`..`21m`, `coat_tiny`..`mini` | Dual-Attention, Global Context, and Co-Scale Transformers |
| **MobileViT / EfficientViT / FastViT / RepViT** | `mobilevit_xxs`..`s`, `efficientvit_b0`..`b2`, `fastvit_t8`..`s12`, `repvit_m0_9`..`m1_5` | Lightweight mobile vision transformers & reparameterized blocks |
| **EfficientFormer / EfficientFormer-V2 / MetaFormer** | `efficientformer_l1`..`l7`, `efficientformerv2_s0`..`l`, `caformer_s18`..`b36` | Conv + Transformer meta-architectures |
| **EfficientNet / TinyNet** | `efficientnet_b0`..`b7`, `tinynet_a`..`e` | Compound-scaled Mobile Inverted Bottlenecks |
| **MobileNetV2 / MobileNetV3 / MobileNetV5 / MNASNet** | `mobilenetv2_050`..`140`, `mobilenetv3_large`/`small`, `mobilenetv5_300m`..`base`, `mnasnet_050`..`140` | Efficient mobile convolutional networks |
| **Inception-v3 / Inception-v4 / Inception-ResNet-v2 / InceptionNeXt** | `inception_v3`, `inception_v4`, `inception_resnet_v2`, `inception_next_atto`..`small` | Factorized & parallel multi-branch convolution networks |
| **DenseNet / DLA / DPN / HRNet / VoVNet** | `densenet121`..`201`, `dla34`..`169`, `dpn68`..`131`, `hrnet_w18`..`w48`, `vovnet39a`..`57a` | Dense connectivity, deep aggregation, dual-path, and high-resolution streams |
| **DarkNet / CSPNet / CSPDarkNet / CSPResNet** | `darknet53`, `cspdarknet53`, `cspresnet50`, `cspresnext50` | Cross-Stage-Partial architectures |
| **VGG / SqueezeNet / ShuffleNetV2 / Xception** | `vgg11_bn`..`19_bn`, `squeezenet1_0`/`1_1`, `shufflenetv2_x0_5`..`x2_0`, `xception`, `xception41`/`65` | Classic deep vision backbones |
| **FasterNet / StarNet / RepGhost / ReXNet / SelecSLS / LCNetV2** | `fasternet_t0`..`s`, `starnet_s050`..`s2`, `repghostnet_050`..`130`, `rexnet_100`..`200`, `selecsls42`/`60`, `lcnetv2_050`..`150` | Partial conv, elementwise-star, and linear-bottleneck efficient designs |
| **NFNet / TResNet / CPUBone / HardCoReNAS** | `nfnet_f0`..`f3`, `tresnet_m`..`xl`, `cpubone_nano`..`b3`, `hardcorenas_a`/`f` | Normalizer-free scaled weight standardization, Space-to-Depth stems, CPU backbones |
| **MLP-Mixer / ResMLP / PoolFormer / ConvMixer / Sequencer2D** | `mixer_b16`..`l16`, `resmlp_12`..`36`, `poolformer_s12`..`s36`, `convmixer_768_32`..`1536_20`, `sequencer2d_s`/`m` | Token & spatial mixing architectures |
| **Gemma4-ViT / NAFlexViT / ViT-SAM / ViTAMIN / SHViT / SwiftFormer** | `gemma4_vit_167m`..`570m`, `naflexvit_base_patch16_gap`..`so400m`, `vit_sam_base`, `vitamin_small`..`large2`, `shvit_t1`..`s2`, `swiftformer_xs`..`l1` | 2D RoPE, flexible resolution, Segment Anything encoders, and single-head transformers |
| **ByoaNet / ByobNet** | `byoanet_pv1`..`s`, `byobnet_s`..`l` | Build-Your-Own-Architecture flexible networks |

---

## Testing & Verification

### Unit Test Suite & Code Coverage (>99.6% Coverage)

Run the full pytest suite with code coverage across all core modules and 94 architecture families:

```bash
# Run all unit tests with pytest & coverage report
pytest tests/ --cov=jimm --cov-report=term-missing

# Or run parallel testing across all CPU cores
pytest tests/ -n auto --cov=jimm
```

### Self-Check Regression Suite

```bash
# Run core test suite (representative models across all families + SPMD + FSDP + data + checkpointing)
python test_jimm.py

# Run backpropagation numerical gradient & convergence verification
python test_backprop.py

# Or run exhaustive check across all 341 models
python test_jimm.py --all
```

---

## Repository Structure

```text
jimm/
  __init__.py       Package exports & version
  registry.py       create_model, list_models, list_modules, register_model
  layers.py         DropPath, PatchEmbed, Mlp, SqueezeExcite, ConvBNAct, global_pool
  data.py           Grain ImageFolder & DataLoader pipeline
  train.py          Optax optimizer, loss functions, JIT train/eval steps
  checkpoint.py     Orbax async checkpointer save/restore
  models/           94 Architecture implementation files (341 registered variants)
test_jimm.py        End-to-end test suite
pyproject.toml      uv project configuration & dependencies
```

---

## License

Apache 2.0
