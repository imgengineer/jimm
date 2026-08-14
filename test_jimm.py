"""End-to-end self-check for jimm. Run: .venv/bin/python test_jimm.py"""
import os
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

import jimm
from jimm import create_model, list_models


def check_registry():
    names = list_models()
    assert {"resnet18", "resnet50", "vit_base_patch16_224"} <= set(names), names
    assert list_models("resnet*") == [n for n in names if n.startswith("resnet")]
    assert "resnet" in jimm.list_modules()
    try:
        create_model("resnet18", pretrained=True)
        raise AssertionError("pretrained=True should raise")
    except NotImplementedError:
        pass
    print("registry OK:", names)


def check_models():
    x = jnp.zeros((2, 224, 224, 3), jnp.float32)  # NHWC
    for name, feat_dim in [("resnet18", 512), ("vit_tiny_patch16_224", 192)]:
        m = create_model(name, num_classes=10)
        m.eval()
        logits = m(x)
        assert logits.shape == (2, 10), (name, logits.shape)
        feats = m.forward_features(x)
        assert feats is not None
        m.reset_classifier(0)
        feats_head = m(x)
        assert feats_head.shape[-1] == feat_dim, (name, feats_head.shape)
        print(f"{name} OK, logits {logits.shape}, devices {jax.devices()}")


REPRESENTATIVE_MODELS = [
    "resnet18", "vit_tiny_patch16_224", "swin_tiny_patch4_window7_224",
    "convnext_tiny", "efficientnet_b0"
]


def check_all_models_forward(mode="representative"):
    """Test forward pass + feature mode across models."""
    if mode == "all":
        models_to_test = list_models()
        print(f"Exhaustive test across all {len(models_to_test)} registered models...")
    elif mode == "modules":
        modules = sorted(jimm.list_modules())
        models_to_test = [jimm.list_models(module=m)[0] for m in modules]
        print(f"Testing {len(models_to_test)} models (1 per architecture module)...")
    else:
        models_to_test = REPRESENTATIVE_MODELS
        print(f"Testing {len(models_to_test)} core representative models...")

    for i, name in enumerate(models_to_test):
        m = create_model(name, num_classes=7)
        size = m.default_cfg.get("input_size", (3, 224, 224))[1]
        x = jnp.zeros((1, size, size, 3), jnp.float32)
        m.eval()
        logits = m(x)
        assert bool(jnp.isfinite(logits).all()), f"NaN in {name}"
        if m.get_classifier() is None:  # encoder-only models (e.g. vit_sam): feature maps in, features out
            assert logits.shape[-1] == m.num_features, (name, logits.shape)
            continue
        assert logits.shape == (1, 7), (name, logits.shape)
        m.reset_classifier(0)
        feats = m(x)
        assert feats.shape[-1] == m.num_features, (name, feats.shape, m.num_features)
        print(f"  [{i+1:>2}/{len(models_to_test)}] {name:<30} OK")


def check_train_step():
    from jimm.train import train_step, make_optimizer, cross_entropy
    mesh = jax.sharding.Mesh(jax.devices(), ('data',))
    P = jax.sharding.PartitionSpec
    data_sharding = jax.sharding.NamedSharding(mesh, P('data', None, None, None))
    label_sharding = jax.sharding.NamedSharding(mesh, P('data',))

    m = create_model("resnet18", num_classes=5)
    m.train()
    opt = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10)
    raw_images = np.random.randn(4, 224, 224, 3).astype(np.float32)
    raw_labels = np.array([0, 1, 2, 3], np.int32)
    images = jax.make_array_from_process_local_data(data_sharding, raw_images)
    labels = jax.make_array_from_process_local_data(label_sharding, raw_labels)
    loss1, acc1 = train_step(m, opt, images, labels, 0.1)
    loss2, _ = train_step(m, opt, images, labels, 0.1)
    assert jnp.isfinite(loss1) and jnp.isfinite(loss2), (loss1, loss2)
    l = cross_entropy(jnp.array([[10.0, 0.0]]), jnp.array([0]), 0.1)
    assert 0 < float(l) < 1.0
    print(f"train_step (SPMD Mesh) OK, loss {float(loss1):.3f} -> {float(loss2):.3f}")


def check_fsdp_step():
    from jimm.train import train_step, make_optimizer, fsdp_shard_model
    mesh = jax.sharding.Mesh(jax.devices(), ('data',))
    P = jax.sharding.PartitionSpec
    data_sharding = jax.sharding.NamedSharding(mesh, P('data', None, None, None))
    label_sharding = jax.sharding.NamedSharding(mesh, P('data',))

    m = create_model("convnext_tiny", num_classes=5)
    m.train()
    opt = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10)
    fsdp_shard_model(m, mesh)
    fsdp_shard_model(opt, mesh)

    # assert FSDP actually shards large weights along the mesh axis
    P = jax.sharding.PartitionSpec
    n_sharded = sum(1 for _, node in nnx.graph.iter_graph(m)
                    if isinstance(node, nnx.Variable)
                    and isinstance(node.get_value(), jax.Array)
                    and hasattr(node.get_value().sharding, "spec")
                    and node.get_value().sharding.spec == P('data', None))
    assert n_sharded > 0, "FSDP sharded no variables"
    raw_images = np.random.randn(4, 224, 224, 3).astype(np.float32)
    raw_labels = np.array([0, 1, 2, 3], np.int32)
    images = jax.make_array_from_process_local_data(data_sharding, raw_images)
    labels = jax.make_array_from_process_local_data(label_sharding, raw_labels)

    loss1, acc1 = train_step(m, opt, images, labels, 0.1)
    loss2, acc2 = train_step(m, opt, images, labels, 0.1)
    assert jnp.isfinite(loss1) and jnp.isfinite(loss2), (loss1, loss2)
    print(f"train_step (FSDP ZeRO-3) OK, loss {float(loss1):.3f} -> {float(loss2):.3f}")


def check_data_and_ckpt():
    from jimm.data import create_loader
    from jimm.checkpoint import save_checkpoint, load_checkpoint
    root = tempfile.mkdtemp()
    try:
        for split in ["train", "val"]:
            for cls in ["a", "b"]:
                os.makedirs(f"{root}/{split}/{cls}")
                for i in range(6):
                    Image.fromarray(np.random.randint(0, 255, (64, 48, 3)).astype(np.uint8)).save(
                        f"{root}/{split}/{cls}/{i}.png")
        loader = create_loader(f"{root}/train", 4, img_size=32, is_training=True, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 32, 32, 3), batch["image"].shape
        assert batch["label"].shape == (4,)
        val = list(create_loader(f"{root}/val", 4, img_size=32, is_training=False, num_workers=0))
        assert len(val) == 3  # 12 images / 4 per batch
        print("data OK:", batch["image"].shape, "val batches:", len(val))

        m = create_model("resnet18", num_classes=2)
        m.eval()
        out_before = m(batch["image"])
        path = save_checkpoint(f"{root}/ckpt", m, epoch=3)
        # corrupt then restore
        nnx.replace_by_pure_dict(nnx.state(m), jax.tree.map(
            lambda a: np.zeros_like(a), nnx.state(m).to_pure_dict()))
        epoch = load_checkpoint(path, m)
        assert epoch == 3
        np.testing.assert_allclose(np.asarray(m(batch["image"])), np.asarray(out_before),
                                   rtol=1e-4, atol=1e-4)
        print("checkpoint OK, roundtrip restores exact outputs")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    import sys
    mode = "all" if "--all" in sys.argv else ("modules" if "--modules" in sys.argv else "representative")
    check_registry()
    check_models()
    check_all_models_forward(mode=mode)
    check_train_step()
    check_fsdp_step()
    check_data_and_ckpt()
    print("ALL CHECKS PASSED")
