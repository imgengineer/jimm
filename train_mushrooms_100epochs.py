"""High performance training of mainstream vision models on Mushrooms dataset for 100 epochs."""
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image
import jax.numpy as jnp
from flax import nnx

import jimm
from jimm.checkpoint import save_checkpoint
from jimm.train import make_optimizer, make_cached_train_step, make_cached_eval_step

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_dataset_to_ram(split_dir: Path, img_size: int = 224):
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    imgs, labels = [], []
    for c in classes:
        for f in sorted((split_dir / c).glob('*.*')):
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                try:
                    im = Image.open(f).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
                    imgs.append(np.asarray(im, dtype=np.uint8))
                    labels.append(cls_to_idx[c])
                except Exception:
                    pass
    x = np.stack(imgs, axis=0)
    y = np.array(labels, dtype=np.int32)
    return x, y, classes


def train_single_model(model_name: str, train_x: np.ndarray, train_y: np.ndarray,
                       val_x: np.ndarray, val_y: np.ndarray, num_classes: int = 9,
                       epochs: int = 100, batch_size: int = 64, lr: float = 1e-3,
                       weight_decay: float = 0.01, smoothing: float = 0.1,
                       out_dir: str = "./checkpoints"):
    print(f"\n=======================================================", flush=True)
    print(f"  Training {model_name} (100 Epochs, lr={lr}, wd={weight_decay}, bs={batch_size})", flush=True)
    print(f"=======================================================", flush=True)

    model = jimm.create_model(model_name, num_classes=num_classes, rngs=nnx.Rngs(0))
    n_train = len(train_x)
    n_val = len(val_x)
    steps_per_epoch = max(1, n_train // batch_size)

    opt = make_optimizer(model, lr=lr, weight_decay=weight_decay, epochs=epochs,
                         steps_per_epoch=steps_per_epoch, clip_grad=1.0)

    model.train()
    cached_train_step = make_cached_train_step(model, opt)
    model.eval()
    cached_eval_step = make_cached_eval_step(model)
    model.train()

    best_val_acc = 0.0
    best_epoch = 0
    history = []
    train_loss, train_acc = 0.0, 0.0
    val_loss, val_acc = 0.0, 0.0
    t_start = time.time()

    val_steps = -(-n_val // batch_size)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = np.random.permutation(n_train)
        loss_sum = jnp.zeros(())
        acc_sum = jnp.zeros(())

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            batch_img = train_x[idx].astype(np.float32) / 255.0
            if np.random.rand() > 0.5:
                batch_img = batch_img[:, :, ::-1, :]
            batch_img = (batch_img - IMAGENET_MEAN) / IMAGENET_STD
            batch_lbl = train_y[idx]

            l, a = cached_train_step(jnp.asarray(batch_img), jnp.asarray(batch_lbl), smoothing)
            loss_sum = loss_sum + l
            acc_sum = acc_sum + a

        try:
            train_loss = float(loss_sum) / max(steps_per_epoch, 1)
            train_acc = float(acc_sum) / max(steps_per_epoch, 1)
        except Exception:
            train_loss, train_acc = 0.0, 0.0

        # Validation
        v_loss_sum = jnp.zeros(())
        v_acc_sum = jnp.zeros(())
        for v_step in range(val_steps):
            v_idx = slice(v_step * batch_size, min((v_step + 1) * batch_size, n_val))
            v_img = (val_x[v_idx].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            v_lbl = val_y[v_idx]
            l, a = cached_eval_step(jnp.asarray(v_img), jnp.asarray(v_lbl))
            v_loss_sum = v_loss_sum + l * len(v_lbl)
            v_acc_sum = v_acc_sum + a * len(v_lbl)

        try:
            val_loss = float(v_loss_sum) / max(n_val, 1)
            val_acc = float(v_acc_sum) / max(n_val, 1)
        except Exception:
            val_loss, val_acc = 0.0, 0.0
        epoch_time = time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(f"{out_dir}/{model_name}_best", model, opt, epoch=epoch,
                            extra={"val_acc": val_acc, "train_acc": train_acc})

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
        })

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"  [{model_name}] Epoch [{epoch:>3}/{epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc*100:5.1f}% | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc*100:5.1f}% "
                  f"(best: {best_val_acc*100:5.1f}% @ep{best_epoch}) "
                  f"[{epoch_time:.2f}s/ep]", flush=True)

    save_checkpoint(f"{out_dir}/{model_name}_final", model, opt, epoch=epochs,
                    extra={"val_acc": val_acc, "best_val_acc": best_val_acc})
    total_time = time.time() - t_start
    print(f"\n✓ Completed {model_name} in {total_time:.1f}s ({total_time/60:.2f} mins). "
          f"Best Val Acc: {best_val_acc*100:.2f}% (Epoch {best_epoch})\n", flush=True)

    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_val_acc": val_acc,
        "final_train_acc": train_acc,
        "final_val_loss": val_loss,
        "total_time_sec": total_time,
        "history": history,
    }


def main():
    split_dir = Path("/home/lzc/Documents/jimm/datasets/mushrooms_split")
    print("Loading Mushroom dataset into RAM...", flush=True)
    t0 = time.time()
    train_x, train_y, classes = load_dataset_to_ram(split_dir / "train")
    val_x, val_y, _ = load_dataset_to_ram(split_dir / "val")
    print(f"Loaded {len(train_x)} train and {len(val_x)} val images across {len(classes)} classes in {time.time()-t0:.1f}s.")

    models_config = [
        ("resnet18", 1e-3, 0.01),
        ("mobilenetv3_large_100", 1e-3, 0.01),
        ("convnext_tiny", 5e-4, 0.05),
    ]

    all_results = {}
    for name, lr, wd in models_config:
        res = train_single_model(name, train_x, train_y, val_x, val_y,
                                 num_classes=len(classes), epochs=100, batch_size=64,
                                 lr=lr, weight_decay=wd, out_dir="./checkpoints")
        all_results[name] = res

    try:
        with open("mushrooms_100epochs_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("Saved full training history to mushrooms_100epochs_results.json")
    except Exception as e:
        print("Could not save JSON:", e)

    print("\n==========================================================================")
    print("             100-EPOCH BENCHMARK RESULTS (MUSHROOMS DATASET)              ")
    print("==========================================================================")
    print(f"{'Model Architecture':25} {'Best Val Acc':>14} {'Final Val Acc':>14} {'Final Train Acc':>16} {'Total Time':>12}")
    print("-" * 85)
    for name, r in all_results.items():
        print(f"{name:25} {r['best_val_acc']*100:13.2f}% {r['final_val_acc']*100:13.2f}% {r['final_train_acc']*100:15.2f}% {r['total_time_sec']:10.1f}s")
    print("==========================================================================")


if __name__ == "__main__":
    main()
