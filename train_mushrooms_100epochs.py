"""High-performance 100-Epoch training using jimm.data and native AMP mixed precision."""
import time
import json
import numpy as np
import jax.numpy as jnp
from flax import nnx

import jimm
from jimm.data import MixupCutmix, create_loader
from jimm.checkpoint import save_checkpoint
from jimm.train import make_optimizer, make_cached_train_step, make_cached_eval_step


def train_single_model(model_name: str, data_dir: str, num_classes: int = 9,
                       epochs: int = 100, batch_size: int = 256, img_size: int = 224,
                       lr: float = 2e-3, weight_decay: float = 0.01, smoothing: float = 0.1,
                       amp: bool = True, out_dir: str = "./checkpoints",
                       auto_augment: str | None = None, mixup_alpha: float = 0.0,
                       cutmix_alpha: float = 0.0):
    print(f"\n=======================================================", flush=True)
    print(f"  Training {model_name} (100 Epochs, AMP={'bfloat16' if amp else 'FP32'}, bs={batch_size}, lr={lr})", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Create Model
    model = jimm.create_model(model_name, num_classes=num_classes, rngs=nnx.Rngs(0))

    # 2. Use dm_pix-backed jimm.data Grain transforms with original-resolution RAM caching.
    train_loader = create_loader(
        f"{data_dir}/train", batch_size=batch_size, img_size=img_size,
        is_training=True, auto_augment=auto_augment,
        num_workers=0, seed=42, in_memory=True,
    )
    val_loader = create_loader(f"{data_dir}/val", batch_size=batch_size, img_size=img_size,
                               is_training=False, num_workers=0, seed=42, in_memory=True)

    mixup = None
    if mixup_alpha > 0 or cutmix_alpha > 0:
        mixup = MixupCutmix(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            label_smoothing=smoothing,
            num_classes=num_classes,
        )

    steps_per_epoch = len(train_loader)
    opt = make_optimizer(model, lr=lr, weight_decay=weight_decay, epochs=epochs,
                         steps_per_epoch=steps_per_epoch, clip_grad=1.0)

    # 3. Create Cached Step Functions with Native AMP (bfloat16) on Tensor Cores
    model.train()
    cached_train_step = make_cached_train_step(model, opt, amp=amp)
    model.eval()
    cached_eval_step = make_cached_eval_step(model, amp=amp)
    model.train()

    best_val_acc = 0.0
    best_epoch = 0
    history = []
    train_loss, train_acc = 0.0, 0.0
    val_loss, val_acc = 0.0, 0.0
    t_start = time.time()

    it = iter(train_loader)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss_sum = jnp.zeros(())
        acc_sum = jnp.zeros(())

        for _ in range(steps_per_epoch):
            batch = next(it)
            images, labels = batch["image"], batch["label"]
            if mixup is not None:
                images, labels = mixup(np.asarray(images), np.asarray(labels))
            images = jnp.asarray(images)
            labels = jnp.asarray(labels)

            l, a = cached_train_step(images, labels, smoothing)
            loss_sum = loss_sum + l
            acc_sum = acc_sum + a

        try:
            train_loss = float(loss_sum) / max(steps_per_epoch, 1)
            train_acc = float(acc_sum) / max(steps_per_epoch, 1)
        except Exception:
            train_loss, train_acc = 0.0, 0.0

        # Validation with jimm.data loader
        v_loss_sum = jnp.zeros(())
        v_acc_sum = jnp.zeros(())
        n_val = 0
        for v_batch in val_loader:
            v_images = jnp.asarray(v_batch["image"])
            v_labels = jnp.asarray(v_batch["label"])
            l, a = cached_eval_step(v_images, v_labels)
            v_loss_sum = v_loss_sum + l * len(v_labels)
            v_acc_sum = v_acc_sum + a * len(v_labels)
            n_val += len(v_labels)

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
    data_dir = "/home/lzc/Documents/jimm/datasets/mushrooms_split"

    models_config = [
        ("resnet18", 2e-3, 0.01),
        ("mobilenetv3_large_100", 2e-3, 0.01),
        ("convnext_tiny", 1e-3, 0.05),
    ]

    all_results = {}
    for name, lr, wd in models_config:
        res = train_single_model(
            name, data_dir, num_classes=9, epochs=100, batch_size=256,
            lr=lr, weight_decay=wd, amp=True, out_dir="./checkpoints",
            auto_augment="rand-m9-n2", mixup_alpha=0.8, cutmix_alpha=1.0,
        )
        all_results[name] = res

    try:
        with open("mushrooms_100epochs_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("Saved full training history to mushrooms_100epochs_results.json")
    except Exception as e:
        print("Could not save JSON:", e)

    print("\n==========================================================================")
    print("      100-EPOCH BENCHMARK RESULTS (jimm.data + AMP bfloat16 on RTX 5090)  ")
    print("==========================================================================")
    print(f"{'Model Architecture':25} {'Best Val Acc':>14} {'Final Val Acc':>14} {'Final Train Acc':>16} {'Total Time':>12}")
    print("-" * 85)
    for name, r in all_results.items():
        print(f"{name:25} {r['best_val_acc']*100:13.2f}% {r['final_val_acc']*100:13.2f}% {r['final_train_acc']*100:15.2f}% {r['total_time_sec']:10.1f}s")
    print("==========================================================================")


if __name__ == "__main__":
    main()
