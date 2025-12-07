"""여러 실험을 일괄 실행하는 스크립트 (보고서 표 작성용)."""
import csv
import time
from pathlib import Path
from typing import List, Dict, Any

from data_preprocessing import build_datasets, CLASS_NAMES
from model_training import train_model

# 보고서 표에 맞춘 기본 실험 설정
EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": "exp1_cnn_base",
        "model": "cnn",
        "augmentations": (),
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "epochs": 12,
        "batch_size": 64,
        "image_size": 96,
        "fine_tune": False,
        "base_trainable_layers": 0,
    },
    {
        "name": "exp2_cnn_drop_flip",
        "model": "cnn",
        "augmentations": ("flip",),
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "epochs": 15,
        "batch_size": 64,
        "image_size": 96,
        "fine_tune": False,
        "base_trainable_layers": 0,
    },
    {
        "name": "exp3_cnn_full_aug",
        "model": "cnn",
        "augmentations": ("flip", "rotation", "brightness"),
        "optimizer": "adam",
        "learning_rate": 7e-4,
        "epochs": 18,
        "batch_size": 64,
        "image_size": 96,
        "fine_tune": False,
        "base_trainable_layers": 0,
    },
    {
        "name": "exp4_resnet_frozen",
        "model": "resnet",
        "augmentations": ("flip", "rotation", "brightness"),
        "optimizer": "adam",
        "learning_rate": 5e-4,
        "epochs": 15,
        "batch_size": 64,
        "image_size": 96,
        "fine_tune": False,
        "base_trainable_layers": 0,
    },
    {
        "name": "exp5_resnet_finetune",
        "model": "resnet",
        "augmentations": ("flip", "rotation", "brightness"),
        "optimizer": "adam",
        "learning_rate": 3e-4,
        "epochs": 12,
        "batch_size": 64,
        "image_size": 96,
        "fine_tune": True,
        "base_trainable_layers": 20,
    },
]


def run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    image_size = (cfg["image_size"], cfg["image_size"])
    train_ds, val_ds, class_names = build_datasets(
        batch_size=cfg["batch_size"],
        image_size=image_size,
        val_split=0.2,
        augmentations=cfg["augmentations"],
    )
    model, history, eval_results, log_dir, checkpoint_dir = train_model(
        model_name=cfg["model"],
        train_ds=train_ds,
        val_ds=val_ds,
        image_size=image_size,
        num_classes=len(class_names),
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        optimizer_name=cfg["optimizer"],
        fine_tune=cfg["fine_tune"],
        base_trainable_layers=cfg["base_trainable_layers"],
    )
    val_loss, val_acc = eval_results
    duration = time.time() - start
    return {
        "name": cfg["name"],
        "model": cfg["model"],
        "augmentations": "|".join(cfg["augmentations"]) if cfg["augmentations"] else "none",
        "optimizer": cfg["optimizer"],
        "learning_rate": cfg["learning_rate"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "time_sec": round(duration, 2),
    }


def main():
    results = []
    for cfg in EXPERIMENTS:
        print(f"Running {cfg['name']} ...")
        summary = run_experiment(cfg)
        print(
            f"{summary['name']} -> val_acc={summary['val_accuracy']:.4f}, "
            f"val_loss={summary['val_loss']:.4f}, time={summary['time_sec']}s"
        )
        results.append(summary)

    out_path = Path("logs") / "experiments.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
