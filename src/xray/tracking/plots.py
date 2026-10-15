"""Vẽ biểu đồ cho báo cáo (PNG, dpi cao)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # không cần màn hình
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..metrics import confusion_counts, roc_curve  # noqa: E402
from .logger import read_csv  # noqa: E402

DPI = 150
plt.rcParams.update({"figure.autolayout": False, "axes.grid": True, "grid.alpha": 0.3})


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _col(rows: List[dict], key: str) -> np.ndarray:
    return np.array([_to_float(r.get(key)) for r in rows])


# ----------------------------------------------------------------------------- training curves
def plot_history(history_csv: Path, out_dir: Path, monitor: Optional[str] = None) -> List[Path]:
    rows = read_csv(history_csv)
    if not rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = _col(rows, "epoch")
    saved = []

    # 1) Loss
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, _col(rows, "train_loss"), "o-", label="Train loss")
    ax.plot(epochs, _col(rows, "val_loss"), "s-", label="Val loss")
    best = [i for i, r in enumerate(rows) if str(r.get("is_best")) in ("1", "True")]
    if best:
        b = best[-1]
        ax.axvline(epochs[b], color="gray", ls="--", lw=1, label=f"Best epoch ({int(epochs[b])})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Training / Validation loss")
    ax.legend()
    p = out_dir / "loss_curve.png"
    fig.tight_layout()
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    saved.append(p)

    # 2) Train vs Val cho AUROC và Accuracy (khoảng cách lớn = overfitting)
    for key, label, fname in [("auroc_mean", "AUROC", "auroc_curve.png"), ("accuracy_mean", "Accuracy@0.5", "accuracy_curve.png")]:
        if f"val_{key}" not in rows[0]:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if f"train_{key}" in rows[0]:
            ax.plot(epochs, _col(rows, f"train_{key}"), "o-", label=f"Train {label}")
        ax.plot(epochs, _col(rows, f"val_{key}"), "s-", label=f"Val {label}")
        if best:
            ax.axvline(epochs[best[-1]], color="gray", ls="--", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{label} (macro mean)")
        ax.set_title(f"Training / Validation {label}")
        ax.legend()
        p = out_dir / fname
        fig.tight_layout()
        fig.savefig(p, dpi=DPI)
        plt.close(fig)
        saved.append(p)

    # 3) Tất cả metric trên tập validation
    metric_keys = [
        ("val_auroc_mean", "AUROC"),
        ("val_auprc_mean", "AUPRC"),
        ("val_accuracy_mean", "Accuracy@0.5"),
        ("val_f1_mean", "F1@0.5"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in metric_keys:
        if key in rows[0]:
            ax.plot(epochs, _col(rows, key), "o-", label=label)
    if best:
        ax.axvline(epochs[best[-1]], color="gray", ls="--", lw=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (macro mean, 14 bệnh)")
    ax.set_title("Validation metrics")
    ax.set_ylim(0, 1)
    ax.legend()
    p = out_dir / "metric_curves.png"
    fig.tight_layout()
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    saved.append(p)

    # 3) Learning rate
    if "lr" in rows[0]:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(epochs, _col(rows, "lr"), "o-")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate (cuối epoch)")
        ax.set_yscale("log")
        ax.set_title("Learning rate schedule")
        p = out_dir / "lr_curve.png"
        fig.tight_layout()
        fig.savefig(p, dpi=DPI)
        plt.close(fig)
        saved.append(p)
    return saved


def plot_per_class_history(per_class_csv: Path, out_dir: Path, metric: str = "auroc") -> Optional[Path]:
    """Heatmap: hàng = bệnh, cột = epoch, màu = AUROC trên tập val."""
    rows = read_csv(per_class_csv)
    if not rows:
        return None
    epochs = sorted({int(r["epoch"]) for r in rows})
    classes = list(dict.fromkeys(r["class"] for r in rows))
    grid = np.full((len(classes), len(epochs)), np.nan)
    for r in rows:
        grid[classes.index(r["class"]), epochs.index(int(r["epoch"]))] = _to_float(r.get(metric))

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(epochs) + 4), 0.42 * len(classes) + 1.5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.grid(False)
    if len(epochs) <= 25:
        for i in range(len(classes)):
            for j in range(len(epochs)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
    ax.set_xlabel("Epoch")
    ax.set_title(f"Validation {metric.upper()} theo từng bệnh")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    p = out_dir / f"per_class_{metric}.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------- evaluation plots
def _grid(n: int):
    cols = 4 if n > 4 else n
    return int(math.ceil(n / cols)), cols


def plot_roc_curves(probs, targets, masks, class_names: Sequence[str], per_class: List[Dict], out_path: Path) -> Path:
    n = len(class_names)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)
    for c, name in enumerate(class_names):
        ax = axes[c // cols][c % cols]
        valid = masks[:, c] > 0
        y, s = targets[valid, c], probs[valid, c]
        if 0 < y.sum() < len(y):
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, lw=1.8)
            thr = per_class[c]["threshold"]
            sens, spec = per_class[c]["sensitivity"], per_class[c]["specificity"]
            ax.plot(1 - spec, sens, "ro", ms=4, label=f"thr={thr:.2f}")
            ax.legend(loc="lower right", fontsize=7)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{name}\nAUROC={per_class[c]['auroc']:.3f}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.supxlabel("False positive rate (1 - Specificity)")
    fig.supylabel("True positive rate (Sensitivity)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_confusion_matrices(probs, targets, masks, class_names, thresholds, out_path: Path) -> Path:
    n = len(class_names)
    rows, cols = _grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.8 * rows), squeeze=False)
    for c, name in enumerate(class_names):
        ax = axes[c // cols][c % cols]
        valid = masks[:, c] > 0
        cc = confusion_counts(targets[valid, c], probs[valid, c], thresholds[c])
        mat = np.array([[cc["tn"], cc["fp"]], [cc["fn"], cc["tp"]]])
        ax.imshow(mat, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred -", "Pred +"], fontsize=7)
        ax.set_yticklabels(["True -", "True +"], fontsize=7)
        ax.set_title(f"{name} (thr={thresholds[c]:.2f})", fontsize=8)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def plot_auroc_with_ci(per_class: List[Dict], ci: Optional[Dict], out_path: Path, title: str = "Test AUROC") -> Path:
    names = [r["class"] for r in per_class]
    vals = np.array([r["auroc"] for r in per_class])
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(names) + 1.5))
    y = np.arange(len(names))
    xerr = None
    if ci:
        lows = np.array([ci["per_class"][names[i]]["auroc"]["low"] for i in order])
        highs = np.array([ci["per_class"][names[i]]["auroc"]["high"] for i in order])
        xerr = np.vstack([vals[order] - lows, highs - vals[order]])
    ax.barh(y, vals[order], xerr=xerr, color="#4C72B0", alpha=0.85, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUROC" + (" (thanh lỗi = CI 95%)" if ci else ""))
    ax.set_title(title)
    for yi, v in zip(y, vals[order]):
        ax.text(min(v + 0.01, 0.96), yi, f"{v:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path
