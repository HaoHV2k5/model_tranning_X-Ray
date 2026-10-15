"""Chạy dự đoán trên một DataLoader và đánh giá đầy đủ (dùng sau khi train xong / trong evaluate.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from ..metrics import bootstrap_ci, compute_multilabel_metrics, youden_threshold
from ..tracking.logger import write_csv
from ..tracking import plots
from ..utils import save_json


@torch.inference_mode()
def run_inference(model, loader, device, criterion=None, amp: bool = False, desc: str = "Infer") -> Dict[str, Any]:
    """Trả về probs/targets/masks dạng numpy [N,C] và loss trung bình (nếu có criterion)."""
    model.eval()
    probs, targets, masks = [], [], []
    loss_sum, weight_sum = 0.0, 0.0
    use_amp = amp and device.type in ("cuda", "mps")
    for img, target, mask in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        img = img.to(device, non_blocking=True, dtype=torch.float32)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(img)
        logits = logits.float()
        if criterion is not None:
            t, m = target.to(device), mask.to(device)
            batch_loss = criterion(logits, t, m)
            w = float(m.sum().item())
            loss_sum += float(batch_loss.item()) * w
            weight_sum += w
        probs.append(torch.sigmoid(logits).cpu())
        targets.append(target)
        masks.append(mask)
    return {
        "probs": torch.cat(probs).numpy(),
        "targets": torch.cat(targets).numpy(),
        "masks": torch.cat(masks).numpy(),
        "loss": loss_sum / weight_sum if weight_sum > 0 else float("nan"),
    }


def compute_thresholds(out: Dict[str, np.ndarray], method: str = "youden", fixed: float = 0.5) -> np.ndarray:
    n_classes = out["probs"].shape[1]
    if method == "fixed":
        return np.full(n_classes, float(fixed))
    if method != "youden":
        raise ValueError("threshold_method phải là 'youden' hoặc 'fixed'")
    thr = []
    for c in range(n_classes):
        valid = out["masks"][:, c] > 0
        thr.append(youden_threshold(out["targets"][valid, c], out["probs"][valid, c], default=fixed))
    return np.asarray(thr)


def full_evaluation(
    model,
    eval_loader,
    threshold_loader,
    device,
    class_names: Sequence[str],
    eval_cfg,
    out_dir: Path,
    split_name: str = "test",
    criterion=None,
    amp: bool = False,
    seed: int = 0,
    log=print,
) -> Dict[str, Any]:
    """Đánh giá trên `eval_loader`, ngưỡng được tối ưu trên `threshold_loader` (tập VAL - tránh rò rỉ test).

    Lưu vào out_dir: metrics.json, per_class.csv, thresholds.json, predictions.csv và các hình.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method = eval_cfg.get("threshold_method", "youden")
    fixed = float(eval_cfg.get("fixed_threshold", 0.5))

    # 1) Ngưỡng từ tập val
    if method == "youden" and threshold_loader is not None:
        log("Tối ưu ngưỡng (Youden J) trên tập validation...")
        thr_out = run_inference(model, threshold_loader, device, amp=amp, desc="Val (threshold)")
        thresholds = compute_thresholds(thr_out, "youden", fixed)
    else:
        thresholds = np.full(len(class_names), fixed)

    # 2) Dự đoán trên tập cần đánh giá
    out = run_inference(model, eval_loader, device, criterion=criterion, amp=amp, desc=f"Eval {split_name}")
    probs, targets, masks = out["probs"], out["targets"], out["masks"]

    per_class, mean = compute_multilabel_metrics(probs, targets, masks, class_names, thresholds)
    _, mean_05 = compute_multilabel_metrics(probs, targets, masks, class_names, [0.5] * len(class_names))

    # 3) Bootstrap CI
    ci = None
    n_boot = int(eval_cfg.get("bootstrap", 0) or 0)
    if n_boot > 0:
        log(f"Bootstrap {n_boot} lần để tính khoảng tin cậy {int(float(eval_cfg.get('ci', 0.95)) * 100)}%...")
        ci = bootstrap_ci(probs, targets, masks, class_names, thresholds, n_boot, float(eval_cfg.get("ci", 0.95)), seed)

    # 4) Lưu kết quả
    metrics = {
        "split": split_name,
        "n_samples": int(probs.shape[0]),
        "loss": out["loss"],
        "threshold_method": "youden (từ tập val)" if (method == "youden" and threshold_loader is not None) else f"fixed {fixed}",
        "mean": mean,
        "mean_at_0.5": mean_05,
        "ci": ci["mean"] if ci else None,
        "ci_level": float(eval_cfg.get("ci", 0.95)) if ci else None,
        "n_bootstrap": n_boot,
    }
    save_json(metrics, out_dir / "metrics.json")
    save_json({n: float(t) for n, t in zip(class_names, thresholds)}, out_dir / "thresholds.json")
    if ci:
        save_json(ci, out_dir / "bootstrap_ci.json")

    rows = []
    for r in per_class:
        row = dict(r)
        if ci:
            row["auroc_ci_low"] = ci["per_class"][r["class"]]["auroc"]["low"]
            row["auroc_ci_high"] = ci["per_class"][r["class"]]["auroc"]["high"]
        rows.append(row)
    write_csv(out_dir / "per_class.csv", rows)

    ds = eval_loader.dataset
    pred_rows = []
    for i in range(probs.shape[0]):
        row: Dict[str, Any] = {"path": ds.rel_paths[i]} if hasattr(ds, "rel_paths") else {"index": i}
        for c, name in enumerate(class_names):
            row[f"{name}__true"] = int(targets[i, c]) if masks[i, c] > 0 else ""
            row[f"{name}__prob"] = float(probs[i, c])
        pred_rows.append(row)
    write_csv(out_dir / "predictions.csv", pred_rows)

    plots.plot_roc_curves(probs, targets, masks, class_names, per_class, out_dir / "roc_curves.png")
    plots.plot_confusion_matrices(probs, targets, masks, class_names, thresholds, out_dir / "confusion_matrices.png")
    plots.plot_auroc_with_ci(per_class, ci, out_dir / "per_class_auroc.png", title=f"{split_name.upper()} AUROC theo từng bệnh")

    # 5) In bảng tóm tắt
    log(f"===== KẾT QUẢ TẬP {split_name.upper()} ({probs.shape[0]} ảnh) =====")
    log(f"{'Bệnh':28s} {'n+':>4s} {'AUROC':>7s} {'AUPRC':>7s} {'Sens':>6s} {'Spec':>6s} {'F1':>6s}")
    for r in per_class:
        log(f"{r['class']:28s} {r['n_pos']:4d} {r['auroc']:7.3f} {r['auprc']:7.3f} "
            f"{r['sensitivity']:6.3f} {r['specificity']:6.3f} {r['f1']:6.3f}")
    ci_txt = ""
    if ci:
        lo, hi = ci["mean"]["auroc_mean"]["low"], ci["mean"]["auroc_mean"]["high"]
        ci_txt = f" (CI95% {lo:.4f}-{hi:.4f})"
    log(f"{'TRUNG BÌNH':28s}      {mean['auroc_mean']:7.4f}{ci_txt}  F1={mean['f1_mean']:.4f}  "
        f"Acc={mean['accuracy_mean']:.4f}")
    return metrics
