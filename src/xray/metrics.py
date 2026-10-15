"""Metric cho bài toán phân loại ĐA NHÃN (mỗi ảnh có thể có nhiều bệnh).

Tất cả được tính riêng cho từng bệnh (binary), sau đó lấy trung bình (macro, bỏ qua NaN).
Viết bằng numpy thuần để nhanh khi bootstrap và không phụ thuộc thêm thư viện.

- AUROC   : khả năng xếp hạng ca bệnh cao hơn ca không bệnh (0.5 = đoán mò, 1.0 = hoàn hảo).
            Đây là metric CHÍNH của CheXpert, không phụ thuộc ngưỡng.
- AUPRC   : (Average Precision) phù hợp hơn khi bệnh hiếm.
- Accuracy, Precision, Sensitivity (Recall), Specificity, F1: cần chọn ngưỡng xác suất.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

THRESHOLD_METRICS = ("accuracy", "precision", "sensitivity", "specificity", "f1")
RANKING_METRICS = ("auroc", "auprc")
ALL_METRICS = RANKING_METRICS + THRESHOLD_METRICS


# ----------------------------------------------------------------------------- binary metrics
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC qua thống kê Mann-Whitney (xử lý điểm bằng nhau bằng hạng trung bình)."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    _, inverse, counts = np.unique(s_sorted, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg_rank = (ends - counts + 1 + ends) / 2.0
    ranks = avg_rank[inverse]
    pos_rank_sum = ranks[y[order] == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def roc_curve(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (fpr, tpr, thresholds). Dự đoán dương khi xác suất >= threshold."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, len(y) - 1]
    tps = np.cumsum(y)[idx]
    fps = (idx + 1) - tps
    tpr = np.r_[0.0, tps / max(n_pos, 1)]
    fpr = np.r_[0.0, fps / max(n_neg, 1)]
    thresholds = np.r_[np.inf, s[idx]]
    return fpr, tpr, thresholds


def youden_threshold(y: np.ndarray, s: np.ndarray, default: float = 0.5) -> float:
    """Ngưỡng tối đa hoá Youden J = Sensitivity + Specificity - 1."""
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return default
    fpr, tpr, thr = roc_curve(y, s)
    best = int(np.argmax(tpr - fpr))
    t = thr[best]
    return float(min(t, 1.0)) if np.isfinite(t) else 1.0


def confusion_counts(y: np.ndarray, s: np.ndarray, threshold: float) -> Dict[str, int]:
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(s) >= threshold
    return {
        "tp": int(np.sum(pred & y)),
        "fp": int(np.sum(pred & ~y)),
        "tn": int(np.sum(~pred & ~y)),
        "fn": int(np.sum(~pred & y)),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def threshold_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> Dict[str, float]:
    c = confusion_counts(y, s, threshold)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = _safe_div(tp, tp + fp)
    sensitivity = _safe_div(tp, tp + fn)
    return {
        "accuracy": _safe_div(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
    }


# ----------------------------------------------------------------------------- multi-label
def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")


def compute_multilabel_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    masks: Optional[np.ndarray],
    class_names: Sequence[str],
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Trả về (danh sách metric từng bệnh, metric trung bình macro)."""
    n_classes = probs.shape[1]
    if masks is None:
        masks = np.ones_like(targets)
    if thresholds is None:
        thresholds = [0.5] * n_classes

    per_class: List[Dict[str, float]] = []
    for c in range(n_classes):
        valid = masks[:, c] > 0
        y = targets[valid, c].astype(np.int64)
        s = probs[valid, c]
        row: Dict[str, float] = {
            "class": class_names[c],
            "n": int(valid.sum()),
            "n_pos": int(y.sum()),
            "threshold": float(thresholds[c]),
            "auroc": auroc(y, s),
            "auprc": average_precision(y, s),
        }
        row.update(threshold_metrics(y, s, thresholds[c]))
        per_class.append(row)

    mean = {f"{m}_mean": _nanmean([r[m] for r in per_class]) for m in ALL_METRICS}
    return per_class, mean


def bootstrap_ci(
    probs: np.ndarray,
    targets: np.ndarray,
    masks: Optional[np.ndarray],
    class_names: Sequence[str],
    thresholds: Sequence[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Khoảng tin cậy bằng bootstrap (lấy mẫu lại có hoàn lại toàn bộ ảnh).

    Trả về {"mean": {metric: {low, high, std}}, "per_class": {tên_bệnh: {metric: {...}}}}.
    """
    rng = np.random.default_rng(seed)
    n = probs.shape[0]
    if masks is None:
        masks = np.ones_like(targets)
    mean_samples = {f"{m}_mean": [] for m in ALL_METRICS}
    class_samples = {name: {m: [] for m in ALL_METRICS} for name in class_names}

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        per_class, mean = compute_multilabel_metrics(probs[idx], targets[idx], masks[idx], class_names, thresholds)
        for k, v in mean.items():
            mean_samples[k].append(v)
        for row in per_class:
            for m in ALL_METRICS:
                class_samples[row["class"]][m].append(row[m])

    alpha = (1.0 - ci) / 2.0

    def summarize(values):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return {"low": float("nan"), "high": float("nan"), "std": float("nan")}
        return {
            "low": float(np.quantile(arr, alpha)),
            "high": float(np.quantile(arr, 1 - alpha)),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        }

    return {
        "n_boot": int(n_boot),
        "ci": ci,
        "mean": {k: summarize(v) for k, v in mean_samples.items()},
        "per_class": {c: {m: summarize(v) for m, v in d.items()} for c, d in class_samples.items()},
    }
