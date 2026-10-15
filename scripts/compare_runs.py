"""So sánh tất cả các lần chạy trong thư mục experiments/.

Ví dụ:
    python scripts/compare_runs.py

    # Lọc theo tên
    python scripts/compare_runs.py --filter medicalpatchnet

    # Sắp xếp theo F1 thay vì AUROC
    python scripts/compare_runs.py --sort val_f1_mean
"""
import argparse
from pathlib import Path

import _common  # noqa: F401

from xray.utils import load_json, resolve_path
from xray.tracking.logger import read_csv


def parse_args():
    p = argparse.ArgumentParser(description="So sánh các lần chạy")
    p.add_argument("--exp-dir", type=str, default="experiments", help="Thư mục chứa các lần chạy")
    p.add_argument("--filter", type=str, default=None, help="Lọc theo tên (bao gồm chuỗi con)")
    p.add_argument("--sort", type=str, default="best_value", help="Cột dùng để sắp xếp")
    p.add_argument("--out", type=str, default=None, help="Lưu bảng ra file CSV")
    return p.parse_args()


def collect_runs(exp_dir: Path, name_filter=None):
    runs = []
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if name_filter and name_filter.lower() not in run_dir.name.lower():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        s = load_json(summary_path)

        # Lấy best epoch từ history.csv (đáng tin hơn summary)
        hist = []
        if (run_dir / "history.csv").exists():
            hist = read_csv(run_dir / "history.csv")
        best_hist = next((r for r in hist if int(r.get("epoch", -1)) == s.get("best_epoch", -1)), None)

        row = {
            "run": run_dir.name,
            "experiment": s.get("experiment", ""),
            "model": s.get("model", ""),
            "status": s.get("status", ""),
            "epochs_done": s.get("epochs_completed", "?"),
            "best_epoch": s.get("best_epoch", "?"),
            "best_value": round(float(s["best_value"]), 4) if s.get("best_value") else "",
            "monitor": s.get("monitor", ""),
            "train_time_min": s.get("train_time_min", ""),
            "lr": s.get("hparams", {}).get("lr", ""),
            "batch_size": s.get("hparams", {}).get("batch_size", ""),
            "img_size": s.get("hparams", {}).get("img_size", ""),
            "seed": s.get("hparams", {}).get("seed", ""),
            "uncertain": s.get("hparams", {}).get("uncertain_policy", ""),
        }
        if best_hist:
            for k in ["val_auroc_mean", "val_f1_mean", "val_accuracy_mean", "val_sensitivity_mean",
                      "val_specificity_mean", "train_loss", "val_loss"]:
                try:
                    row[k] = round(float(best_hist[k]), 4) if best_hist.get(k) else ""
                except (ValueError, TypeError):
                    pass
        if s.get("test"):
            tm = s["test"].get("mean", {})
            row["test_auroc"] = round(float(tm.get("auroc_mean", 0)), 4) if tm.get("auroc_mean") else ""
            row["test_f1"] = round(float(tm.get("f1_mean", 0)), 4) if tm.get("f1_mean") else ""
            ci = s["test"].get("ci", {})
            if ci and "auroc_mean" in ci:
                row["test_auroc_ci_low"] = round(float(ci["auroc_mean"]["low"]), 4)
                row["test_auroc_ci_high"] = round(float(ci["auroc_mean"]["high"]), 4)
        row["notes"] = s.get("notes", "")
        runs.append(row)
    return runs


def main():
    args = parse_args()
    exp_dir = resolve_path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Chưa có lần chạy nào trong {exp_dir}")
        return
    runs = collect_runs(exp_dir, args.filter)
    if not runs:
        print("Không tìm thấy lần chạy nào (thiếu summary.json).")
        return

    sort_key = args.sort
    runs.sort(key=lambda r: float(r.get(sort_key) or -1e9), reverse=True)

    # In bảng ra terminal
    cols = ["run", "model", "status", "best_epoch", "val_auroc_mean", "val_f1_mean", "test_auroc", "test_f1",
            "epochs_done", "lr", "train_time_min", "notes"]
    col_w = {c: max(len(c), max((len(str(r.get(c, ""))) for r in runs), default=0)) for c in cols}
    header = " | ".join(c.ljust(col_w[c]) for c in cols)
    sep = "-+-".join("-" * col_w[c] for c in cols)
    print(f"\n{'BẢNG XẾP HẠNG CÁC LẦN CHẠY':^{len(header)}}")
    print(f"(sắp xếp theo {sort_key}, thư mục {exp_dir})")
    print(sep)
    print(header)
    print(sep)
    for r in runs:
        print(" | ".join(str(r.get(c, "")).ljust(col_w[c]) for c in cols))
    print(sep)
    print(f"Tổng: {len(runs)} lần chạy")

    if args.out:
        from xray.tracking.logger import write_csv
        write_csv(Path(args.out), runs, list(runs[0].keys()) if runs else [])
        print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
