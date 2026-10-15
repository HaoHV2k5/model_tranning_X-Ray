"""Đánh giá một checkpoint trên bất kỳ tập dữ liệu nào.

Ví dụ:
    # Đánh giá best.pt trên tập test (dùng ngưỡng Youden từ tập val)
    python scripts/evaluate.py \\
        --checkpoint experiments/20261015-091200_medicalpatchnet/checkpoints/best.pt

    # Đánh giá bằng ngưỡng cố định 0.5 thay vì Youden
    python scripts/evaluate.py \\
        --checkpoint experiments/.../checkpoints/best.pt \\
        --split test --threshold fixed
"""
import argparse
from pathlib import Path

import _common  # noqa: F401

from xray.config import load_config
from xray.data import build_dataset, build_loader
from xray.engine import full_evaluation, load_model_for_inference
from xray.tracking.logger import setup_logging
from xray.utils import collect_env_info, get_device, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Đánh giá checkpoint trên tập test / val")
    p.add_argument("--checkpoint", required=True, type=str, help="Đường dẫn tới file .pt")
    p.add_argument("--config", type=str, default=None, help="Dùng config riêng (nếu checkpoint không chứa config)")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Tập cần đánh giá")
    p.add_argument("--threshold", type=str, default="youden", choices=["youden", "fixed"])
    p.add_argument("--fixed-thr", type=float, default=0.5)
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--out", type=str, default=None, help="Thư mục lưu kết quả (mặc định: <run_dir>/<split>_eval/)")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging()
    ckpt_path = Path(args.checkpoint).resolve()
    cfg_path = Path(args.config) if args.config else None
    model, cfg, ckpt = load_model_for_inference(ckpt_path, config_path=cfg_path)

    seed = int(cfg.experiment.get("seed", 42))
    seed_everything(seed)
    device = get_device(cfg.get("device", "auto"))
    model = model.to(device)

    env = collect_env_info(device, seed)
    log.info(f"Thiết bị: {device} | Checkpoint: {ckpt_path}")
    log.info(f"Đánh giá tập '{args.split}' | ngưỡng: {args.threshold}")

    eval_ds = build_dataset(cfg, args.split)
    eval_loader = build_loader(eval_ds, cfg, train=False, seed=seed)
    val_loader = None
    if args.threshold == "youden" and args.split != "val":
        val_ds = build_dataset(cfg, "val")
        val_loader = build_loader(val_ds, cfg, train=False, seed=seed)

    # Thư mục lưu kết quả
    if args.out:
        out_dir = Path(args.out)
    else:
        run_dir = ckpt_path.parent.parent  # checkpoints/ -> run_dir/
        out_dir = run_dir / f"{args.split}_eval"

    eval_override = cfg.eval.to_dict() if hasattr(cfg.eval, "to_dict") else dict(cfg.eval)
    eval_override["threshold_method"] = args.threshold
    eval_override["fixed_threshold"] = args.fixed_thr
    if args.no_bootstrap:
        eval_override["bootstrap"] = 0

    from xray.config import Config
    full_evaluation(
        model, eval_loader, val_loader, device,
        eval_ds.class_names, Config(eval_override), out_dir,
        split_name=args.split, amp=False, seed=seed, log=log.info,
    )
    log.info(f"Kết quả lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
