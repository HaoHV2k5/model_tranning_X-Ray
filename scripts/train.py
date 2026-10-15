"""Huấn luyện model.

Ví dụ:
    # chạy thử nhanh để kiểm tra pipeline
    python scripts/train.py --config configs/debug.yaml

    # train thật
    python scripts/train.py --config configs/medicalpatchnet.yaml

    # đổi tham số không cần sửa file
    python scripts/train.py --config configs/medicalpatchnet.yaml --name lr3e-4 --set train.optimizer.lr=3e-4

    # train tiếp một lần chạy bị dừng giữa chừng (Ctrl+C, tắt máy...)
    python scripts/train.py --resume experiments/20261015-091200_medicalpatchnet
"""
import argparse
import time
import traceback
from pathlib import Path

import _common  # noqa: F401  (phải import trước torch)

import torch

from xray.config import load_config, save_config
from xray.data import build_dataset, build_loader
from xray.engine import MaskedBCEWithLogits, Trainer, full_evaluation, load_weights, read_checkpoint
from xray.models import build_model
from xray.tracking import ExperimentLogger, create_run_dir
from xray.tracking import plots
from xray.utils import collect_env_info, count_parameters, get_device, load_json, resolve_path, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Train model X-quang (CheXpert)")
    p.add_argument("--config", type=str, help="File YAML cấu hình, vd configs/medicalpatchnet.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Ghi đè tham số, vd train.epochs=5")
    p.add_argument("--name", type=str, default=None, help="Tên lần chạy (ghi đè experiment.name)")
    p.add_argument("--resume", type=str, default=None, help="Thư mục lần chạy cần train tiếp")
    p.add_argument("--no-test", action="store_true", help="Không đánh giá tập test khi train xong")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = list(args.set)
    if args.name:
        overrides.append(f"experiment.name={args.name}")

    # ------------------------------------------------------------ config + thư mục chạy
    if args.resume:
        run_dir = resolve_path(args.resume)
        if not (run_dir / "checkpoints" / "last.pt").exists():
            raise FileNotFoundError(f"Không thấy {run_dir}/checkpoints/last.pt để train tiếp")
        cfg = load_config(run_dir / "config.yaml", overrides)  # cho phép tăng train.epochs khi resume
    else:
        if not args.config:
            raise SystemExit("Cần --config (hoặc --resume)")
        cfg = load_config(args.config, overrides)
        run_dir = create_run_dir(cfg.experiment.output_dir, cfg.experiment.name)

    logger = ExperimentLogger(run_dir, use_tensorboard=True)
    log = logger.info
    save_config(cfg, run_dir / "config.yaml")

    seed = int(cfg.experiment.seed)
    seed_everything(seed, bool(cfg.experiment.get("deterministic", False)))
    device = get_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    env = collect_env_info(device, seed)
    save_json(env, run_dir / ("env.json" if not args.resume else f"env_resume_{int(time.time())}.json"))

    log("=" * 80)
    log(f"Thư mục lần chạy : {run_dir}")
    log(f"Thiết bị         : {device} | torch {env['torch']} | python {env['python']}")
    if cfg.experiment.get("notes"):
        log(f"Ghi chú          : {cfg.experiment.notes}")

    # ------------------------------------------------------------ dữ liệu
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    class_names = train_ds.class_names
    if int(cfg.model.num_classes) != len(class_names):
        raise ValueError(f"model.num_classes={cfg.model.num_classes} nhưng CSV có {len(class_names)} nhãn")
    train_loader = build_loader(train_ds, cfg, train=True, seed=seed)
    val_loader = build_loader(val_ds, cfg, train=False, seed=seed)
    log(f"Dữ liệu          : train {len(train_ds)} | val {len(val_ds)} ảnh | {len(class_names)} bệnh "
        f"| ảnh {cfg.data.img_size}px | nhãn không chắc: {cfg.data.uncertain_policy}")
    pos_counts = train_ds.positive_counts().int().tolist()
    log("Số ca dương (train): " + ", ".join(f"{n}={c}" for n, c in zip(class_names, pos_counts)))

    # ------------------------------------------------------------ model
    model = build_model(cfg.model)
    n_params = count_parameters(model)
    log(f"Model            : {cfg.model.name} {dict(cfg.model.get('params', {}))} | {n_params['total'] / 1e6:.2f}M tham số")
    log(f"Train            : {cfg.train.epochs} epoch | batch {cfg.train.batch_size} | "
        f"{cfg.train.optimizer.name} lr={cfg.train.optimizer.lr} | scheduler {cfg.train.scheduler.name} "
        f"| chọn best theo {cfg.train.monitor}")
    log("=" * 80)

    pos_weight = train_ds.pos_weight() if cfg.train.loss.get("pos_weight", False) else None
    trainer = Trainer(cfg, model, train_loader, val_loader, device, logger, class_names, pos_weight)
    if args.resume:
        trainer.resume_from(run_dir / "checkpoints" / "last.pt")

    # ------------------------------------------------------------ train
    t_start = time.time()
    status = "failed"
    try:
        status = trainer.fit()
    except KeyboardInterrupt:
        status = "interrupted"
        logger.warning("Đã dừng bằng Ctrl+C. Train tiếp bằng: "
                       f"python scripts/train.py --resume {run_dir.relative_to(resolve_path('.'))}")
    except Exception:
        logger.warning("Lỗi khi train:\n" + traceback.format_exc())
        raise
    finally:
        train_time = time.time() - t_start
        if (run_dir / "history.csv").exists():
            plots.plot_history(run_dir / "history.csv", run_dir / "figures")
            plots.plot_per_class_history(run_dir / "val_per_class.csv", run_dir / "figures")
            log(f"Đã vẽ biểu đồ vào {run_dir / 'figures'}")

        summary_path = run_dir / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else {}
        summary.update({
            "run_name": run_dir.name,
            "experiment": cfg.experiment.name,
            "notes": cfg.experiment.get("notes", ""),
            "status": status,
            "model": cfg.model.name,
            "model_params": dict(cfg.model.get("params", {})),
            "num_parameters": n_params["total"],
            "hparams": {
                "img_size": cfg.data.img_size,
                "uncertain_policy": cfg.data.uncertain_policy,
                "epochs": cfg.train.epochs,
                "batch_size": cfg.train.batch_size,
                "optimizer": cfg.train.optimizer.name,
                "lr": cfg.train.optimizer.lr,
                "scheduler": cfg.train.scheduler.name,
                "pos_weight": cfg.train.loss.get("pos_weight", False),
                "seed": seed,
            },
            "monitor": trainer.monitor,
            "best_epoch": trainer.best_epoch,
            "best_value": trainer.best_value,
            "epochs_completed": trainer.start_epoch - 1 if status == "failed" else None,
            "train_time_min": round(summary.get("train_time_min", 0) + train_time / 60, 2),
            "device": str(device),
        })
        if (run_dir / "history.csv").exists():
            from xray.tracking import read_csv
            hist = read_csv(run_dir / "history.csv")
            summary["epochs_completed"] = int(hist[-1]["epoch"]) if hist else 0
            best_row = next((r for r in hist if int(r["epoch"]) == trainer.best_epoch), None)
            if best_row:
                summary["best_val"] = {k: float(v) for k, v in best_row.items() if k.startswith("val_")}
        save_json(summary, summary_path)

    # ------------------------------------------------------------ đánh giá test với best.pt
    if status in ("completed", "early_stopped") and cfg.train.get("eval_test_at_end", True) and not args.no_test:
        best_path = run_dir / "checkpoints" / "best.pt"
        if best_path.exists():
            log(f"Đánh giá tập TEST với {best_path.name} (epoch {trainer.best_epoch})")
            load_weights(trainer.model, read_checkpoint(best_path))
            test_ds = build_dataset(cfg, "test")
            test_loader = build_loader(test_ds, cfg, train=False, seed=seed)
            metrics = full_evaluation(
                trainer.model, test_loader, val_loader, device, class_names, cfg.eval, run_dir / "test",
                split_name="test", criterion=MaskedBCEWithLogits().to(device), amp=trainer.use_amp,
                seed=seed, log=log,
            )
            summary["test"] = {"mean": metrics["mean"], "ci": metrics["ci"], "loss": metrics["loss"],
                               "threshold_method": metrics["threshold_method"]}
            save_json(summary, summary_path)

    logger.close()
    log(f"XONG. Kết quả ở: {run_dir}")


if __name__ == "__main__":
    main()
