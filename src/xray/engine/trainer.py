"""Vòng lặp huấn luyện: train -> validate -> ghi log -> lưu checkpoint -> early stopping."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from ..metrics import compute_multilabel_metrics
from ..tracking.logger import ExperimentLogger
from ..utils import empty_device_cache
from .builders import MaskedBCEWithLogits, build_optimizer, build_scheduler
from .checkpoint import read_checkpoint, save_checkpoint
from .evaluator import run_inference

HISTORY_FIELDS = [
    "epoch", "lr", "train_loss", "train_auroc_mean", "train_accuracy_mean",
    "val_loss", "val_auroc_mean", "val_auprc_mean", "val_accuracy_mean", "val_precision_mean",
    "val_sensitivity_mean", "val_specificity_mean", "val_f1_mean",
    "epoch_time_s", "is_best",
]
PER_CLASS_FIELDS = ["epoch", "class", "n_pos", "auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"]


class Trainer:
    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        logger: ExperimentLogger,
        class_names: List[str],
        pos_weight: Optional[torch.Tensor] = None,
    ):
        self.cfg = cfg
        self.tcfg = cfg.train
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.class_names = class_names
        self.run_dir: Path = logger.run_dir
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = MaskedBCEWithLogits(pos_weight.to(device) if pos_weight is not None else None)
        self.optimizer = build_optimizer(self.tcfg.optimizer, self.model)
        self.epochs = int(self.tcfg.epochs)
        self.scheduler = build_scheduler(self.tcfg.scheduler, self.optimizer, self.epochs, len(train_loader))

        self.use_amp = bool(self.tcfg.get("amp", False)) and device.type in ("cuda", "mps")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp and device.type == "cuda" else None
        self.grad_clip = self.tcfg.get("grad_clip")

        self.monitor = str(self.tcfg.get("monitor", "val_auroc_mean"))
        self.mode = str(self.tcfg.get("monitor_mode", "max"))
        es = self.tcfg.get("early_stopping", {}) or {}
        self.es_enabled = bool(es.get("enabled", False))
        self.patience = int(es.get("patience", 5))
        self.min_delta = float(es.get("min_delta", 0.0))
        self.log_every = max(1, int(self.tcfg.get("log_every_n_steps", 10)))

        # trạng thái (được lưu trong checkpoint để train tiếp)
        self.start_epoch = 1
        self.global_step = 0
        self.best_value: Optional[float] = None
        self.best_epoch = 0
        self.bad_epochs = 0

    # ------------------------------------------------------------------ checkpoint
    def _state(self, epoch: int) -> Dict[str, Any]:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "monitor": self.monitor,
            "class_names": self.class_names,
            "config": self.cfg.to_dict(),
        }

    def resume_from(self, path: Path) -> None:
        ckpt = read_checkpoint(path)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler and ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.scaler and ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_value = ckpt.get("best_value")
        self.best_epoch = int(ckpt.get("best_epoch", 0))
        self.bad_epochs = int(ckpt.get("bad_epochs", 0))
        self.logger.info(
            f"Tiếp tục train từ {path} (epoch {self.start_epoch}/{self.epochs}, "
            f"best {self.monitor}={self.best_value} @ epoch {self.best_epoch})"
        )

    # ------------------------------------------------------------------ helpers
    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _is_improvement(self, value: float) -> bool:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    # ------------------------------------------------------------------ train / val
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_sum, n_batches = 0.0, 0
        probs, targets, masks = [], [], []
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [train]", dynamic_ncols=True, leave=False)
        for img, target, mask in pbar:
            img = img.to(self.device, non_blocking=True, dtype=torch.float32)
            target = target.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(img)
            loss = self.criterion(logits, target, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Loss = {loss.item()} tại epoch {epoch}, step {self.global_step}. "
                    "Thử giảm learning rate hoặc tắt amp."
                )

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            loss_val = float(loss.item())
            loss_sum += loss_val
            n_batches += 1
            probs.append(torch.sigmoid(logits.detach().float()).cpu())
            targets.append(target.cpu())
            masks.append(mask.cpu())
            pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{loss_sum / n_batches:.4f}", lr=f"{self._current_lr():.2e}")
            if self.global_step % self.log_every == 0:
                self.logger.scalars({"step/train_loss": loss_val, "step/lr": self._current_lr()}, self.global_step)

        _, mean = compute_multilabel_metrics(
            torch.cat(probs).numpy(), torch.cat(targets).numpy(), torch.cat(masks).numpy(), self.class_names
        )
        return {
            "train_loss": loss_sum / max(1, n_batches),
            "train_auroc_mean": mean["auroc_mean"],
            "train_accuracy_mean": mean["accuracy_mean"],
        }

    def validate(self, epoch: int):
        out = run_inference(
            self.model, self.val_loader, self.device, criterion=self.criterion, amp=self.use_amp,
            desc=f"Epoch {epoch}/{self.epochs} [val]",
        )
        per_class, mean = compute_multilabel_metrics(out["probs"], out["targets"], out["masks"], self.class_names)
        metrics = {"val_loss": out["loss"]}
        metrics.update({f"val_{k}": v for k, v in mean.items()})
        return metrics, per_class

    # ------------------------------------------------------------------ main loop
    def fit(self) -> str:
        """Trả về trạng thái: completed | early_stopped."""
        log = self.logger
        status = "completed"
        if self.start_epoch > self.epochs:
            log.info("Đã train đủ số epoch, không còn gì để chạy.")
            return status

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            train_metrics = self.train_one_epoch(epoch)
            empty_device_cache(self.device)
            val_metrics, per_class = self.validate(epoch)
            empty_device_cache(self.device)
            epoch_time = time.time() - t0

            value = val_metrics.get(self.monitor)
            improved = self._is_improvement(value)
            if improved:
                self.best_value, self.best_epoch, self.bad_epochs = float(value), epoch, 0
            else:
                self.bad_epochs += 1

            row = {"epoch": epoch, "lr": self._current_lr(), **train_metrics, **val_metrics,
                   "epoch_time_s": round(epoch_time, 1), "is_best": int(improved)}
            log.append_csv("history.csv", row, HISTORY_FIELDS)
            for r in per_class:
                log.append_csv("val_per_class.csv", {"epoch": epoch, **r}, PER_CLASS_FIELDS)

            tb = {f"epoch/{k}": v for k, v in row.items() if k not in ("epoch", "is_best")}
            tb.update({f"val_auroc_per_class/{r['class']}": r["auroc"] for r in per_class})
            log.scalars(tb, epoch)

            save_checkpoint(self.ckpt_dir / "last.pt", **self._state(epoch))
            if improved:
                save_checkpoint(self.ckpt_dir / "best.pt", **self._state(epoch))

            log.info(
                f"Epoch {epoch:3d}/{self.epochs} | train_loss {train_metrics['train_loss']:.4f} "
                f"| val_loss {val_metrics['val_loss']:.4f} | val_AUROC {val_metrics['val_auroc_mean']:.4f} "
                f"| val_Acc {val_metrics['val_accuracy_mean']:.4f} | val_F1 {val_metrics['val_f1_mean']:.4f} "
                f"| lr {self._current_lr():.2e} | {epoch_time:.0f}s" + ("  ★ best" if improved else "")
            )

            if self.es_enabled and self.bad_epochs >= self.patience:
                log.info(f"Early stopping: {self.monitor} không cải thiện sau {self.patience} epoch.")
                status = "early_stopped"
                break
        return status
