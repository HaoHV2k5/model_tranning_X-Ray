"""Loss, optimizer, scheduler."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogits(nn.Module):
    """BCEWithLogits bỏ qua các nhãn có mask=0 (dùng khi uncertain_policy=ignore)."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), reduction="none", pos_weight=self.pos_weight
        )
        if mask is None:
            return loss.mean()
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def build_optimizer(opt_cfg, model: nn.Module) -> torch.optim.Optimizer:
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.lr)
    wd = float(opt_cfg.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=float(opt_cfg.get("momentum", 0.9)), nesterov=True)
    raise ValueError(f"Optimizer '{name}' chưa hỗ trợ (adamw | adam | sgd)")


def build_scheduler(sched_cfg, optimizer, epochs: int, steps_per_epoch: int):
    """Scheduler được step SAU MỖI BATCH (giống code gốc dùng OneCycleLR)."""
    name = str(sched_cfg.get("name", "none")).lower()
    total = max(1, epochs * steps_per_epoch)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in optimizer.param_groups],
            total_steps=total,
            pct_start=float(sched_cfg.get("pct_start", 0.05)),
        )
    if name == "cosine":
        warmup = int(float(sched_cfg.get("warmup_epochs", 0)) * steps_per_epoch)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name in ("none", "constant"):
        return None
    raise ValueError(f"Scheduler '{name}' chưa hỗ trợ (onecycle | cosine | none)")
