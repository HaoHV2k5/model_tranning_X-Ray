"""Quản lý thư mục thí nghiệm và ghi log (file log, CSV, JSON, TensorBoard)."""
from __future__ import annotations

import csv
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..utils import resolve_path, save_json


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-") or "run"


def create_run_dir(output_dir, name: str) -> Path:
    """experiments/20261015-091200_medicalpatchnet/ - không bao giờ ghi đè lần chạy cũ."""
    base = resolve_path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{stamp}_{_slug(name)}"
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{stamp}_{_slug(name)}_{suffix}"
        suffix += 1
    for sub in ("checkpoints", "figures"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(log_file: Optional[Path] = None, name: str = "xray") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class ExperimentLogger:
    """Một chỗ duy nhất để ghi mọi thứ của 1 lần chạy."""

    def __init__(self, run_dir: Path, use_tensorboard: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = setup_logging(self.run_dir / "train.log")
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except Exception as exc:  # tensorboard chưa cài -> vẫn chạy bình thường
                self.log.warning(f"Không bật được TensorBoard ({exc}). Chỉ ghi CSV/JSON.")

    # ------------------------------------------------------------------ text
    def info(self, msg: str) -> None:
        self.log.info(msg)

    def warning(self, msg: str) -> None:
        self.log.warning(msg)

    # ------------------------------------------------------------------ tensorboard
    def scalars(self, values: Dict[str, float], step: int) -> None:
        if self.tb is None:
            return
        for key, value in values.items():
            if value is None:
                continue
            try:
                self.tb.add_scalar(key, float(value), step)
            except (TypeError, ValueError):
                continue

    def text(self, tag: str, content: str) -> None:
        if self.tb is not None:
            self.tb.add_text(tag, content)

    # ------------------------------------------------------------------ files
    def append_csv(self, filename: str, row: Dict[str, Any], fieldnames: Optional[Iterable[str]] = None) -> None:
        path = self.run_dir / filename
        fieldnames = list(fieldnames or row.keys())
        new_file = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow({k: _fmt(row.get(k)) for k in fieldnames})

    def save_json(self, filename: str, obj: Any) -> None:
        save_json(obj, self.run_dir / filename)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        for h in list(self.log.handlers):
            h.flush()


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def read_csv(path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows: list, fieldnames: Optional[Iterable[str]] = None) -> None:
    if not rows:
        return
    fieldnames = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r.get(k)) for k in fieldnames})
