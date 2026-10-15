"""Tiện ích chung: đường dẫn, seed, thiết bị, thông tin môi trường."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# x-quang/src/xray/utils.py -> x-quang/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(p) -> Path:
    """Đường dẫn tương đối được tính từ thư mục gốc dự án (chạy script ở đâu cũng được)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# ----------------------------------------------------------------------------- seed
def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Seed riêng cho từng worker của DataLoader (augmentation khác nhau nhưng tái lập được)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------------- device
def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Yêu cầu device=mps nhưng máy không hỗ trợ MPS")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Yêu cầu device=cuda nhưng không có GPU NVIDIA")
    return torch.device(name)


def empty_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- env info
def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def collect_env_info(device: torch.device, seed: int) -> Dict[str, Any]:
    import torchvision

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "seed": seed,
        "git_commit": _git_commit(),
    }


def save_json(obj: Any, path) -> None:
    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Không serialize được kiểu {type(o)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)


def load_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
