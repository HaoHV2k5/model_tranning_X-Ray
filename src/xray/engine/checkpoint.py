"""Lưu / nạp checkpoint. Hỗ trợ cả checkpoint mới (dict đầy đủ) lẫn file trọng số cũ của MedicalPatchNet."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config, load_config
from ..models import build_model


def save_checkpoint(path: Path, **state: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # ghi nguyên tử: không bị hỏng file nếu tắt máy giữa chừng


def _clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # bỏ tiền tố do torch.compile ("_orig_mod.") / DataParallel ("module.") thêm vào
    cleaned = {}
    for key, value in sd.items():
        key = key.replace("_orig_mod.", "")
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def read_checkpoint(path) -> Dict[str, Any]:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):  # file lưu nguyên model (MedicalPatchNet.pt cũ)
        return {"model_state": obj.state_dict()}
    if isinstance(obj, dict) and "model_state" in obj:
        return obj
    if isinstance(obj, dict):  # chỉ có state_dict (MedicalPatchNet_weights.pt cũ)
        return {"model_state": obj}
    raise ValueError(f"Không nhận dạng được định dạng checkpoint: {path}")


def load_weights(model: nn.Module, ckpt: Dict[str, Any], strict: bool = True) -> nn.Module:
    model.load_state_dict(_clean_state_dict(ckpt["model_state"]), strict=strict)
    return model


def load_model_for_inference(
    checkpoint: Path, config: Optional[Config] = None, config_path: Optional[Path] = None
) -> Tuple[nn.Module, Config, Dict[str, Any]]:
    """Ưu tiên config lưu trong checkpoint -> config_path -> config truyền vào."""
    ckpt = read_checkpoint(checkpoint)
    if "config" in ckpt and ckpt["config"]:
        cfg = Config(ckpt["config"])
    elif config_path is not None:
        cfg = load_config(config_path)
    elif config is not None:
        cfg = config
    else:
        raise ValueError("Checkpoint không chứa config, hãy truyền --config")
    model = build_model(cfg.model, pretrained_override=False)
    load_weights(model, ckpt)
    model.eval()
    return model, cfg, ckpt
