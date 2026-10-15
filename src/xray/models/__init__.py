"""Registry model: thêm model mới chỉ cần tạo file + gắn decorator @register_model("ten")."""
from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    def decorator(cls):
        key = name.lower()
        if key in MODEL_REGISTRY:
            raise KeyError(f"Model '{key}' đã được đăng ký")
        MODEL_REGISTRY[key] = cls
        return cls

    return decorator


def list_models():
    return sorted(MODEL_REGISTRY)


def build_model(model_cfg, pretrained_override=None) -> nn.Module:
    """Tạo model từ phần `model:` trong config.

    pretrained_override=False dùng khi load checkpoint (không cần tải lại trọng số ImageNet).
    """
    name = str(model_cfg.name).lower()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' chưa đăng ký. Có sẵn: {list_models()}")
    params = dict(model_cfg.get("params", {}) or {})
    if pretrained_override is not None and "pretrained" in params:
        params["pretrained"] = pretrained_override
    return MODEL_REGISTRY[name](num_classes=int(model_cfg.num_classes), **params)


# Import các module model để chúng tự đăng ký vào registry.
from . import patchnet  # noqa: E402,F401
