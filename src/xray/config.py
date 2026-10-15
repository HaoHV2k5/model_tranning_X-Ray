"""Đọc cấu hình YAML có kế thừa (`_base_`) và ghi đè từ dòng lệnh (`--set a.b=c`)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

BASE_KEY = "_base_"


class Config(dict):
    """dict cho phép truy cập kiểu thuộc tính: cfg.train.epochs == cfg["train"]["epochs"]."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = _wrap(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"Config không có khoá '{key}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _wrap(value)

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(self.to_dict(), memo))

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> Dict[str, Any]:
        return _unwrap(self)


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Gộp đệ quy: giá trị trong `override` thay thế giá trị trong `base`."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_with_bases(path: Path, _seen: Optional[set] = None) -> Dict[str, Any]:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Kế thừa config bị lặp vòng tại {path}")
    _seen.add(path)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    base_ref = data.pop(BASE_KEY, None)
    if base_ref is None:
        return data
    bases = [base_ref] if isinstance(base_ref, str) else list(base_ref)
    merged: Dict[str, Any] = {}
    for ref in bases:
        merged = deep_merge(merged, _load_yaml_with_bases(path.parent / ref, _seen))
    return deep_merge(merged, data)


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Áp dụng các chuỗi dạng `train.epochs=5`. Giá trị được parse bằng YAML (5 -> int, true -> bool...)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override sai định dạng '{item}', cần dạng khoa.con=giatri")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Không thể ghi đè '{dotted}': '{part}' không phải nhóm cấu hình")
        node[parts[-1]] = value
    return cfg


def load_config(path, overrides: Optional[Iterable[str]] = None) -> Config:
    data = _load_yaml_with_bases(Path(path))
    data = apply_overrides(data, overrides or [])
    return Config(data)


def save_config(cfg: Config, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
