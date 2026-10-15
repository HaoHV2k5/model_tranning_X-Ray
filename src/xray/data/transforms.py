"""Biến đổi ảnh (giống MedicalPatchNet gốc)."""
from __future__ import annotations

from torchvision.transforms import v2


def build_transforms(data_cfg, train: bool):
    size = (int(data_cfg.img_size), int(data_cfg.img_size))
    aug = data_cfg.get("augment", {}) or {}
    if train and aug.get("enabled", False):
        return v2.Compose(
            [
                v2.RandomResizedCrop(size=size, scale=tuple(aug.get("rrc_scale", (0.5, 1.0))), antialias=True),
                v2.RandomRotation(degrees=float(aug.get("rotation_deg", 5))),
                v2.ColorJitter(brightness=float(aug.get("brightness", 0.3))),
            ]
        )
    return v2.Compose([v2.Resize(size=size, antialias=True)])
