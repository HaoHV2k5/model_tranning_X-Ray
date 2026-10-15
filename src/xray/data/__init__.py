"""Tạo Dataset / DataLoader từ config."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..utils import resolve_path, worker_init_fn
from .dataset import CheXpertDataset
from .transforms import build_transforms

SPLIT_TO_CSV_KEY = {"train": "train_csv", "val": "val_csv", "test": "test_csv"}


def build_dataset(cfg, split: str, train_mode: bool = None) -> CheXpertDataset:
    """train_mode=True -> có augmentation. Mặc định chỉ tập 'train' được augment."""
    if split not in SPLIT_TO_CSV_KEY:
        raise ValueError(f"split phải thuộc {list(SPLIT_TO_CSV_KEY)}")
    train_mode = (split == "train") if train_mode is None else train_mode
    d = cfg.data
    return CheXpertDataset(
        root=resolve_path(d.root),
        csv_file=d[SPLIT_TO_CSV_KEY[split]],
        transform=build_transforms(d, train=train_mode),
        label_start_col=int(d.label_start_col),
        uncertain_policy=d.uncertain_policy,
        cache_in_ram=bool(d.cache_in_ram),
        limit=d.get("limit_samples"),
    )


def build_loader(dataset, cfg, train: bool, seed: int = 0) -> DataLoader:
    num_workers = int(cfg.data.num_workers)
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = dict(
        batch_size=int(cfg.train.batch_size),
        shuffle=train,
        drop_last=train and len(dataset) > int(cfg.train.batch_size),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


__all__ = ["CheXpertDataset", "build_dataset", "build_loader", "build_transforms"]
