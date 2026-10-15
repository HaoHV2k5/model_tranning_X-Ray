"""Dataset CheXpert (port từ MedicalPatchNet/ChexpertDataset.py, viết lại gọn và cấu hình được)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchvision.io as tv_io
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

UNCERTAIN_POLICIES = ("zeros", "ones", "ignore")


def _parse_label(raw: str, policy: str) -> Tuple[float, float]:
    """Trả về (nhãn, mask). mask=0 nghĩa là bỏ qua khi tính loss/metric.

    Quy ước CheXpert: 1.0 = có bệnh, 0.0 = không, -1.0 = không chắc, rỗng = không được ghi nhận.
    Giống code gốc: ô rỗng được coi là 0 (không bệnh).
    """
    raw = raw.strip()
    if raw in ("", "0.0", "0"):
        return 0.0, 1.0
    if raw in ("1.0", "1"):
        return 1.0, 1.0
    if raw in ("-1.0", "-1"):
        if policy == "zeros":
            return 0.0, 1.0
        if policy == "ones":
            return 1.0, 1.0
        return 0.0, 0.0  # ignore
    raise ValueError(f"Giá trị nhãn không hợp lệ: '{raw}'")


class CheXpertDataset(Dataset):
    """Mỗi phần tử trả về (ảnh [1,H,W] float 0..1, nhãn [C], mask [C])."""

    def __init__(
        self,
        root,
        csv_file: str,
        transform: Optional[Callable] = None,
        label_start_col: int = 5,
        uncertain_policy: str = "zeros",
        cache_in_ram: bool = False,
        limit: Optional[int] = None,
        strip_prefix: str = "CheXpert-v1.0/",
    ):
        if uncertain_policy not in UNCERTAIN_POLICIES:
            raise ValueError(f"uncertain_policy phải thuộc {UNCERTAIN_POLICIES}")
        self.root = Path(root)
        self.csv_path = self.root / csv_file
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            header, *rows = list(csv.reader(f))
        if limit is not None:
            rows = rows[: int(limit)]

        self.class_names: List[str] = header[label_start_col:]
        self.meta_columns: List[str] = header[:label_start_col]
        self.transform = transform
        self.cache_in_ram = cache_in_ram
        self._cache: Dict[int, torch.Tensor] = {}

        self.rel_paths: List[str] = []
        self.meta: List[Dict[str, str]] = []
        targets, masks = [], []
        for row in rows:
            rel = row[0].replace(strip_prefix, "")
            self.rel_paths.append(rel)
            self.meta.append(dict(zip(self.meta_columns, row[:label_start_col])))
            parsed = [_parse_label(v, uncertain_policy) for v in row[label_start_col:]]
            targets.append([p[0] for p in parsed])
            masks.append([p[1] for p in parsed])

        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)

    # ------------------------------------------------------------------ helpers
    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def image_path(self, idx: int) -> Path:
        return self.root / self.rel_paths[idx]

    def positive_counts(self) -> torch.Tensor:
        return (self.targets * self.masks).sum(dim=0)

    def pos_weight(self, max_weight: float = 50.0) -> torch.Tensor:
        """Trọng số cho lớp dương = số âm / số dương (dùng cho BCE khi dữ liệu mất cân bằng)."""
        pos = self.positive_counts()
        neg = self.masks.sum(dim=0) - pos
        return (neg / pos.clamp(min=1.0)).clamp(max=max_weight)

    def load_image_uint8(self, idx: int) -> torch.Tensor:
        """Đọc ảnh xám dạng uint8 [1,H,W] (cache uint8 tốn RAM ít hơn 4 lần so với float)."""
        if self.cache_in_ram and idx in self._cache:
            return self._cache[idx]
        path = str(self.image_path(idx))
        try:
            img = tv_io.read_image(path, tv_io.ImageReadMode.GRAY)
        except Exception:
            img = pil_to_tensor(Image.open(path).convert("L"))
        if self.cache_in_ram:
            self._cache[idx] = img
        return img

    # ------------------------------------------------------------------ Dataset API
    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int):
        img = self.load_image_uint8(idx).float() / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx], self.masks[idx]
