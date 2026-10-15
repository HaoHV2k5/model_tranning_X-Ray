"""Thiết lập chung cho các script: thêm src/ vào sys.path và biến môi trường cho Apple Silicon."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Cho phép các phép toán chưa hỗ trợ trên MPS tự chạy bằng CPU thay vì báo lỗi
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
