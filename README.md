# Dự án X-quang — CheXpert Multi-label Classification

Phân loại 14 bệnh trên ảnh X-quang ngực (CheXpert subset: 4000 train / 500 val / 500 test).

## Cấu trúc dự án

```
x-quang/
├── configs/                  # Cấu hình train (YAML)
│   ├── base.yaml             # Tham số mặc định đầy đủ (có giải thích từng dòng)
│   ├── medicalpatchnet.yaml  # Model chính
│   ├── efficientnet_baseline.yaml  # Baseline so sánh
│   └── debug.yaml            # Chạy thử nhanh (vài phút)
├── data/
│   └── chexpert_subset/      # Dataset (4000+500+500 ảnh, 14 nhãn bệnh)
├── src/xray/                 # Thư viện code chính
│   ├── config.py             # Đọc YAML + ghi đè từ dòng lệnh
│   ├── data/                 # Dataset, DataLoader, transforms
│   ├── models/               # Registry model (thêm model mới = 1 file)
│   ├── engine/               # Trainer, Evaluator, Checkpoint, Loss, Scheduler
│   ├── metrics.py            # AUROC, AUPRC, F1, bootstrap CI (numpy)
│   └── tracking/             # Logger (CSV + JSON + TensorBoard), Plots
├── scripts/
│   ├── train.py              # Huấn luyện
│   ├── evaluate.py           # Đánh giá checkpoint trên tập test/val
│   ├── predict.py            # Dự đoán 1 ảnh + heatmap
│   └── compare_runs.py       # Bảng xếp hạng tất cả lần chạy
├── experiments/              # Mỗi lần chạy = 1 thư mục riêng (không bị ghi đè)
│   └── 20261015-HHMMSS_<tên>/
│       ├── config.yaml       # Config đã dùng (tái lập kết quả)
│       ├── env.json          # Phiên bản thư viện, thiết bị, seed
│       ├── train.log         # Log chi tiết
│       ├── history.csv       # Loss/AUROC/F1... theo từng epoch
│       ├── val_per_class.csv # AUROC từng bệnh theo từng epoch
│       ├── summary.json      # Tóm tắt + kết quả test cuối cùng
│       ├── checkpoints/
│       │   ├── best.pt       # Checkpoint tốt nhất (theo val AUROC)
│       │   └── last.pt       # Checkpoint epoch cuối (dùng để train tiếp)
│       ├── tensorboard/      # Xem bằng: tensorboard --logdir experiments/
│       ├── figures/          # loss_curve.png, auroc_curve.png, per_class_auroc.png...
│       └── test/             # metrics.json, per_class.csv, roc_curves.png, predictions.csv
├── legacy/                   # Script cũ (tham khảo, không dùng nữa)
└── MedicalPatchNet/          # Code gốc (tham khảo)
```

## Cách dùng

### 1. Kích hoạt môi trường ảo

```bash
# Cách 1: kích hoạt một lần, rồi gõ python bình thường trong session đó
source .venv/bin/activate

# Cách 2: gọi trực tiếp mà không cần activate (dùng khi chạy từ bất kỳ thư mục nào)
# Thay python bằng .venv/bin/python trong mọi lệnh bên dưới
```

> **Lưu ý Mac:** lệnh `python` không tồn tại mặc định trên macOS.  
> Sau khi `source .venv/bin/activate`, dùng `python` bình thường.  
> Nếu chưa activate, dùng `.venv/bin/python`.

### 2. Chạy thử (kiểm tra pipeline, vài phút)

```bash
# Sau khi activate:
python scripts/train.py --config configs/debug.yaml

# Không activate:
.venv/bin/python scripts/train.py --config configs/debug.yaml
```

Kiểm tra kết quả:
```bash
ls experiments/         # thấy thư mục mới
ls experiments/*/figures/   # loss_curve.png, auroc_curve.png...
```

### 3. Train thật sự

```bash
# Train MedicalPatchNet (20 epoch, ~2-5h tuỳ máy)
python scripts/train.py --config configs/medicalpatchnet.yaml

# Đổi tham số không cần sửa file
python scripts/train.py --config configs/medicalpatchnet.yaml \
    --name lr3e-4 \
    --set train.optimizer.lr=3e-4 train.epochs=30
```

### 4. Theo dõi trong lúc train

```bash
# Mở terminal mới, xem biểu đồ realtime
tensorboard --logdir experiments/
# Truy cập http://localhost:6006
```

### 5. Xem kết quả sau khi train

```bash
# Bảng xếp hạng tất cả lần chạy
python scripts/compare_runs.py

# Đánh giá thêm (nếu cần)
python scripts/evaluate.py --checkpoint experiments/<tên_thư_mục>/checkpoints/best.pt

# Dự đoán ảnh đầu tiên trong tập test + heatmap
python scripts/predict.py --checkpoint experiments/<tên_thư_mục>/checkpoints/best.pt
```

### 6. Train tiếp khi bị dừng giữa chừng

```bash
python scripts/train.py --resume experiments/<tên_thư_mục>

# Tăng số epoch khi resume
python scripts/train.py --resume experiments/<tên_thư_mục> --set train.epochs=30
```

## Thêm model mới

```python
# Tạo src/xray/models/densenet.py
import torch.nn as nn
from . import register_model

@register_model("densenet121")
class DenseNet121(nn.Module):
    def __init__(self, num_classes=14, **kwargs):
        ...
```

```yaml
# configs/densenet.yaml
_base_: base.yaml
experiment:
  name: densenet121
model:
  name: densenet121
  num_classes: 14
```

## Metric chính

| Metric | Ý nghĩa |
|--------|---------|
| **AUROC** | Khả năng phân biệt ca bệnh vs không bệnh (không phụ thuộc ngưỡng). 0.5 = đoán mò, 1.0 = hoàn hảo. **Metric chính của CheXpert.** |
| **AUPRC** | Average Precision. Phù hợp khi bệnh hiếm. |
| **F1** | Kết hợp Precision và Recall. Dùng ngưỡng Youden J tối ưu trên tập val. |
| **Sensitivity** | Tỷ lệ phát hiện đúng ca bệnh (quan trọng trong y tế). |
| **Specificity** | Tỷ lệ phát hiện đúng ca không bệnh. |

## Kết quả thực nghiệm (Benchmark)

Mô hình MedicalPatchNet được huấn luyện trên tập CheXpert subset (4.000 train / 500 val / 500 test), đánh giá độc lập trên tập **TEST**:

| Bệnh | AUROC | Độ nhạy (Sens) | Độ đặc hiệu (Spec) | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Support Devices** | **0.892** | 94.7% | 73.6% | 0.474 |
| **Edema** | **0.868** | 74.5% | 81.5% | 0.442 |
| **Consolidation** | **0.863** | 80.4% | 75.5% | 0.406 |
| **Pleural Effusion** | **0.853** | 59.6% | 85.0% | 0.413 |
| **Lung Opacity** | **0.847** | 91.5% | 67.1% | 0.419 |
| **Atelectasis** | **0.838** | 82.4% | 73.9% | 0.400 |
| **Pneumonia** | **0.809** | 90.2% | 57.2% | 0.318 |
| **Pneumothorax** | **0.797** | 88.2% | 49.7% | 0.280 |
| **Cardiomegaly** | **0.795** | 72.5% | 75.5% | 0.374 |
| **Pleural Other** | **0.777** | 62.7% | 75.5% | 0.332 |
| **Lung Lesion** | **0.758** | 76.5% | 66.1% | 0.322 |
| **No Finding** | **0.713** | 74.5% | 61.0% | 0.288 |
| **Fracture** | **0.701** | 68.6% | 64.4% | 0.285 |
| **Enlarged Cardiomediastinum** | **0.699** | 68.6% | 61.7% | 0.271 |
| **TRUNG BÌNH (Macro Mean)** | **`0.8007`** | **`77.51%`** | **`69.13%`** | **`0.3588`** |

> **Ghi chú khoa học:** So với bài báo gốc (huấn luyện trên toàn bộ 223.414 ảnh của CheXpert), mô hình đạt **Độ nhạy 77.51%** (vượt mức 76.3% của bài báo) và **Macro-AUROC 0.8007** ngay trong điều kiện dữ liệu thu nhỏ (chỉ 4.000 ảnh).

## Chạy trên Google Colab (GPU A100 / T4)

```python
# 1. Clone repo và cài đặt
!git clone https://github.com/HaoHV2k5/model_tranning_X-Ray.git
%cd model_tranning_X-Ray
!pip install -q -r requirements.txt

# 2. Giải nén dataset từ Google Drive
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p data
!unzip -q "/content/drive/MyDrive/chexpert_subset_new.zip" -d data/

# 3. Train siêu tốc với GPU NVIDIA + FP16
!python scripts/train.py --config configs/medicalpatchnet.yaml \
    --set train.epochs=50 train.batch_size=32 train.amp=true
```

## Lưu ý

- **Chống ngủ trên macOS:** Dùng lệnh `caffeinate -dims python scripts/train.py ...` để máy Mac không bị tắt màn hình / sleep giữa chừng.
- **Nhãn không chắc (-1.0):** mặc định coi là "không bệnh" (`uncertain_policy: zeros`), giống paper MedicalPatchNet gốc. Đổi sang `ones` hoặc `ignore` trong config.
- **Ngưỡng xác suất:** AUROC không cần ngưỡng. Các metric khác (F1, Acc...) dùng ngưỡng Youden J tối ưu trên **tập val** để tránh rò rỉ thông tin từ tập test.
- **Tái lập kết quả:** seed và toàn bộ config được lưu trong từng thư mục thí nghiệm.
