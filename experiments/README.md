# Nhật ký thí nghiệm (Experiments)

Thư mục này lưu trữ toàn bộ cấu hình, kết quả và nhật ký của các lần huấn luyện mô hình:

- `20261015-093837_medicalpatchnet`: Chạy thử nghiệm ban đầu
- `20261015-115340_medicalpatchnet`: Chạy tinh chỉnh tham số
- `20261015-175050_medicalpatchnet`: Lần chạy chính đạt benchmark Macro-AUROC **0.8007** và Sensitivity **77.51%**

## Cấu trúc mỗi thư mục lần chạy:
- `config.yaml`: Toàn bộ cấu hình siêu tham số (đảm bảo tái lập kết quả)
- `env.json`: Môi trường runtime, seed, thiết bị (GPU/CPU/MPS)
- `history.csv` & `val_per_class.csv`: Lịch sử training và AUROC từng bệnh qua các epoch
- `summary.json`: Báo cáo tổng kết metrics
- `figures/`: Biểu đồ đường cong loss, AUROC, độ chính xác, per-class AUROC
- `test/`: Kết quả đánh giá chi tiết trên tập TEST độc lập (ROC curves, confusion matrices, bootstrap CI, predictions.csv)
- `tensorboard/`: File log sự kiện trực quan hóa bằng TensorBoard

> **Lưu ý về Checkpoint Weights (`.pt`):**
> Các file trọng số mô hình (`best.pt`, `last.pt` dung lượng ~232MB/file) được loại trừ khỏi Git để tuân thủ giới hạn file của GitHub (< 100MB) và được lưu trữ trên Google Drive / Cloud Storage.
