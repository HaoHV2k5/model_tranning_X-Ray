#!/bin/bash
set -e

# Xác định thư mục script và thư mục dự án
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Chuyển vào thư mục MedicalPatchNet nếu đang ở thư mục gốc
if [ -d "$SCRIPT_DIR/MedicalPatchNet" ]; then
    cd "$SCRIPT_DIR/MedicalPatchNet"
fi

# Tự động tìm đường dẫn chexpert_subset
if [ -z "$CHEXPERT_ROOT" ]; then
    if [ -d "../chexpert_subset" ]; then
        export CHEXPERT_ROOT="../chexpert_subset"
    elif [ -d "./chexpert_subset" ]; then
        export CHEXPERT_ROOT="./chexpert_subset"
    fi
fi

# Cấu hình tối ưu PyTorch cho Apple Silicon (MPS)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

mkdir -p savedModels

# Ưu tiên python từ .venv nếu có
if [ -f "../.venv/bin/python3" ]; then
    PYTHON_BIN="../.venv/bin/python3"
elif [ -f "./.venv/bin/python3" ]; then
    PYTHON_BIN="./.venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

echo "=== KHỞI ĐỘNG HUẤN LUYỆN TRÊN MAC (APPLE SILICON MPS) ==="
echo "Python: $PYTHON_BIN"
echo "Dataset Root: $CHEXPERT_ROOT"
echo "Epochs: ${EPOCHS:-1}"
echo "Batch size: ${BATCH_SIZE:-16}"

$PYTHON_BIN trainClassification.py \
    -lr 1e-4 \
    -bs ${BATCH_SIZE:-16} \
    -ep ${EPOCHS:-1} \
    -patchSize 64 \
    -imgSize 512 \
    -pr 2 \
    -savePath MedicalPatchNet.pt
