"""
Chạy từ thư mục gốc hoặc thư mục MedicalPatchNet (đã activate .venv):
    python3 predict_test.py                      # ngẫu nhiên hoặc ảnh số 0 trong test.csv
    python3 predict_test.py 5                    # lấy ảnh số 5 trong test.csv
    python3 predict_test.py 0 savedModels/MedicalPatchNet_weights.pt
Kết quả: in xác suất 14 bệnh + lưu predict_result.png (ảnh gốc + heatmap bệnh cao nhất)
Đã tối ưu hóa cho Mac M1/M2/M3 (Apple Silicon MPS - tăng tốc ~45x so với CPU).
"""
import os, sys, csv, random
import torch
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Hỗ trợ chạy từ thư mục gốc hoặc MedicalPatchNet/
if os.path.exists("MedicalPatchNet") and "MedicalPatchNet" not in sys.path:
    sys.path.insert(0, "MedicalPatchNet")
from NetworkModel import ScalePatchNet

def find_root():
    env_root = os.environ.get("CHEXPERT_ROOT")
    if env_root and os.path.exists(env_root):
        return env_root
    for candidate in ["chexpert_subset", "../chexpert_subset", "./chexpert_subset"]:
        if os.path.exists(candidate):
            return candidate
    return "chexpert_subset"

def find_weights(arg_w):
    candidates = [
        arg_w,
        "savedModels/MedicalPatchNet_weights.pt",
        "MedicalPatchNet/savedModels/MedicalPatchNet_weights.pt",
        "savedModels/MedicalPatchNet.pt",
        "MedicalPatchNet/savedModels/MedicalPatchNet.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return arg_w or "savedModels/MedicalPatchNet_weights.pt"

ROOT = find_root()
raw_idx = sys.argv[1] if len(sys.argv) > 1 else None
weights_arg = sys.argv[2] if len(sys.argv) > 2 else None
weights = find_weights(weights_arg)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Thiết bị tính toán: {device.type.upper()} ({'Apple Silicon GPU (MPS)' if device.type == 'mps' else 'CPU'})")

# đọc nhãn thật từ test.csv để so sánh
test_csv_path = os.path.join(ROOT, "test.csv")
with open(test_csv_path) as f:
    rows = list(csv.reader(f))
head, data = rows[0], rows[1:]
labels = head[5:]

if raw_idx is None:
    idx = 0
else:
    idx = int(raw_idx)
row = data[idx]
img_path = os.path.join(ROOT, row[0])
truth = {n: v for n, v in zip(labels, row[5:])}

model = ScalePatchNet(patchSize=64, outFeatures=14).to(device)
sd = torch.load(weights, map_location=device, weights_only=False)
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval()

img = Image.open(img_path).convert("L")
x = F.resize(F.to_tensor(img), [512, 512], antialias=True).unsqueeze(0).to(device)

with torch.inference_mode():
    patch_logits = model.forwardRawPatches(x)            # (1, 64, 14)
    probs = torch.sigmoid(patch_logits.mean(dim=1))[0].cpu()

print(f"Ảnh #{idx}: {img_path}\n")
print(f"{'Bệnh':30s} {'Dự đoán':>8s}   Nhãn thật")
order = sorted(range(14), key=lambda i: -probs[i].item())
for i in order:
    t = truth[labels[i]]
    t = {"1.0": "CÓ", "0.0": "không", "-1.0": "không chắc", "": "-"}.get(t, t)
    print(f"{labels[i]:30s} {probs[i].item():8.2f}   {t}")

top = order[0]
heat = patch_logits[0, :, top].reshape(8, 8).cpu().numpy()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("Ảnh gốc"); ax[0].axis("off")
ax[1].imshow(img, cmap="gray")
ax[1].imshow(heat, cmap="jet", alpha=0.45, extent=(0, img.size[0], img.size[1], 0), interpolation="bilinear")
ax[1].set_title(f"Heatmap: {labels[top]}"); ax[1].axis("off")
plt.tight_layout()
out_png = "predict_result.png"
plt.savefig(out_png, dpi=120)
print(f"\nĐã lưu {out_png}")
