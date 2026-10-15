"""Dự đoán MỘT ảnh và xuất heatmap từng patch.

Ví dụ:
    # Dự đoán ảnh đầu tiên trong tập test (lấy từ test.csv)
    python scripts/predict.py --checkpoint experiments/.../checkpoints/best.pt

    # Dự đoán ảnh bất kỳ
    python scripts/predict.py \\
        --checkpoint experiments/.../checkpoints/best.pt \\
        --image data/chexpert_subset/images/train/patient48197/study1/view1_frontal.jpg

    # Lấy ảnh thứ 5 (index 4) trong tập val
    python scripts/predict.py \\
        --checkpoint experiments/.../checkpoints/best.pt \\
        --split val --index 4
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from xray.engine import load_model_for_inference
from xray.data import build_dataset
from xray.tracking.logger import setup_logging
from xray.utils import get_device, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Dự đoán 1 ảnh X-quang và xuất heatmap")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", type=str, default=None, help="Đường dẫn ảnh tùy ý")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--index", type=int, default=0, help="Chỉ số ảnh trong tập --split")
    p.add_argument("--out", type=str, default="predict_result.png")
    p.add_argument("--top-k", type=int, default=3, help="Số bệnh hàng đầu hiển thị heatmap")
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()


def load_image(path, img_size: int) -> torch.Tensor:
    img = Image.open(str(path)).convert("L")
    img = TF.resize(TF.to_tensor(img), [img_size, img_size], antialias=True)
    return img.unsqueeze(0)  # [1,1,H,W]


def main():
    args = parse_args()
    log = setup_logging()
    cfg_path = Path(args.config) if args.config else None
    model, cfg, _ = load_model_for_inference(Path(args.checkpoint), config_path=cfg_path)
    seed_everything(int(cfg.experiment.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    model = model.to(device).eval()
    img_size = int(cfg.data.img_size)

    # Lấy ảnh và nhãn thật (nếu từ dataset)
    img_path, truth = None, None
    if args.image:
        img_path = Path(args.image)
    else:
        ds = build_dataset(cfg, args.split, train_mode=False)
        idx = args.index % len(ds)
        img_path = ds.image_path(idx)
        truth = {ds.class_names[c]: int(ds.targets[idx, c].item()) for c in range(ds.num_classes)
                 if ds.masks[idx, c].item() > 0}

    log.info(f"Ảnh: {img_path}")
    pil = Image.open(str(img_path)).convert("L")
    x = load_image(img_path, img_size).to(device, dtype=torch.float32)

    has_patch = hasattr(model, "forward_patches")
    with torch.inference_mode():
        if has_patch:
            patch_logits = model.forward_patches(x)           # [1, n_patch, C]
            probs = torch.sigmoid(patch_logits.mean(1))[0]    # [C]
        else:
            probs = torch.sigmoid(model(x))[0]
            patch_logits = None

    class_names = ds.class_names if not args.image else [f"class_{i}" for i in range(len(probs))]

    # In bảng kết quả
    order = torch.argsort(probs, descending=True).tolist()
    print(f"\n{'Bệnh':30s} {'Xác suất':>9s}   Nhãn thật")
    for c in order:
        true_label = ("CÓ" if truth[class_names[c]] == 1 else "không") if (truth and class_names[c] in truth) else "-"
        print(f"{class_names[c]:30s}  {probs[c].item():7.1%}   {true_label}")

    # Vẽ heatmap
    top_classes = order[:args.top_k]
    n_top = len(top_classes)
    cols = 1 + n_top
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4.5))
    ax_img = axes[0]
    ax_img.imshow(pil, cmap="gray")
    ax_img.set_title(f"Ảnh gốc\n{img_path.name}", fontsize=9)
    ax_img.axis("off")

    if has_patch and patch_logits is not None:
        nh, nw = model.patch_grid(img_size) if hasattr(model, "patch_grid") else (img_size // model.patch_size,) * 2
        for k, c in enumerate(top_classes):
            prob_val = probs[c].item()
            heat = patch_logits[0, :, c].reshape(nh, nw).cpu().numpy()
            ax = axes[k + 1]
            ax.imshow(pil, cmap="gray")
            ax.imshow(heat, cmap="jet", alpha=0.45,
                      extent=(0, pil.width, pil.height, 0), interpolation="bilinear")
            title = f"{class_names[c]}\np={prob_val:.1%}"
            if truth and class_names[c] in truth:
                title += f" | GT={'CÓ' if truth[class_names[c]] else 'Không'}"
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    else:
        for ax in axes[1:]:
            ax.axis("off")

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=150)
    log.info(f"Đã lưu kết quả: {out}")


if __name__ == "__main__":
    main()
