"""MedicalPatchNet (port từ MedicalPatchNet/NetworkModel.py - lớp ScalePatchNet).

Ý tưởng: chia ảnh thành lưới patch không chồng lấn, mỗi patch được phân loại ĐỘC LẬP
bằng cùng một backbone CNN, logits cuối = trung bình logits của các patch.
=> Logits từng patch chính là bản đồ giải thích (heatmap) mà không cần Grad-CAM.

Tên thuộc tính `baseBackbone` được giữ nguyên để load được checkpoint cũ
(MedicalPatchNet/savedModels/MedicalPatchNet_weights.pt).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from . import register_model

_BACKBONES = {
    "efficientnet_v2_s": (torchvision.models.efficientnet_v2_s, "EfficientNet_V2_S_Weights"),
    "efficientnet_b0": (torchvision.models.efficientnet_b0, "EfficientNet_B0_Weights"),
}


def _build_efficientnet(name: str, pretrained: bool, in_channels: int, num_classes: int, init_gray_from_rgb: bool):
    if name not in _BACKBONES:
        raise KeyError(f"Backbone '{name}' chưa hỗ trợ. Có sẵn: {list(_BACKBONES)}")
    fn, weights_name = _BACKBONES[name]
    weights = getattr(torchvision.models, weights_name).IMAGENET1K_V1 if pretrained else None
    net = fn(weights=weights)

    # Lớp conv đầu: 3 kênh RGB -> in_channels (ảnh X-quang là ảnh xám 1 kênh)
    old_conv: nn.Conv2d = net.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    if init_gray_from_rgb and pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels)
    net.features[0][0] = new_conv

    # Lớp phân loại cuối -> num_classes
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, num_classes, bias=True)
    return net


@register_model("medicalpatchnet")
class MedicalPatchNet(nn.Module):
    supports_patch_heatmap = True

    def __init__(
        self,
        num_classes: int = 14,
        patch_size: int = 64,
        backbone: str = "efficientnet_v2_s",
        pretrained: bool = True,
        in_channels: int = 1,
        init_gray_from_rgb: bool = False,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.num_classes = num_classes
        self.baseBackbone = _build_efficientnet(backbone, pretrained, in_channels, num_classes, init_gray_from_rgb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,1,H,W] -> logits [B,C]."""
        return self.forward_patches(x).mean(dim=1)

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,1,H,W] -> logits từng patch [B, số_patch, C] (thứ tự hàng-trước-cột)."""
        b, _, h, w = x.shape
        p = self.patch_size
        if h % p or w % p:
            raise ValueError(f"Kích thước ảnh ({h}x{w}) phải chia hết cho patch_size={p}")
        x = x.unfold(2, p, p).unfold(3, p, p)  # [B,C,nh,nw,p,p]
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, x.size(1), p, p)
        x = self.baseBackbone(x)
        return x.view(b, -1, x.size(-1))

    def forward_scaled_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Logits patch nhân với xác suất toàn ảnh (heatmap 'scaled' trong paper gốc)."""
        patch_logits = self.forward_patches(x)
        global_prob = torch.sigmoid(patch_logits.mean(dim=1)).unsqueeze(1)
        return patch_logits * global_prob

    def patch_grid(self, img_size: int):
        n = img_size // self.patch_size
        return n, n

    # Tên cũ để tương thích với code MedicalPatchNet gốc
    forwardRawPatches = forward_patches
    forwardScaledPatches = forward_scaled_patches
