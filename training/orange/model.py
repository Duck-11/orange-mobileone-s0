"""MobileOne-S0 thật, lấy từ package `timm` (port của apple/ml-mobileone).
Thay thế hoàn toàn MobileNetV3Small cũ.

Quy ước I/O (giống tinh thần `include_preprocessing=True` cũ):
  - Input : ảnh RGB, float32, NCHW, giá trị 0..255 (chưa chia 255).
  - Chuẩn hoá mean/std lấy ĐÚNG từ pretrained_cfg của weights timm, nằm TRONG model.
  - Output: logit BAD (p_BAD = sigmoid). Nhãn GOOD=0, BAD=1.
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn

TIMM_NAME = "mobileone_s0.apple_in1k"   # variant S0 + weights ImageNet của Apple (qua timm)
IMAGE_SIZE = 224
LABELS = {"GOOD": 0, "BAD": 1}


class OrangeMobileOneS0(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2, timm_name: str = TIMM_NAME):
        super().__init__()
        import timm
        self.timm_name = timm_name
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        cfg = self.backbone.pretrained_cfg
        self.norm_cfg = {"mean": list(cfg["mean"]), "std": list(cfg["std"]),
                         "input_size": list(cfg["input_size"])}
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1) * 255.0
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1) * 255.0
        self.register_buffer("pix_mean", mean, persistent=False)
        self.register_buffer("pix_std", std, persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x_0_255: torch.Tensor) -> torch.Tensor:
        x = (x_0_255 - self.pix_mean) / self.pix_std
        return self.head(self.dropout(self.backbone(x))).squeeze(1)

    def p_bad(self, x_0_255: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_0_255))


class DeployPBad(nn.Module):
    """Bọc để ONNX xuất thẳng p_BAD (sigmoid nằm trong graph)."""
    def __init__(self, m: OrangeMobileOneS0):
        super().__init__(); self.m = m
    def forward(self, x):
        return self.m.p_bad(x).unsqueeze(1)


def reparameterize(model: OrangeMobileOneS0) -> OrangeMobileOneS0:
    """Bản deploy (đã gộp nhánh). Trả BẢN SAO, model train giữ nguyên."""
    from timm.utils import reparameterize_model
    return reparameterize_model(copy.deepcopy(model).eval(), inplace=True)


def freeze_batchnorm(model: nn.Module) -> None:
    """Dùng cho phase 1 (BN frozen). Gọi lại SAU mỗi lần model.train()."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def save_checkpoint(model: OrangeMobileOneS0, path, extra: dict | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "timm_name": model.timm_name,
                "norm_cfg": model.norm_cfg, "labels": LABELS, "extra": extra or {}}, path)


def load_checkpoint(path, dropout: float = 0.2) -> OrangeMobileOneS0:
    ck = torch.load(path, map_location="cpu",weights_only=True)
    m = OrangeMobileOneS0(pretrained=False, dropout=dropout, timm_name=ck["timm_name"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()
