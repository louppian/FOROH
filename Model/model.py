import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from Dataset.dataset import NUM_CLASSES


SUPPORTED_BACKBONES = ("coatnet_2", "inception_v3", "resnet18")
SUPPORTED_LOSSES = ("cdw_ce", "ce", "foroh")


class ClassificationModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.net = _create_timm_model(backbone, NUM_CLASSES)

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return {"logits": out}


def _create_timm_model(backbone, num_classes):
    name = backbone.lower()
    if name == "inception_v3":
        return timm.create_model("inception_v3", pretrained=True, num_classes=num_classes)
    if name == "coatnet_2":
        return timm.create_model("coatnet_2_rw_224.sw_in12k_ft_in1k", pretrained=True, num_classes=num_classes)
    if name == "resnet18":
        return timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    raise ValueError(f"Unknown backbone: {backbone}")

class FOROHModel(nn.Module):
    def __init__(self, backbone, proj_dim=128):
        super().__init__()
        self.backbone = _create_timm_model(backbone, 0)
        feat_dim = self.backbone.num_features
        self.projector = nn.Sequential(nn.Linear(feat_dim, proj_dim), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(proj_dim, proj_dim))
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))

    def forward(self, x):
        z = self.backbone(x)
        if z.ndim > 2:
            z = z.flatten(1)
        u = F.normalize(self.projector(z), dim=-1)
        w = F.normalize(self.w_raw, dim=0)
        with torch.amp.autocast("cuda", enabled=False):
            cos = torch.clamp(u.float() @ w.float(), -1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cos)
            score = theta / math.pi * (NUM_CLASSES - 1)
        return {"score": score, "theta": theta, "embedding": u}


def build_model(backbone, loss_name):
    if loss_name == "foroh":
        return FOROHModel(backbone)
    return ClassificationModel(backbone)
