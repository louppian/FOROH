"""E01 model definitions split from the original root 3_train.py.

Only the three coordinate variants differ. Backbone/projector structure follows the
original training code.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class _Projector(nn.Sequential):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(proj_dim, proj_dim),
        )


class EuclideanHuberHead(nn.Module):
    """Matched Euclidean scalar regression control."""

    def __init__(self, in_dim: int, proj_dim: int = 128, c_max: int = 3):
        super().__init__()
        self.c_max = c_max
        self.projector = _Projector(in_dim, proj_dim)
        self.fc = nn.Linear(proj_dim, 1, bias=False)

    def forward(self, z):
        score = self.fc(self.projector(z)).squeeze(-1)
        return score, None, None, None


class NormalizedCosineHead(nn.Module):
    """Same hypersphere/projector as FOROH, cosine score mapping only."""

    def __init__(self, in_dim: int, proj_dim: int = 128, c_max: int = 3):
        super().__init__()
        self.c_max = c_max
        self.projector = _Projector(in_dim, proj_dim)
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))

    def _get_w(self):
        return self.w_raw / (self.w_raw.norm() + 1e-8)

    def forward(self, z):
        u = F.normalize(self.projector(z), dim=-1)
        w = self._get_w()
        with torch.amp.autocast("cuda", enabled=False):
            u32, w32 = u.float(), w.float()
            cos = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            score = ((1.0 - cos) / 2.0) * self.c_max
        return score, u, theta, None


class FOROHHead(nn.Module):
    """Original FOROH: projector -> normalize -> acos(u.w)/pi * Cmax."""

    def __init__(self, in_dim: int, proj_dim: int = 128, c_max: int = 3):
        super().__init__()
        self.c_max = c_max
        self.projector = _Projector(in_dim, proj_dim)
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))

    def _get_w(self):
        return self.w_raw / (self.w_raw.norm() + 1e-8)

    def forward(self, z):
        u = F.normalize(self.projector(z), dim=-1)
        w = self._get_w()
        with torch.amp.autocast("cuda", enabled=False):
            u32, w32 = u.float(), w.float()
            cos = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            score = (theta / math.pi) * self.c_max
        return score, u, theta, None


class OrdinalModel(nn.Module):
    FEAT_DIMS = {"resnet50": 2048}

    def __init__(self, backbone_name: str, head: nn.Module, pretrained: bool = True):
        super().__init__()
        if backbone_name != "resnet50":
            raise ValueError("E01 intentionally supports only the original ResNet50 gate")
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.backbone.fc = nn.Identity()
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(variant: str, proj_dim: int = 128, c_max: int = 3):
    feat_dim = OrdinalModel.FEAT_DIMS["resnet50"]
    heads = {
        "euclidean_huber": EuclideanHuberHead,
        "normalized_cosine": NormalizedCosineHead,
        "foroh": FOROHHead,
    }
    if variant not in heads:
        raise ValueError(f"Unknown E01 variant: {variant}")
    head = heads[variant](feat_dim, proj_dim, c_max)
    return OrdinalModel("resnet50", head), head
