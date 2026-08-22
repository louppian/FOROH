"""E02 model definitions: point prototype versus FOROH level set."""

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


class FOROHHead(nn.Module):
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


class PointPrototypeHead(nn.Module):
    """Same hypersphere/projector capacity, but each grade is a fixed-meridian point."""

    def __init__(self, in_dim: int, proj_dim: int = 128, c_max: int = 3):
        super().__init__()
        self.c_max = c_max
        self.projector = _Projector(in_dim, proj_dim)
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))
        self.register_buffer("q_ref", F.normalize(torch.randn(proj_dim), dim=0))

    def _get_w(self):
        return self.w_raw / (self.w_raw.norm() + 1e-8)

    def _get_q(self, w):
        q = self.q_ref - torch.dot(self.q_ref, w) * w
        if q.norm() < 1e-6:
            basis = torch.zeros_like(w)
            basis[torch.argmin(w.abs())] = 1.0
            q = basis - torch.dot(basis, w) * w
        return F.normalize(q, dim=0)

    def get_prototypes(self):
        w = self._get_w()
        q = self._get_q(w)
        theta = torch.arange(
            self.c_max + 1, device=w.device, dtype=w.dtype
        ) * (math.pi / self.c_max)
        prototypes = (
            theta.cos().unsqueeze(1) * w.unsqueeze(0)
            + theta.sin().unsqueeze(1) * q.unsqueeze(0)
        )
        return F.normalize(prototypes, dim=-1)

    def forward(self, z):
        u = F.normalize(self.projector(z), dim=-1)
        w = self._get_w()
        with torch.amp.autocast("cuda", enabled=False):
            u32, w32 = u.float(), w.float()
            cos = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            score = (theta / math.pi) * self.c_max
            prototypes = self.get_prototypes().float()
        return score, u, theta, prototypes


class OrdinalModel(nn.Module):
    def __init__(self, head: nn.Module, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.backbone.fc = nn.Identity()
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(variant: str, proj_dim: int = 128, c_max: int = 3):
    feat_dim = 2048
    if variant == "point_prototype":
        head = PointPrototypeHead(feat_dim, proj_dim, c_max)
    elif variant == "foroh":
        head = FOROHHead(feat_dim, proj_dim, c_max)
    else:
        raise ValueError(f"Unknown E02 variant: {variant}")
    return OrdinalModel(head), head
