"""FOROH Model package."""

import torch.nn as nn

from .backbone import FEAT_DIMS, build_backbone, freeze_backbone
from .heads import (
    HEAD_REGISTRY,
    EuclideanHuberHead,
    FOROHHead,
    NormalizedCosineHead,
    PointPrototypeHead,
)


class OrdinalModel(nn.Module):
    """Backbone + Head."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(method, backbone_name, c_max, proj_dim=128, dropout=0.3, pretrained=True):
    backbone = build_backbone(backbone_name, pretrained)
    feat_dim = FEAT_DIMS[backbone_name]
    head_cls = HEAD_REGISTRY[method]
    head = head_cls(feat_dim, proj_dim=proj_dim, c_max=c_max, dropout=dropout)
    return OrdinalModel(backbone, head), head
