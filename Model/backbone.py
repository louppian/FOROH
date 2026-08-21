"""Backbone builder for ordinal regression models."""

import torch.nn as nn
from torchvision import models

FEAT_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "inception_v3": 2048,
    "densenet161": 2208,
    "efficientnet_b3": 1536,
}


def build_backbone(name, pretrained=True):
    if name == "resnet50":
        bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        bb.fc = nn.Identity()
    elif name == "resnet18":
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        bb.fc = nn.Identity()
    elif name == "inception_v3":
        bb = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT if pretrained else None)
        bb.fc = nn.Identity()
        bb.aux_logits = False
        bb.AuxLogits = None
    elif name == "densenet161":
        bb = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT if pretrained else None)
        bb.classifier = nn.Identity()
    elif name == "efficientnet_b3":
        bb = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        bb.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return bb


def freeze_backbone(backbone, name, n_layers):
    """Freeze first n_layers groups. Returns (frozen_count, total_count)."""
    if n_layers <= 0:
        return 0, sum(p.numel() for p in backbone.parameters())

    if "resnet" in name:
        groups = [["conv1", "bn1"], ["layer1"], ["layer2"], ["layer3"], ["layer4"]]
        targets = [t for g in groups[:n_layers] for t in g]
    elif name == "inception_v3":
        pool = ["Conv2d_1a", "Conv2d_2a", "Conv2d_2b", "Conv2d_3b",
                "Conv2d_4a", "Mixed_5b", "Mixed_5c", "Mixed_5d"]
        targets = pool[:min(n_layers * 2, len(pool))]
    elif "densenet" in name:
        pool = ["features.conv0", "features.norm0",
                "features.denseblock1", "features.transition1",
                "features.denseblock2", "features.transition2",
                "features.denseblock3", "features.transition3"]
        targets = pool[:min(n_layers * 2, len(pool))]
    elif "efficientnet" in name:
        targets = [f"features.{i}" for i in range(min(n_layers * 2, 9))]
    else:
        targets = []

    frozen = 0
    for pname, p in backbone.named_parameters():
        if any(pname.startswith(t) for t in targets):
            p.requires_grad = False
            frozen += p.numel()

    total = sum(p.numel() for p in backbone.parameters())
    return frozen, total
