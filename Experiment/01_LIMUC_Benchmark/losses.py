import torch
import torch.nn as nn
import torch.nn.functional as F

from Dataset.dataset import NUM_CLASSES


CDW_ALPHA = 5.0


def _with_aux(outputs, targets, loss_fn):
    loss = loss_fn(outputs["logits"], targets)
    aux_logits = outputs.get("aux_logits")
    if aux_logits is not None:
        loss = loss + 0.4 * loss_fn(aux_logits, targets)
    return loss


class CELoss(nn.Module):
    def forward(self, outputs, targets):
        return _with_aux(outputs, targets, F.cross_entropy)


class CDWCELoss(nn.Module):
    def __init__(self, alpha=CDW_ALPHA):
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, targets):
        return _with_aux(outputs, targets, self._cdw_ce)

    def _cdw_ce(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        class_idx = torch.arange(NUM_CLASSES, device=logits.device, dtype=torch.float32)
        dist = (class_idx.unsqueeze(0) - targets.float().unsqueeze(1)).abs().pow(self.alpha)
        log_complement = torch.log(torch.clamp(1.0 - probs, min=1e-7))
        return -(dist * log_complement).sum(dim=1).mean()


class WeightedCELoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        counts = torch.bincount(targets, minlength=self.num_classes).float().clamp(min=1.0)
        weight = (1.0 / counts) / (1.0 / counts).sum() * self.num_classes
        return _with_aux(outputs, targets, lambda logits, t: F.cross_entropy(logits, t, weight=weight.to(logits.device)))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def _focal(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

    def forward(self, outputs, targets):
        return _with_aux(outputs, targets, self._focal)


import math


class FOROHLoss(nn.Module):
    def forward(self, outputs, targets):
        theta = outputs["theta"]
        theta_y = math.pi * targets.float() / (NUM_CLASSES - 1)
        eps = (theta - theta_y) / math.pi
        return (eps ** 2).mean()


def build_loss(name):
    if name == "ce":
        return CELoss()
    if name == "weighted_ce":
        return WeightedCELoss()
    if name == "focal":
        return FocalLoss()
    if name == "cdw_ce":
        return CDWCELoss()
    if name == "foroh":
        return FOROHLoss()
    raise ValueError(f"Unknown loss: {name}")
