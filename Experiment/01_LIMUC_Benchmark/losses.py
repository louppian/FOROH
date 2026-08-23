import torch
import torch.nn as nn
import torch.nn.functional as F

from Dataset.dataset import NUM_CLASSES


CDW_ALPHA = 5.0
HUBER_DELTA = 0.5


class CELoss(nn.Module):
    def forward(self, outputs, targets):
        return F.cross_entropy(outputs["logits"], targets)


class CDWCELoss(nn.Module):
    def __init__(self, alpha=CDW_ALPHA):
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, targets):
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=1)
        class_idx = torch.arange(NUM_CLASSES, device=logits.device, dtype=torch.float32)
        dist = (class_idx.unsqueeze(0) - targets.float().unsqueeze(1)).abs().pow(self.alpha)
        log_complement = torch.log(torch.clamp(1.0 - probs, min=1e-7))
        return -(dist * log_complement).sum(dim=1).mean()


class FOROHLoss(nn.Module):
    def forward(self, outputs, targets):
        diff = outputs["score"] - targets.float()
        abs_diff = diff.abs()
        loss = torch.where(abs_diff <= HUBER_DELTA, 0.5 * diff.square() / HUBER_DELTA, abs_diff - 0.5 * HUBER_DELTA)
        return loss.mean()


def build_loss(name):
    if name == "ce":
        return CELoss()
    if name == "cdw_ce":
        return CDWCELoss()
    if name == "foroh":
        return FOROHLoss()
    raise ValueError(f"Unknown loss: {name}")
