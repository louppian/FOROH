"""Phase-1 entrypoint built directly on the original ``3_train.py`` pipeline.

The original data loading, augmentation, optimizer, scheduler, early stopping,
metrics, checkpoint format, and CLI defaults are kept.  Only the model head,
loss, and point-prototype decoding are extended for the four Phase-1 controls.

Variants
--------
- ``euclidean_huber``: matched 2-layer projector + scalar Huber readout
- ``normalized_cosine``: normalized cosine score
- ``foroh``: original angular FOROH score
- ``point_prototype``: fixed-meridian spherical point prototypes

Examples
--------
python Experiment/train.py --phase1-variant euclidean_huber --dataset limuc
python Experiment/train.py --phase1-variant normalized_cosine --dataset limuc
python Experiment/train.py --phase1-variant foroh --dataset limuc
python Experiment/train.py --phase1-variant point_prototype --dataset limuc
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    f1_score,
    recall_score,
    cohen_kappa_score,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN = ROOT / "3_train.py"

spec = importlib.util.spec_from_file_location("foroh_original_train", BASE_TRAIN)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import original trainer: {BASE_TRAIN}")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

_BASE_COMPUTE_LOSS = T.compute_loss
_BASE_EVALUATE = T.evaluate
_BASE_PARSE_ARGS = T.parse_args


def _projector(in_dim: int, proj_dim: int) -> nn.Sequential:
    """Exactly the projector used by the original FOROH head."""
    return nn.Sequential(
        nn.Linear(in_dim, proj_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(proj_dim, proj_dim),
    )


class Phase1Head(nn.Module):
    """Minimal Phase-1 extension of the original FOROH head."""

    def __init__(self, in_dim: int, proj_dim: int, c_max: int, variant: str):
        super().__init__()
        self.c_max = c_max
        self.variant = variant
        self.projector = _projector(in_dim, proj_dim)

        if variant == "euclidean_huber":
            # bias=False matches the number of readout parameters to FOROH's w.
            self.readout = nn.Linear(proj_dim, 1, bias=False)
            self.w_raw = None
        else:
            self.readout = None
            self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))

        if variant == "point_prototype":
            # Fixed reference; no extra trainable direction beyond FOROH's w.
            self.register_buffer("q_ref", F.normalize(torch.randn(proj_dim), dim=0))

    def _get_w(self):
        return F.normalize(self.w_raw, dim=0)

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
        return (
            theta.cos().unsqueeze(1) * w.unsqueeze(0)
            + theta.sin().unsqueeze(1) * q.unsqueeze(0)
        )

    def forward(self, z):
        h = self.projector(z)

        if self.variant == "euclidean_huber":
            # Intentionally unclipped during training; original evaluation rounds/clips.
            score = self.readout(h).squeeze(-1)
            dummy_theta = torch.zeros_like(score)
            return score, h, dummy_theta, None

        u = F.normalize(h, dim=-1)
        w = self._get_w()
        with torch.amp.autocast("cuda", enabled=False):
            u32, w32 = u.float(), w.float()
            cos = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)

            if self.variant == "normalized_cosine":
                score = ((1.0 - cos) / 2.0) * self.c_max
                extra = None
            elif self.variant == "foroh":
                score = (theta / math.pi) * self.c_max
                extra = None
            elif self.variant == "point_prototype":
                score = (theta / math.pi) * self.c_max
                extra = self.get_prototypes()
            else:
                raise ValueError(self.variant)

        return score, u, theta, extra


def _huber_per(error, delta=0.5):
    ad = error.abs()
    return torch.where(
        ad <= delta,
        0.5 * error.square() / delta,
        ad - 0.5 * delta,
    )


def compute_loss(method, output, labels, head, class_weights=None, aux_lambda=0.0):
    """Original loss dispatcher plus Phase-1 FOROH variants."""
    if method != "FOROH" or not isinstance(head, Phase1Head):
        return _BASE_COMPUTE_LOSS(method, output, labels, head, class_weights, aux_lambda)

    score, u, _theta, extra = output

    if head.variant == "point_prototype":
        # Pure point constraint: geodesic distance to the target grade prototype.
        prototypes = extra.float()
        target = prototypes[labels]
        cos = torch.clamp((u.float() * target).sum(-1), -1 + 1e-7, 1 - 1e-7)
        error = torch.acos(cos) * (head.c_max / math.pi)
    else:
        error = score - labels.float()

    per = _huber_per(error, delta=0.5)
    loss = (
        (per * class_weights[labels]).mean()
        if class_weights is not None
        else per.mean()
    )
    return loss, {head.variant: loss.item()}


def evaluate(model, loader, method, head, c_max, device, theta_boundaries=None):
    """Use original evaluation, except point prototype uses nearest prototype."""
    if method != "FOROH" or not isinstance(head, Phase1Head) or head.variant != "point_prototype":
        return _BASE_EVALUATE(
            model, loader, method, head, c_max, device,
            theta_boundaries=theta_boundaries,
        )

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for imgs, y in loader:
            _score, u, _theta, prototypes = model(imgs.to(device))
            sim = u.float() @ prototypes.float().T
            preds = sim.argmax(dim=-1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    P = np.asarray(all_preds).astype(int)
    Y = np.asarray(all_labels).astype(int)
    metrics = {
        "mae": mean_absolute_error(Y, P),
        "qwk": cohen_kappa_score(Y, P, weights="quadratic"),
        "acc": accuracy_score(Y, P),
        "macro_f1": f1_score(Y, P, average="macro", zero_division=0),
        "off_by_1": float(np.mean(np.abs(P - Y) <= 1)),
    }
    recalls = recall_score(
        Y, P, average=None, labels=list(range(c_max + 1)), zero_division=0
    )
    for i, r in enumerate(recalls):
        metrics[f"recall_g{i}"] = r
    return metrics


def build_model(args):
    feat_dim = T.OrdinalModel.FEAT_DIMS[args.backbone]
    c_max = T.CMAX[args.dataset]
    num_classes = c_max + 1

    if args.method == "FOROH":
        head = Phase1Head(
            feat_dim,
            proj_dim=args.proj_dim,
            c_max=c_max,
            variant=args.phase1_variant,
        )
    elif args.method == "ce":
        head = T.CEHead(feat_dim, num_classes)
    elif args.method == "coral":
        head = T.CORALHead(feat_dim, num_classes)
    elif args.method == "corn":
        head = T.CORNHead(feat_dim, num_classes)
    elif args.method == "mse":
        head = T.MSEHead(feat_dim, c_max)
    else:
        raise ValueError(args.method)

    return T.OrdinalModel(args.backbone, head), head, c_max


def parse_args():
    """Original CLI plus one Phase-1 selector; original defaults are preserved."""
    p = argparse.ArgumentParser(description="FOROH Phase-1 on original trainer")

    p.add_argument("--method", default="FOROH", choices=["FOROH", "ce", "coral", "corn", "mse"])
    p.add_argument(
        "--phase1-variant",
        default="foroh",
        choices=["euclidean_huber", "normalized_cosine", "foroh", "point_prototype"],
    )
    p.add_argument("--dataset", default="limuc", choices=["limuc", "retinamnist", "aptos", "kneexray"])
    p.add_argument(
        "--backbone", default="resnet50",
        choices=["resnet50", "resnet18", "inception_v3", "densenet161", "efficientnet_b3"],
    )

    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--optimize-boundaries", action="store_true")
    p.add_argument("--aux-ce", action="store_true")
    p.add_argument("--aux-lambda", type=float, default=0.3)
    p.add_argument("--freq-weight", action="store_true")
    p.add_argument("--boundary-grid", type=int, default=200)

    # These are the ORIGINAL 3_train.py defaults.
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--freeze-layers", type=int, default=2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "reduce_lr"])
    p.add_argument("--scheduler-patience", type=int, default=10)
    p.add_argument("--oversample", action="store_true")
    p.add_argument("--augment-preset", default="default", choices=["default", "dixit"])
    p.add_argument("--exp", type=int, default=0)

    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="outputs")

    args = p.parse_args()
    if args.backbone == "inception_v3" and args.img_size == 224:
        args.img_size = 299
    if args.backbone == "efficientnet_b3" and args.img_size == 224:
        args.img_size = 256

    # Phase-1 variants are all routed through the original FOROH training path.
    if args.method == "FOROH" and args.phase1_variant == "point_prototype" and args.optimize_boundaries:
        raise ValueError("Boundary optimization is not defined for nearest-prototype decoding")
    return args


# Patch only extension points; T.main remains the original training loop.
T.compute_loss = compute_loss
T.evaluate = evaluate
T.build_model = build_model
T.parse_args = parse_args


if __name__ == "__main__":
    T.main()
