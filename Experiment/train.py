"""Paper-reproduction entrypoint for FOROH.

This wraps the original ``3_train.py`` training pipeline while restoring the
ablation switches required by the FOROH draft:

- arccos score vs cosine regression
- Huber / L1 / SmoothL1 / MSE loss on the FOROH score
- learnable vs fixed random severity axis
- projector on/off

The paper defaults are used here (batch 64, AdamW, backbone/head LR
1e-4/1e-3, cosine scheduler, freeze_layers=2). Existing root scripts remain
untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN = ROOT / "3_train.py"

spec = importlib.util.spec_from_file_location("foroh_base_train", BASE_TRAIN)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {BASE_TRAIN}")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

_BASE_COMPUTE_LOSS = T.compute_loss


class FOROHHead(nn.Module):
    """FOROH head with the ablation switches used in the paper draft."""

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 128,
        c_max: int = 3,
        *,
        use_arccos: bool = True,
        fixed_w: bool = False,
        no_projector: bool = False,
        aux_ce: bool = False,
    ) -> None:
        super().__init__()
        self.c_max = c_max
        self.use_arccos = use_arccos
        self.no_projector = no_projector
        self.proj_dim = in_dim if no_projector else proj_dim

        if no_projector:
            self.projector = None
        else:
            self.projector = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(proj_dim, proj_dim),
            )

        w_init = F.normalize(torch.randn(self.proj_dim), dim=0)
        if fixed_w:
            self.register_buffer("w_raw", w_init)
        else:
            self.w_raw = nn.Parameter(w_init)

        self.aux_classifier = (
            nn.Linear(self.proj_dim, c_max + 1) if aux_ce else None
        )
        self.loss_fn = "huber"

    def _get_w(self):
        return self.w_raw / (self.w_raw.norm() + 1e-8)

    def forward(self, z):
        u = z if self.projector is None else self.projector(z)
        u = F.normalize(u, dim=-1)
        w = self._get_w()

        # Keep angular operations in float32: acos near +/-1 is unstable in fp16.
        with torch.amp.autocast("cuda", enabled=False):
            u32, w32 = u.float(), w.float()
            cos = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            if self.use_arccos:
                score = (theta / math.pi) * self.c_max
            else:
                # Cosine-regression ablation used in the draft.
                score = ((1.0 - cos) / 2.0) * self.c_max

        aux_logits = (
            self.aux_classifier(u32)
            if self.aux_classifier is not None and self.training
            else None
        )
        return score, u, theta, aux_logits


def compute_loss(method, output, labels, head, class_weights=None, aux_lambda=0.0):
    """Loss dispatcher with FOROH loss-function ablations."""
    if method != "FOROH":
        return _BASE_COMPUTE_LOSS(
            method, output, labels, head, class_weights, aux_lambda
        )

    score, _u, _theta, aux_logits = output
    y = labels.float()
    loss_fn = getattr(head, "loss_fn", "huber")

    if loss_fn == "huber":
        delta = 0.5
        diff = score - y
        abs_diff = diff.abs()
        per = torch.where(
            abs_diff <= delta,
            0.5 * diff.square() / delta,
            abs_diff - 0.5 * delta,
        )
    elif loss_fn == "l1":
        per = (score - y).abs()
    elif loss_fn == "smooth_l1":
        per = F.smooth_l1_loss(score, y, reduction="none")
    elif loss_fn == "mse":
        per = (score - y).square()
    else:
        raise ValueError(f"Unknown FOROH loss: {loss_fn}")

    loss = (
        (per * class_weights[labels]).mean()
        if class_weights is not None
        else per.mean()
    )
    details = {loss_fn: loss.item()}

    if aux_logits is not None and aux_lambda > 0:
        aux = F.cross_entropy(aux_logits, labels)
        total = loss + aux_lambda * aux
        details.update(aux_ce=aux.item(), total=total.item())
        return total, details
    return loss, details


def build_model(args):
    feat_dim = T.OrdinalModel.FEAT_DIMS[args.backbone]
    c_max = T.CMAX[args.dataset]
    num_classes = c_max + 1

    if args.method == "FOROH":
        head = FOROHHead(
            feat_dim,
            args.proj_dim,
            c_max,
            use_arccos=not args.no_arccos,
            fixed_w=args.fixed_w,
            no_projector=args.no_projector,
            aux_ce=args.aux_ce,
        )
        head.loss_fn = args.loss_fn
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
    p = argparse.ArgumentParser(description="FOROH paper reproduction")

    p.add_argument(
        "--method",
        default="FOROH",
        choices=["FOROH", "ce", "coral", "corn", "mse"],
    )
    p.add_argument(
        "--dataset",
        default="limuc",
        choices=["limuc", "retinamnist", "aptos", "kneexray"],
    )
    p.add_argument(
        "--backbone",
        default="resnet50",
        choices=[
            "resnet50",
            "resnet18",
            "inception_v3",
            "densenet161",
            "efficientnet_b3",
        ],
    )

    # FOROH architecture / paper ablations.
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument(
        "--loss-fn",
        default="huber",
        choices=["huber", "l1", "smooth_l1", "mse"],
    )
    p.add_argument(
        "--no-arccos",
        action="store_true",
        help="Use ((1-cos)/2)*C_max instead of acos(cos)/pi*C_max",
    )
    p.add_argument(
        "--fixed-w",
        action="store_true",
        help="Keep the randomly initialized severity axis fixed",
    )
    p.add_argument(
        "--no-projector",
        action="store_true",
        help="L2-normalize backbone features directly",
    )

    # Existing optional variants kept for compatibility.
    p.add_argument("--optimize-boundaries", action="store_true")
    p.add_argument("--aux-ce", action="store_true")
    p.add_argument("--aux-lambda", type=float, default=0.3)
    p.add_argument("--freq-weight", action="store_true")
    p.add_argument("--boundary-grid", type=int, default=200)

    # Paper setup defaults (Section 3.1).
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4, help="Backbone LR")
    p.add_argument("--lr-head", type=float, default=1e-3, help="Head LR")
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
    p.add_argument("--fold", type=int, default=0, help="Fold index, -1 for all folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="Result")

    args = p.parse_args()
    if args.backbone == "inception_v3" and args.img_size == 224:
        args.img_size = 299
    if args.backbone == "efficientnet_b3" and args.img_size == 224:
        args.img_size = 256
    return args


# Patch only the pieces needed by the root training loop.
T.FOROHHead = FOROHHead
T.compute_loss = compute_loss
T.build_model = build_model
T.parse_args = parse_args


if __name__ == "__main__":
    T.main()
