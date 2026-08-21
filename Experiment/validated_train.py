"""Validated experiment entrypoint for current FOROH studies.

Use this file for new experiments. The root ``3_train.py`` and
``Experiment/train.py`` are preserved for historical checkpoint reproduction.

This entrypoint adds:
- matched Euclidean Huber regression
- normalized cosine regression
- FOROH angular regression
- hyperspherical point-prototype control
- standard threshold decoding for CORAL/CORN
- unclipped scalar MSE training
- explicit LIMUC patient-map audit and split seed

Historical boundary optimization is intentionally unavailable here because the
old pipeline selected default vs optimized rules after observing test QWK.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error, recall_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "3_train.py"
spec = importlib.util.spec_from_file_location("foroh_base_train", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {BASE}")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
BASE_LOSS = T.compute_loss


def projector(in_dim: int, proj_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, proj_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(proj_dim, proj_dim),
    )


def huber_per_sample(pred, target, delta=0.5):
    diff = pred - target
    ad = diff.abs()
    return torch.where(ad <= delta, 0.5 * diff.square() / delta, ad - 0.5 * delta)


class MatchedScalarHead(nn.Module):
    """FOROH-matched projector followed by an unconstrained scalar readout."""

    def __init__(self, in_dim, proj_dim=128, c_max=3):
        super().__init__()
        self.c_max = c_max
        self.projector = projector(in_dim, proj_dim)
        # bias=False gives the readout the same number of direction parameters as w.
        self.readout = nn.Linear(proj_dim, 1, bias=False)

    def forward(self, z):
        return self.readout(self.projector(z)).squeeze(-1)


class RawMSEHead(nn.Module):
    """Standard scalar MSE head without train-time clamp."""

    def __init__(self, in_dim, c_max=3):
        super().__init__()
        self.c_max = c_max
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)


class CosineHead(nn.Module):
    """Normalized projector + learnable axis + cosine score."""

    def __init__(self, in_dim, proj_dim=128, c_max=3):
        super().__init__()
        self.c_max = c_max
        self.projector = projector(in_dim, proj_dim)
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))

    def forward(self, z):
        u = F.normalize(self.projector(z), dim=-1)
        w = self.w_raw / (self.w_raw.norm() + 1e-8)
        cos = torch.clamp(u.float() @ w.float(), -1 + 1e-7, 1 - 1e-7)
        return ((1.0 - cos) / 2.0) * self.c_max


class FOROHHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128, c_max=3):
        super().__init__()
        self.c_max = c_max
        self.projector = projector(in_dim, proj_dim)
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))

    def forward(self, z):
        u = F.normalize(self.projector(z), dim=-1)
        w = self.w_raw / (self.w_raw.norm() + 1e-8)
        with torch.amp.autocast("cuda", enabled=False):
            cos = torch.clamp(u.float() @ w.float(), -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            score = (theta / math.pi) * self.c_max
        return score, u, theta, None


class PointPrototypeHead(nn.Module):
    """Point control: same polar targets as FOROH, but azimuth is fixed too."""

    def __init__(self, in_dim, proj_dim=128, c_max=3):
        super().__init__()
        self.c_max = c_max
        self.projector = projector(in_dim, proj_dim)
        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))
        self.register_buffer("q_ref", F.normalize(torch.randn(proj_dim), dim=0))

    def frame(self):
        w = self.w_raw / (self.w_raw.norm() + 1e-8)
        q = self.q_ref - torch.dot(self.q_ref, w) * w
        if q.norm() < 1e-6:
            basis = torch.zeros_like(w)
            basis[torch.argmin(w.abs())] = 1.0
            q = basis - torch.dot(basis, w) * w
        q = q / (q.norm() + 1e-8)
        return w, q

    def forward(self, z):
        u = F.normalize(self.projector(z), dim=-1)
        w, q = self.frame()
        with torch.amp.autocast("cuda", enabled=False):
            u32, w32, q32 = u.float(), w.float(), q.float()
            cos = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            score = (theta / math.pi) * self.c_max
        return score, u, theta, w32, q32


def compute_loss(method, output, labels, head, class_weights=None, aux_lambda=0.0):
    y = labels.float()
    if method in ("scalar_huber", "cosine"):
        loss = huber_per_sample(output, y).mean()
        return loss, {"huber": loss.item()}

    if method == "point_proto":
        _score, u, _theta, w, q = output
        target_theta = (math.pi / head.c_max) * y
        proto = torch.cos(target_theta).unsqueeze(1) * w.unsqueeze(0)
        proto = proto + torch.sin(target_theta).unsqueeze(1) * q.unsqueeze(0)
        proto = F.normalize(proto, dim=-1)
        dot = torch.clamp((u.float() * proto).sum(-1), -1 + 1e-7, 1 - 1e-7)
        geodesic = torch.acos(dot)
        # Convert geodesic error to grade units so delta=0.5 matches FOROH scale.
        point_error = (geodesic / math.pi) * head.c_max
        loss = huber_per_sample(point_error, torch.zeros_like(point_error)).mean()
        return loss, {"prototype_huber": loss.item()}

    if method == "FOROH":
        score, _u, _theta, _aux = output
        loss = huber_per_sample(score, y).mean()
        return loss, {"huber": loss.item()}

    return BASE_LOSS(method, output, labels, head, class_weights, aux_lambda)


def build_model(args):
    feat_dim = T.OrdinalModel.FEAT_DIMS[args.backbone]
    c_max = T.CMAX[args.dataset]
    C = c_max + 1
    if args.method == "FOROH":
        head = FOROHHead(feat_dim, args.proj_dim, c_max)
    elif args.method == "scalar_huber":
        head = MatchedScalarHead(feat_dim, args.proj_dim, c_max)
    elif args.method == "cosine":
        head = CosineHead(feat_dim, args.proj_dim, c_max)
    elif args.method == "point_proto":
        head = PointPrototypeHead(feat_dim, args.proj_dim, c_max)
    elif args.method == "ce":
        head = T.CEHead(feat_dim, C)
    elif args.method == "coral":
        head = T.CORALHead(feat_dim, C)
    elif args.method == "corn":
        head = T.CORNHead(feat_dim, C)
    elif args.method == "mse":
        head = RawMSEHead(feat_dim, c_max)
    else:
        raise ValueError(args.method)
    return T.OrdinalModel(args.backbone, head), head, c_max


def evaluate(model, loader, method, head, c_max, device, theta_boundaries=None):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for imgs, y in loader:
            imgs = imgs.to(device)
            if method == "FOROH":
                score, _, _, _ = model(imgs)
                preds = score.round().clamp(0, c_max).cpu()
            elif method in ("scalar_huber", "cosine", "mse"):
                preds = head(model.backbone(imgs)).round().clamp(0, c_max).cpu()
            elif method == "point_proto":
                score, *_ = head(model.backbone(imgs))
                preds = score.round().clamp(0, c_max).cpu()
            elif method == "ce":
                preds = model(imgs).argmax(-1).cpu().float()
            elif method == "coral":
                logits = head(model.backbone(imgs))
                preds = (torch.sigmoid(logits) > 0.5).sum(-1).clamp(0, c_max).cpu().float()
            elif method == "corn":
                logits = head(model.backbone(imgs))
                cumulative = torch.cumprod(torch.sigmoid(logits), dim=-1)
                preds = (cumulative > 0.5).sum(-1).clamp(0, c_max).cpu().float()
            else:
                raise ValueError(method)
            preds_all.extend(preds.numpy())
            labels_all.extend(y.numpy())

    P = np.asarray(preds_all).astype(int)
    Y = np.asarray(labels_all).astype(int)
    metrics = {
        "mae": mean_absolute_error(Y, P),
        "qwk": cohen_kappa_score(Y, P, weights="quadratic"),
        "acc": accuracy_score(Y, P),
        "macro_f1": f1_score(Y, P, average="macro", zero_division=0),
        "off_by_1": float(np.mean(np.abs(P - Y) <= 1)),
    }
    rec = recall_score(Y, P, average=None, labels=list(range(c_max + 1)), zero_division=0)
    for i, r in enumerate(rec):
        metrics[f"recall_g{i}"] = r
    return metrics


def audit_limuc_patient_map(root: Path):
    base = root / "train_and_validation_sets"
    pdir = root / "patient_based_classified_images"
    if not base.exists() or not pdir.exists():
        raise FileNotFoundError(f"Missing LIMUC split folders under {root}")

    fname_to_pids = defaultdict(set)
    patient_count = 0
    for folder in pdir.iterdir():
        if not folder.is_dir():
            continue
        patient_count += 1
        for img in folder.rglob("*"):
            if T.LIMUCDataset._is_image(img):
                fname_to_pids[img.name].add(folder.name)

    names = []
    for grade in range(4):
        gdir = T.LIMUCDataset._grade_dir(base, grade)
        if gdir and gdir.exists():
            names.extend(p.name for p in gdir.iterdir() if T.LIMUCDataset._is_image(p))

    unmapped = sorted(n for n in names if n not in fname_to_pids)
    ambiguous = sorted(n for n, pids in fname_to_pids.items() if len(pids) != 1)
    report = {
        "train_images": len(names),
        "mapped_images": len(names) - len(unmapped),
        "unmapped_images": len(unmapped),
        "ambiguous_filenames": len(ambiguous),
        "patient_folders": patient_count,
    }
    print(f"  LIMUC patient-map audit: {report}")
    if unmapped:
        print(f"  first unmapped: {unmapped[:10]}")
    if ambiguous:
        print(f"  first ambiguous: {ambiguous[:10]}")
    return report


def build_datasets(args, fold=0):
    root = ROOT / "data"
    tr_tf = T.get_transforms("train", args.img_size, args.augment_preset)
    va_tf = T.get_transforms("val", args.img_size, args.augment_preset)
    if args.dataset == "limuc":
        kw = dict(fold=fold, n_folds=args.n_folds, seed=args.split_seed)
        tr = T.LIMUCDataset(root / "limuc", "train", tr_tf, **kw)
        va = T.LIMUCDataset(root / "limuc", "val", va_tf, **kw)
        te = T.LIMUCDataset(root / "limuc", "test", va_tf)
    elif args.dataset == "aptos":
        kw = dict(fold=fold, n_folds=args.n_folds, seed=args.split_seed,
                  preprocess_dixit=(args.augment_preset == "dixit"))
        tr = T.APTOSDataset(root / "aptos2019", "train", tr_tf, **kw)
        va = T.APTOSDataset(root / "aptos2019", "val", va_tf, **kw)
        te = va
    elif args.dataset == "retinamnist":
        tr = T.RetinaMNISTDataset(root / "retinamnist", "train", tr_tf, 224)
        va = T.RetinaMNISTDataset(root / "retinamnist", "val", va_tf, 224)
        te = T.RetinaMNISTDataset(root / "retinamnist", "test", va_tf, 224)
    elif args.dataset == "kneexray":
        tr = T.KneeXrayDataset(root / "kneexray", "train", tr_tf)
        va = T.KneeXrayDataset(root / "kneexray", "val", va_tf)
        te = T.KneeXrayDataset(root / "kneexray", "test", va_tf)
    else:
        raise ValueError(args.dataset)
    return tr, va, te


def parse_args():
    p = argparse.ArgumentParser(description="FOROH validated experiments")
    p.add_argument("--method", default="FOROH",
                   choices=["FOROH", "scalar_huber", "cosine", "point_proto", "ce", "coral", "corn", "mse"])
    p.add_argument("--dataset", default="limuc", choices=["limuc", "retinamnist", "aptos", "kneexray"])
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "resnet18", "inception_v3", "densenet161", "efficientnet_b3"])
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
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
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42, help="model/training RNG seed")
    p.add_argument("--split-seed", type=int, default=42, help="data split RNG seed")
    p.add_argument("--allow-unmapped-patients", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="Result")

    # Attributes expected by the historical main loop.
    p.set_defaults(optimize_boundaries=False, aux_ce=False, aux_lambda=0.0,
                   freq_weight=False, boundary_grid=200)
    args = p.parse_args()
    if args.backbone == "inception_v3" and args.img_size == 224:
        args.img_size = 299
    if args.backbone == "efficientnet_b3" and args.img_size == 224:
        args.img_size = 256

    if args.dataset == "limuc":
        report = audit_limuc_patient_map(ROOT / "data" / "limuc")
        if not args.allow_unmapped_patients and (report["unmapped_images"] or report["ambiguous_filenames"]):
            p.error("LIMUC patient mapping is incomplete/ambiguous; refusing patient-level CV. Fix mapping first. Use --allow-unmapped-patients only for debugging.")
    return args


T.compute_loss = compute_loss
T.build_model = build_model
T.build_datasets = build_datasets
T.evaluate = evaluate
T.parse_args = parse_args

if __name__ == "__main__":
    T.main()
