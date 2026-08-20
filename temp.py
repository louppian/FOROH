"""
FOROH + sin²θ regularization experiment
========================================
sin²θ = ||u_⊥||² regularization pushes u toward w-axis,
injecting rank-2 gradient (meridian + longitude) instead of rank-1.

L_total = L_FOROH(ŷ, y) + λ · mean(sin²θ)

Usage:
  python run_sin2_exp.py                   # all 3 datasets, fold 0
  python run_sin2_exp.py --dataset limuc   # single dataset
  python run_sin2_exp.py --sin2-lambda 0.1 # tune λ
"""

import os, sys, math, json, argparse, random, subprocess
from pathlib import Path
from collections import defaultdict, Counter
from itertools import product as iter_product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, f1_score,
    recall_score, cohen_kappa_score,
)


# ─── Heads ────────────────────────────────────────────────────────────────────

class FOROHHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128, c_max=3,
                 use_arccos=True, fixed_w=False, no_projector=False, aux_ce=False):
        super().__init__()
        self.c_max = c_max
        self.use_arccos = use_arccos
        self.no_projector = no_projector

        if no_projector:
            self.projector = None
            self.proj_dim = in_dim
        else:
            self.proj_dim = proj_dim
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

    def _get_w(self):
        return self.w_raw / (self.w_raw.norm() + 1e-8)

    def forward(self, z):
        u = self.projector(z) if self.projector is not None else z
        u = F.normalize(u, dim=-1)
        w = self._get_w()

        with torch.amp.autocast("cuda", enabled=False):
            u32, w32 = u.float(), w.float()
            cos = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            if self.use_arccos:
                theta = torch.acos(cos)
                score = (theta / math.pi) * self.c_max
            else:
                theta = torch.acos(cos)
                score = ((1.0 - cos) / 2.0) * self.c_max

            # sin²θ = 1 - cos²θ = ||u_⊥||²
            sin2_theta = 1.0 - cos ** 2

        aux_logits = None
        if self.aux_classifier is not None and self.training:
            aux_logits = self.aux_classifier(u32)

        return score, u, theta, aux_logits, sin2_theta


class CEHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, z):
        return self.fc(z)


class CORALHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))
        self.num_classes = num_classes

    def forward(self, z):
        return self.fc(z) + self.biases.unsqueeze(0)


class CORNHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.classifiers = nn.ModuleList(
            [nn.Linear(in_dim, 1) for _ in range(num_classes - 1)]
        )

    def forward(self, z):
        return torch.cat([clf(z) for clf in self.classifiers], dim=-1)


class MSEHead(nn.Module):
    def __init__(self, in_dim, c_max):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.c_max = c_max

    def forward(self, z):
        return self.fc(z).squeeze(-1).clamp(0, self.c_max)


# ─── Model ────────────────────────────────────────────────────────────────────

class OrdinalModel(nn.Module):
    FEAT_DIMS = {
        "resnet18": 512, "resnet50": 2048,
        "inception_v3": 2048, "densenet161": 2208,
        "efficientnet_b3": 1536,
    }

    def __init__(self, backbone_name, head, pretrained=True):
        super().__init__()
        bb = self._build_backbone(backbone_name, pretrained)
        self.backbone = bb
        self.head = head

    @staticmethod
    def _build_backbone(name, pretrained):
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

    def forward(self, x):
        return self.head(self.backbone(x))


# ─── Datasets ─────────────────────────────────────────────────────────────────

class LIMUCDataset(Dataset):
    C_MAX = 3

    def __init__(self, root, split="train", transform=None, fold=None, n_folds=5, seed=42):
        self.root, self.transform, self.c_max = Path(root), transform, self.C_MAX

        if split == "test":
            self.samples = self._load_dir(self.root / "test_set", range(4))
        else:
            base = self.root / "train_and_validation_sets"
            patient_dir = self.root / "patient_based_classified_images"
            fname_to_pid = self._build_pid_map(patient_dir)

            all_samples = []
            for grade in range(4):
                gdir = self._grade_dir(base, grade)
                if gdir and gdir.exists():
                    for p in sorted(gdir.iterdir()):
                        if self._is_image(p):
                            pid = fname_to_pid.get(p.name, f"unk_{p.stem}")
                            all_samples.append((str(p), grade, pid))

            unique_pids = sorted({s[2] for s in all_samples})
            pid_labels = defaultdict(list)
            for _, g, pid in all_samples:
                pid_labels[pid].append(g)
            strat = [Counter(pid_labels[p]).most_common(1)[0][0] for p in unique_pids]

            if fold is not None:
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                tr, va = list(skf.split(unique_pids, strat))[fold]
                idx = tr if split == "train" else va
                selected = {unique_pids[i] for i in idx}
            else:
                selected = set(unique_pids)

            self.samples = [(p, g) for p, g, pid in all_samples if pid in selected]
            print(f"    [{split}] {len(self.samples)} images, {len(selected)} patients")

    @staticmethod
    def _is_image(p):
        return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    @staticmethod
    def _grade_dir(base, grade):
        for fmt in [f"Mayo {grade}", f"Mayo{grade}"]:
            d = base / fmt
            if d.exists():
                return d
        return None

    @staticmethod
    def _build_pid_map(patient_dir):
        m = {}
        if patient_dir.exists():
            for folder in patient_dir.iterdir():
                if folder.is_dir():
                    for img in folder.rglob("*"):
                        if LIMUCDataset._is_image(img):
                            m[img.name] = folder.name
        return m

    def _load_dir(self, base, grades):
        samples = []
        for g in grades:
            gdir = self._grade_dir(base, g)
            if gdir and gdir.exists():
                for p in sorted(gdir.iterdir()):
                    if self._is_image(p):
                        samples.append((str(p), g))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img, label


class APTOSDataset(Dataset):
    C_MAX = 4

    def __init__(self, root, split="train", transform=None, fold=None, n_folds=5, seed=42,
                 preprocess_dixit=False):
        self.root, self.transform, self.c_max = Path(root), transform, self.C_MAX
        self.preprocess_dixit = preprocess_dixit
        import csv
        with open(self.root / "train.csv") as f:
            reader = csv.reader(f)
            next(reader)
            rows = [(r[0], int(r[1])) for r in reader]

        if fold is not None and split in ("train", "val"):
            ids, labels = zip(*rows)
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            tr, va = list(skf.split(ids, labels))[fold]
            rows = [rows[i] for i in (tr if split == "train" else va)]

        img_dir = self.root / "train_images"
        self.samples = [(str(img_dir / f"{c}.png"), l)
                        for c, l in rows if (img_dir / f"{c}.png").exists()]
        print(f"    [{split}] {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.preprocess_dixit:
            img = self._dixit_preprocess(img)
        return self.transform(img) if self.transform else img, label

    @staticmethod
    def _dixit_preprocess(img):
        import cv2
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        mask = gray > 7
        coords = np.argwhere(mask)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            arr = arr[y0:y1, x0:x1]
        blur = cv2.GaussianBlur(arr, (0, 0), 10)
        arr = cv2.addWeighted(arr, 4, blur, -4, 128)
        return Image.fromarray(arr)


class KneeXrayDataset(Dataset):
    C_MAX = 4

    def __init__(self, root, split="train", transform=None, **kwargs):
        self.root, self.transform, self.c_max = Path(root), transform, self.C_MAX
        base = self.root / "KneeXrayData" / "ClsKLData" / "kneeKL224" / split
        self.samples = []
        for g in range(5):
            gdir = base / str(g)
            if gdir.exists():
                for p in sorted(gdir.iterdir()):
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        self.samples.append((str(p), g))
        print(f"    [{split}] {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img, label


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(split, img_size=224, augment_preset="default"):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if split == "train":
        if augment_preset == "dixit":
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(20),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2),
                                        scale=(0.8, 1.2), shear=15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), norm,
            ])
        else:
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.ToTensor(), norm,
            ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), norm,
    ])


# ─── Loss (with sin²θ regularization) ────────────────────────────────────────

HUBER_DELTA = 0.5

def compute_loss(method, output, labels, head,
                 loss_fn="huber", class_weights=None, aux_lambda=0.0,
                 sin2_lambda=0.0):
    """
    FOROH loss with optional sin²θ regularization:
      L = L_FOROH + λ_sin2 · mean(sin²θ)

    sin²θ = ||u_⊥||² gradient is orthogonal to the primary meridian gradient,
    raising per-sample gradient rank from 1 to 2.
    """
    y = labels.float()

    if method == "FOROH":
        score, u, theta, aux_logits, sin2_theta = output

        if loss_fn == "huber":
            diff = score - y
            ad = diff.abs()
            per = torch.where(ad <= HUBER_DELTA,
                              0.5 * diff**2 / HUBER_DELTA,
                              ad - 0.5 * HUBER_DELTA)
        elif loss_fn == "l1":
            per = (score - y).abs()
        elif loss_fn == "smooth_l1":
            per = F.smooth_l1_loss(score, y, reduction="none")
        elif loss_fn == "mse":
            per = (score - y) ** 2
        else:
            raise ValueError(f"Unknown loss: {loss_fn}")

        loss = ((per * class_weights[labels]).mean()
                if class_weights is not None else per.mean())
        details = {"foroh": loss.item()}

        total = loss

        # ── sin²θ regularization ──
        if sin2_lambda > 0 and sin2_theta is not None:
            sin2_loss = sin2_theta.mean()
            total = total + sin2_lambda * sin2_loss
            details["sin2"] = sin2_loss.item()

        # ── Auxiliary CE ──
        if aux_logits is not None and aux_lambda > 0:
            aux = F.cross_entropy(aux_logits, labels)
            total = total + aux_lambda * aux
            details["aux_ce"] = aux.item()

        details["total"] = total.item()
        return total, details

    elif method == "ce":
        loss = F.cross_entropy(output, labels)
        return loss, {"ce": loss.item()}

    elif method == "coral":
        C = head.num_classes
        tgt = (labels.unsqueeze(1) >
               torch.arange(C - 1, device=labels.device)).float()
        loss = F.binary_cross_entropy_with_logits(output, tgt)
        return loss, {"coral": loss.item()}

    elif method == "corn":
        C = head.num_classes
        total, n = 0.0, 0
        for k in range(C - 1):
            mask = labels >= k
            if mask.sum() == 0:
                continue
            total += F.binary_cross_entropy_with_logits(
                output[mask, k], (labels[mask] > k).float())
            n += 1
        loss = total / max(n, 1)
        return loss, {"corn": loss.item()}

    elif method == "mse":
        loss = F.mse_loss(output, y)
        return loss, {"mse": loss.item()}

    raise ValueError(method)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, loader, method, head, c_max, device):
    model.eval()
    all_preds, all_labels = [], []
    sin2_vals = []

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for imgs, y in loader:
            imgs = imgs.to(device)
            if method == "FOROH":
                score, _, theta, _, sin2 = model(imgs)
                preds = score.round().clamp(0, c_max).cpu()
                sin2_vals.extend(sin2.cpu().numpy())
            elif method == "ce":
                preds = model(imgs).argmax(-1).cpu().float()
            elif method == "coral":
                z = model.backbone(imgs)
                preds = torch.sigmoid(head(z)).sum(-1).round().clamp(0, c_max).cpu()
            elif method == "corn":
                z = model.backbone(imgs)
                cp = torch.cumprod(torch.sigmoid(head(z)), dim=-1)
                preds = cp.sum(-1).round().clamp(0, c_max).cpu()
            elif method == "mse":
                z = model.backbone(imgs)
                preds = head(z).round().clamp(0, c_max).cpu()
            else:
                raise ValueError(method)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    P = np.array(all_preds).astype(int)
    Y = np.array(all_labels).astype(int)

    metrics = {
        "mae":      mean_absolute_error(Y, P),
        "qwk":      cohen_kappa_score(Y, P, weights="quadratic"),
        "acc":      accuracy_score(Y, P),
        "macro_f1": f1_score(Y, P, average="macro", zero_division=0),
        "off_by_1": float(np.mean(np.abs(P - Y) <= 1)),
    }
    for i, r in enumerate(recall_score(Y, P, average=None,
                                        labels=list(range(c_max + 1)),
                                        zero_division=0)):
        metrics[f"recall_g{i}"] = r
    if sin2_vals:
        metrics["mean_sin2"] = float(np.mean(sin2_vals))
    return metrics


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, method, head, device,
                    scaler, loss_fn, class_weights, aux_lambda, sin2_lambda):
    model.train()
    total, details_acc, n = 0.0, defaultdict(float), 0

    for imgs, y in loader:
        imgs, y = imgs.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            if method == "FOROH":
                out = model(imgs)
            elif method == "ce":
                out = model(imgs)
            elif method in ("coral", "corn"):
                out = head(model.backbone(imgs))
            elif method == "mse":
                out = head(model.backbone(imgs))
            loss, det = compute_loss(method, out, y, head, loss_fn,
                                     class_weights, aux_lambda, sin2_lambda)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += loss.item()
        for k, v in det.items():
            details_acc[k] += v
        n += 1

    return total / max(n, 1), {k: v / max(n, 1) for k, v in details_acc.items()}


# ─── Build Helpers ────────────────────────────────────────────────────────────

CMAX = {"limuc": 3, "aptos": 4, "kneexray": 4}

def build_model(args):
    feat_dim = OrdinalModel.FEAT_DIMS[args.backbone]
    c_max = CMAX[args.dataset]
    C = c_max + 1

    if args.method == "FOROH":
        head = FOROHHead(feat_dim, args.proj_dim, c_max,
                         use_arccos=True, fixed_w=False,
                         no_projector=False, aux_ce=False)
    elif args.method == "ce":
        head = CEHead(feat_dim, C)
    elif args.method == "coral":
        head = CORALHead(feat_dim, C)
    elif args.method == "corn":
        head = CORNHead(feat_dim, C)
    elif args.method == "mse":
        head = MSEHead(feat_dim, c_max)
    else:
        raise ValueError(args.method)
    return OrdinalModel(args.backbone, head), head, c_max


def build_datasets(args, fold=0):
    root = Path(__file__).resolve().parent / "data"
    tr_tf = get_transforms("train", args.img_size, args.augment_preset)
    va_tf = get_transforms("val", args.img_size, args.augment_preset)

    if args.dataset == "limuc":
        kw = dict(fold=fold, n_folds=args.n_folds)
        tr = LIMUCDataset(root / "limuc", "train", tr_tf, **kw)
        va = LIMUCDataset(root / "limuc", "val",   va_tf, **kw)
        te = LIMUCDataset(root / "limuc", "test",  va_tf)
    elif args.dataset == "aptos":
        kw = dict(fold=fold, n_folds=args.n_folds,
                  preprocess_dixit=(args.augment_preset == "dixit"))
        tr = APTOSDataset(root / "aptos2019", "train", tr_tf, **kw)
        va = APTOSDataset(root / "aptos2019", "val",   va_tf, **kw)
        te = va
    elif args.dataset == "kneexray":
        tr = KneeXrayDataset(root / "kneexray", "train", tr_tf)
        va = KneeXrayDataset(root / "kneexray", "val",   va_tf)
        te = KneeXrayDataset(root / "kneexray", "test",  va_tf)
    else:
        raise ValueError(args.dataset)
    return tr, va, te


def freeze_backbone(model, backbone_name, n_layers):
    if n_layers <= 0:
        return "none"
    if "resnet" in backbone_name:
        targets = []
        if n_layers >= 1: targets += ["conv1", "bn1"]
        if n_layers >= 2: targets += ["layer1"]
        if n_layers >= 3: targets += ["layer2"]
        if n_layers >= 4: targets += ["layer3"]
        if n_layers >= 5: targets += ["layer4"]
    elif backbone_name == "inception_v3":
        pool = ["Conv2d_1a", "Conv2d_2a", "Conv2d_2b", "Conv2d_3b",
                "Conv2d_4a", "Mixed_5b", "Mixed_5c", "Mixed_5d"]
        targets = pool[:min(n_layers * 2, len(pool))]
    elif "densenet" in backbone_name:
        pool = ["features.conv0", "features.norm0",
                "features.denseblock1", "features.transition1",
                "features.denseblock2", "features.transition2",
                "features.denseblock3", "features.transition3"]
        targets = pool[:min(n_layers * 2, len(pool))]
    elif "efficientnet" in backbone_name:
        targets = [f"features.{i}" for i in range(min(n_layers * 2, 9))]
    else:
        targets = []

    frozen = 0
    for name, p in model.backbone.named_parameters():
        if any(name.startswith(t) for t in targets):
            p.requires_grad = False
            frozen += p.numel()
    total = sum(p.numel() for p in model.backbone.parameters())
    trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    return (f"Freeze {targets} — {frozen/1e6:.1f}M/{total/1e6:.1f}M "
            f"({frozen/total*100:.0f}%), trainable {trainable/1e6:.1f}M")


def compute_class_weights(dataset, c_max, device):
    if hasattr(dataset, "samples"):
        lbls = [s[1] for s in dataset.samples]
    else:
        lbls = dataset.labels.tolist()
    counts = Counter(lbls)
    n, C = len(lbls), c_max + 1
    w = torch.zeros(C, device=device)
    for k in range(C):
        w[k] = n / (C * max(counts.get(k, 1), 1))
    w /= w.mean()
    return w


def print_metrics(metrics, c_max, prefix=""):
    sin2_str = f"  sin²θ={metrics['mean_sin2']:.4f}" if "mean_sin2" in metrics else ""
    print(f"{prefix}MAE={metrics['mae']:.4f}  QWK={metrics['qwk']:.4f}  "
          f"ACC={metrics['acc']:.4f}  F1={metrics['macro_f1']:.4f}  "
          f"Off-by-1={metrics['off_by_1']:.4f}{sin2_str}")
    recall_str = "  ".join(f"G{i}={metrics[f'recall_g{i}']:.3f}"
                           for i in range(c_max + 1))
    print(f"{prefix}Recall: {recall_str}")


# ─── Experiment Configs ───────────────────────────────────────────────────────

EXPERIMENTS = [
    # LIMUC — R50, fold 0
    {"tag": "limuc_FOROH",      "dataset": "limuc",    "backbone": "resnet50",    "method": "FOROH", "sin2_lambda": 0.0},
    {"tag": "limuc_FOROH_sin2", "dataset": "limuc",    "backbone": "resnet50",    "method": "FOROH", "sin2_lambda": 0.1},
    {"tag": "limuc_CE",         "dataset": "limuc",    "backbone": "resnet50",    "method": "ce",    "sin2_lambda": 0.0},
    # APTOS — R50, fold 0
    {"tag": "aptos_FOROH",      "dataset": "aptos",    "backbone": "resnet50",    "method": "FOROH", "sin2_lambda": 0.0},
    {"tag": "aptos_FOROH_sin2", "dataset": "aptos",    "backbone": "resnet50",    "method": "FOROH", "sin2_lambda": 0.1},
    {"tag": "aptos_CE",         "dataset": "aptos",    "backbone": "resnet50",    "method": "ce",    "sin2_lambda": 0.0},
    # KneeXray — D161, fold 0
    {"tag": "knee_FOROH",       "dataset": "kneexray", "backbone": "densenet161", "method": "FOROH", "sin2_lambda": 0.0},
    {"tag": "knee_FOROH_sin2",  "dataset": "kneexray", "backbone": "densenet161", "method": "FOROH", "sin2_lambda": 0.1},
    {"tag": "knee_CE",          "dataset": "kneexray", "backbone": "densenet161", "method": "ce",    "sin2_lambda": 0.0},
]


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FOROH sin²θ regularization experiment")
    p.add_argument("--dataset", default=None,
                   choices=["limuc", "aptos", "kneexray"],
                   help="Run single dataset (default: all 3)")
    p.add_argument("--sin2-lambda", type=float, default=None,
                   help="Override sin²θ λ for FOROH+sin2 experiments")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--freeze-layers", type=int, default=2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="outputs/sin2_exp")
    p.add_argument("--augment-preset", default="default")
    return p.parse_args()


def run_single(args, cfg):
    """Run one experiment configuration."""
    # Override args from cfg
    args.dataset = cfg["dataset"]
    args.backbone = cfg["backbone"]
    args.method = cfg["method"]
    sin2_lam = cfg["sin2_lambda"]
    if args.sin2_lambda_override is not None and "sin2" in cfg["tag"]:
        sin2_lam = args.sin2_lambda_override
    tag = cfg["tag"]

    # Auto img_size
    img_size = args.img_size
    if args.backbone == "inception_v3" and img_size == 224:
        img_size = 299
    if args.backbone == "efficientnet_b3" and img_size == 224:
        img_size = 256
    args.img_size = img_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  {tag}  |  {args.method}  |  {args.dataset}  |  {args.backbone}")
    print(f"  sin²θ λ = {sin2_lam}")
    print(f"{'='*70}")

    model, head, c_max = build_model(args)
    model = model.to(device)
    train_ds, val_ds, test_ds = build_datasets(args, args.fold)

    print(f"  Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")
    freeze_info = freeze_backbone(model, args.backbone, args.freeze_layers)
    print(f"  {freeze_info}")

    tr_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(val_ds, args.batch_size * 2, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(test_ds, args.batch_size * 2, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_mae, best_ep, best_state = float("inf"), 0, None
    patience_cnt = 0

    for ep in range(1, args.epochs + 1):
        loss, det = train_one_epoch(model, tr_loader, optimizer, args.method,
                                    head, device, scaler, "huber", None, 0.0,
                                    sin2_lam)
        scheduler.step()
        vm = evaluate(model, va_loader, args.method, head, c_max, device)

        det_str = " ".join(f"{k}={v:.4f}" for k, v in det.items())
        sin2_str = f" sin²θ={vm.get('mean_sin2', 0):.4f}" if "mean_sin2" in vm else ""
        print(f"  Ep {ep:3d} | loss={loss:.4f} ({det_str}) | "
              f"val MAE={vm['mae']:.4f} QWK={vm['qwk']:.4f}{sin2_str}")

        if vm["mae"] < best_mae:
            best_mae, best_ep = vm["mae"], ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  Early stopping at epoch {ep}")
                break

    print(f"\n  Best epoch: {best_ep} (val MAE={best_mae:.4f})")
    model.load_state_dict(best_state)
    model = model.to(device)

    tm = evaluate(model, te_loader, args.method, head, c_max, device)
    print(f"\n  ── Test ──")
    print_metrics(tm, c_max, prefix="  ")

    # Save
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {"args": vars(args), "tag": tag, "sin2_lambda": sin2_lam,
            "best_epoch": best_ep, "test_metrics": tm,
            "model_state": best_state}
    torch.save(save, out_dir / f"fold{args.fold}.pt")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"tag": tag, "sin2_lambda": sin2_lam,
                   "best_epoch": best_ep, "test": tm}, f, indent=2, default=float)

    return tag, tm


def main():
    args = parse_args()
    args.sin2_lambda_override = args.sin2_lambda  # save original
    args.sin2_lambda = None  # clear to avoid confusion

    exps = EXPERIMENTS
    if args.dataset:
        exps = [e for e in exps if e["dataset"] == args.dataset]

    results = {}
    for cfg in exps:
        # reset img_size for each experiment
        args.img_size = 224
        tag, tm = run_single(args, cfg)
        results[tag] = tm

    # ── Summary Table ──
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: sin²θ regularization experiment (fold {args.fold})")
    print(f"{'='*80}")
    print(f"{'Tag':<25s} {'MAE':>6s} {'QWK':>6s} {'ACC':>6s} {'F1':>6s}  {'sin²θ':>6s}  Best_Ep")
    print("-" * 80)
    for tag, m in results.items():
        sin2 = f"{m['mean_sin2']:.4f}" if "mean_sin2" in m else "  —   "
        print(f"{tag:<25s} {m['mae']:6.4f} {m['qwk']:6.4f} {m['acc']:6.4f} "
              f"{m['macro_f1']:6.4f}  {sin2}  —")

    # Save summary
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSummary saved to {out}/summary.json")


if __name__ == "__main__":
    main()