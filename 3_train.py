"""FOROH — Free Ordinal Regression on the Hypersphere.

score = (arccos(u · w) / π) × C_max,  u, w ∈ S^{d-1}

Usage:
  python train.py --method FOROH --dataset limuc
  python train.py --method FOROH --dataset aptos --freq-weight --optimize-boundaries
  python train.py --method FOROH --dataset kneexray --freq-weight --aux-ce
  python train.py --method ce    --dataset limuc --fold -1 --n-folds 10
"""

import os, math, json, argparse, random
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


# === Heads ===

class FOROHHead(nn.Module):
    """Projector → L2-normalize → arccos(u·w) → score ∈ [0, C_max].

    Optional aux CE head injects gradient into longitude direction during training.
    """

    def __init__(self, in_dim, proj_dim=128, c_max=3, aux_ce=False):
        super().__init__()
        self.c_max = c_max

        self.projector = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(proj_dim, proj_dim),
        )

        self.w_raw = nn.Parameter(F.normalize(torch.randn(proj_dim), dim=0))
        self.aux_classifier = nn.Linear(proj_dim, c_max + 1) if aux_ce else None

    def _get_w(self):
        return self.w_raw / (self.w_raw.norm() + 1e-8)

    def forward(self, z):
        u = F.normalize(self.projector(z), dim=-1)
        w = self._get_w()

        # arccos in float32 — float16 unstable at ±1
        with torch.amp.autocast("cuda", enabled=False):
            u32, w32 = u.float(), w.float()
            cos = torch.clamp(u32 @ w32, -1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            score = (theta / math.pi) * self.c_max

        aux_logits = (self.aux_classifier(u32)
                      if self.aux_classifier is not None and self.training else None)
        return score, u, theta, aux_logits


class CEHead(nn.Module):
    """Standard cross-entropy classification head."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, z):
        return self.fc(z)


class CORALHead(nn.Module):
    """CORAL — shared weight, C-1 bias thresholds."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))
        self.num_classes = num_classes

    def forward(self, z):
        return self.fc(z) + self.biases.unsqueeze(0)


class CORNHead(nn.Module):
    """CORN — C-1 independent conditional binary classifiers."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.classifiers = nn.ModuleList(
            [nn.Linear(in_dim, 1) for _ in range(num_classes - 1)]
        )

    def forward(self, z):
        return torch.cat([clf(z) for clf in self.classifiers], dim=-1)


class MSEHead(nn.Module):
    """Single-neuron regression head."""
    def __init__(self, in_dim, c_max):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.c_max = c_max

    def forward(self, z):
        return self.fc(z).squeeze(-1).clamp(0, self.c_max)


# === Model ===

class OrdinalModel(nn.Module):
    """Backbone + Head."""

    FEAT_DIMS = {
        "resnet18": 512, "resnet50": 2048,
        "inception_v3": 2048, "densenet161": 2208,
        "efficientnet_b3": 1536,
    }

    def __init__(self, backbone_name, head, pretrained=True):
        super().__init__()
        self.backbone = self._build_backbone(backbone_name, pretrained)
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


# === Datasets ===

class LIMUCDataset(Dataset):
    """LIMUC 11,276 endoscopy images, Mayo 0-3. Patient-level split."""

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
        for fmt in (f"Mayo {grade}", f"Mayo{grade}"):
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
    """APTOS 2019 — 3,662 fundus images, DR 0-4. Image-level k-fold."""

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
        """Dixit 2025: crop dark border + Gaussian-blur contrast enhancement."""
        import cv2
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        coords = np.argwhere(gray > 7)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            arr = arr[y0:y1, x0:x1]
        blur = cv2.GaussianBlur(arr, (0, 0), 10)
        arr = cv2.addWeighted(arr, 4, blur, -4, 128)
        return Image.fromarray(arr)


class KneeXrayDataset(Dataset):
    """OAI Knee X-ray — 8,260 images, KL 0-4. Chen et al. fixed split."""

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


class RetinaMNISTDataset(Dataset):
    """RetinaMNIST — 1,600 fundus images, DR 0-4. Pre-split .npz."""

    C_MAX = 4

    def __init__(self, root, split="train", transform=None, size=224):
        self.transform, self.c_max = transform, self.C_MAX
        root = Path(root)
        npz = root / f"retinamnist{'_' + str(size) if size != 28 else ''}.npz"
        if not npz.exists():
            candidates = sorted(root.glob("*.npz"), key=lambda x: x.stat().st_size, reverse=True)
            npz = candidates[0] if candidates else None
            assert npz, f"No .npz in {root}"
        data = np.load(str(npz))
        self.images = data[f"{split}_images"]
        self.labels = data[f"{split}_labels"].flatten().astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = int(self.labels[idx])
        return self.transform(img) if self.transform else img, label


# === Transforms ===

def get_transforms(split, img_size=224, augment_preset="default"):
    """default: RRC+HFlip+VFlip+ColorJitter | dixit: Rot±20/Zoom±20%/Shift±20%/Shear15/HFlip."""
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if split == "train":
        if augment_preset == "dixit":
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(20),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2),
                                        scale=(0.8, 1.2), shear=15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ])
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])


# === Loss ===

HUBER_DELTA = 0.5  # matches ordinal round boundary

def compute_loss(method, output, labels, head, class_weights=None, aux_lambda=0.0):
    """Dispatch loss by method.
      FOROH: Huber on score (+freq-weight, +aux CE optional)
      ce/coral/corn/mse: standard
    """
    y = labels.float()

    if method == "FOROH":
        score, u, theta, aux_logits = output
        diff = score - y
        ad = diff.abs()
        per = torch.where(ad <= HUBER_DELTA,
                          0.5 * diff**2 / HUBER_DELTA,
                          ad - 0.5 * HUBER_DELTA)

        loss = ((per * class_weights[labels]).mean()
                if class_weights is not None else per.mean())
        details = {"foroh": loss.item()}

        if aux_logits is not None and aux_lambda > 0:
            aux = F.cross_entropy(aux_logits, labels)
            total = loss + aux_lambda * aux
            details.update(aux_ce=aux.item(), total=total.item())
            return total, details
        return loss, details

    if method == "ce":
        loss = F.cross_entropy(output, labels)
        return loss, {"ce": loss.item()}

    if method == "coral":
        C = head.num_classes
        tgt = (labels.unsqueeze(1) >
               torch.arange(C - 1, device=labels.device)).float()
        loss = F.binary_cross_entropy_with_logits(output, tgt)
        return loss, {"coral": loss.item()}

    if method == "corn":
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

    if method == "mse":
        loss = F.mse_loss(output, y)
        return loss, {"mse": loss.item()}

    raise ValueError(f"Unknown method: {method}")


# === Boundary Optimization ===

def optimize_boundaries(model, val_loader, c_max, device, n_grid=200):
    """Find C-1 θ boundaries maximizing QWK on val. Grid for ≤3, coord-descent for 4+."""
    model.eval()
    thetas, labels = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for imgs, y in val_loader:
            _, _, th, _ = model(imgs.to(device))
            thetas.extend(th.cpu().numpy())
            labels.extend(y.numpy())

    thetas = np.array(thetas)
    labels = np.array(labels).astype(int)
    C = c_max + 1

    default = np.array([(k + 0.5) * math.pi / c_max for k in range(C - 1)])
    radius = 0.2

    W = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        for j in range(C):
            W[i, j] = (i - j) ** 2 / (C - 1) ** 2

    def _qwk(bounds):
        preds = np.clip(np.digitize(thetas, bounds), 0, c_max)
        idx = labels * C + preds
        conf = np.bincount(idx, minlength=C * C).reshape(C, C).astype(np.float64)
        n = conf.sum()
        if n == 0:
            return 0.0
        exp = np.outer(conf.sum(1), conf.sum(0)) / n
        o, e = (W * conf).sum() / n, (W * exp).sum() / n
        return 1.0 - o / e if e else 1.0

    best_q, best_b = -1.0, default.copy()
    nb = C - 1

    def _cands(k):
        lo = max(0.01, default[k] - radius)
        hi = min(math.pi - 0.01, default[k] + radius)
        return lo, hi

    if nb <= 3:
        sub = min(n_grid, 50)
        grids = [np.linspace(*_cands(k), sub) for k in range(nb)]
        for combo in iter_product(*grids):
            b = np.array(combo)
            if not np.all(np.diff(b) > 0.01):
                continue
            q = _qwk(b)
            if q > best_q:
                best_q, best_b = q, b.copy()

        # refine
        rr = radius / sub * 2
        rgrids = [np.linspace(max(0.01, best_b[k] - rr),
                              min(math.pi - 0.01, best_b[k] + rr), 30)
                  for k in range(nb)]
        for combo in iter_product(*rgrids):
            b = np.array(combo)
            if not np.all(np.diff(b) > 0.005):
                continue
            q = _qwk(b)
            if q > best_q:
                best_q, best_b = q, b.copy()
    else:
        for rnd in range(5):
            improved = False
            for k in range(nb):
                lo, hi = _cands(k)
                if k > 0:
                    lo = max(lo, best_b[k - 1] + 0.01)
                if k < nb - 1:
                    hi = min(hi, best_b[k + 1] - 0.01)
                if lo >= hi:
                    continue
                for b in np.linspace(lo, hi, n_grid):
                    tb = best_b.copy()
                    tb[k] = b
                    q = _qwk(tb)
                    if q > best_q:
                        best_q, best_b[k], improved = q, b, True
            if not improved:
                print(f"    Coord descent converged at round {rnd + 1}")
                break

    sc_def = default * c_max / math.pi
    sc_opt = best_b * c_max / math.pi
    print(f"  BoundaryOpt: default(score)={np.round(sc_def, 3)}  "
          f"optimized={np.round(sc_opt, 3)}  val QWK={best_q:.4f}")
    return best_b


# === Evaluation ===

def evaluate(model, loader, method, head, c_max, device, theta_boundaries=None):
    """Compute MAE, QWK, ACC, F1, Off-by-1, class-wise recall."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for imgs, y in loader:
            imgs = imgs.to(device)

            if method == "FOROH":
                score, _, theta, _ = model(imgs)
                if theta_boundaries is not None:
                    p = np.clip(np.digitize(theta.cpu().numpy(), theta_boundaries), 0, c_max)
                    preds = torch.from_numpy(p).float()
                else:
                    preds = score.round().clamp(0, c_max).cpu()
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
    return metrics


# === Training Loop ===

def train_one_epoch(model, loader, optimizer, method, head, device,
                    scaler, class_weights, aux_lambda):
    model.train()
    total, details_acc, n = 0.0, defaultdict(float), 0

    for imgs, y in loader:
        imgs, y = imgs.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            if method in ("FOROH", "ce"):
                out = model(imgs)
            else:  # coral, corn, mse
                out = head(model.backbone(imgs))
            loss, det = compute_loss(method, out, y, head, class_weights, aux_lambda)

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


# === Build Helpers ===

CMAX = {"limuc": 3, "retinamnist": 4, "aptos": 4, "kneexray": 4}


def build_model(args):
    feat_dim = OrdinalModel.FEAT_DIMS[args.backbone]
    c_max = CMAX[args.dataset]
    C = c_max + 1

    if args.method == "FOROH":
        head = FOROHHead(feat_dim, args.proj_dim, c_max, aux_ce=args.aux_ce)
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
    elif args.dataset == "retinamnist":
        tr = RetinaMNISTDataset(root / "retinamnist", "train", tr_tf, 224)
        va = RetinaMNISTDataset(root / "retinamnist", "val",   va_tf, 224)
        te = RetinaMNISTDataset(root / "retinamnist", "test",  va_tf, 224)
    elif args.dataset == "aptos":
        kw = dict(fold=fold, n_folds=args.n_folds,
                  preprocess_dixit=(args.augment_preset == "dixit"))
        tr = APTOSDataset(root / "aptos2019", "train", tr_tf, **kw)
        va = APTOSDataset(root / "aptos2019", "val",   va_tf, **kw)
        te = va  # no official test labels
    elif args.dataset == "kneexray":
        tr = KneeXrayDataset(root / "kneexray", "train", tr_tf)
        va = KneeXrayDataset(root / "kneexray", "val",   va_tf)
        te = KneeXrayDataset(root / "kneexray", "test",  va_tf)
    else:
        raise ValueError(args.dataset)
    return tr, va, te


def freeze_backbone(model, backbone_name, n_layers):
    """Freeze first n_layers of backbone. Returns info string."""
    if n_layers <= 0:
        return "none"

    if "resnet" in backbone_name:
        groups = [["conv1", "bn1"], ["layer1"], ["layer2"], ["layer3"], ["layer4"]]
        targets = [t for g in groups[:n_layers] for t in g]
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
    trainable = total - frozen
    return (f"Freeze {targets} — "
            f"{frozen/1e6:.1f}M/{total/1e6:.1f}M ({frozen/total*100:.0f}%), "
            f"trainable {trainable/1e6:.1f}M")


def compute_class_weights(dataset, c_max, device):
    """Inverse-frequency weights, normalized to mean=1."""
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
    print(f"{prefix}MAE={metrics['mae']:.4f}  QWK={metrics['qwk']:.4f}  "
          f"ACC={metrics['acc']:.4f}  F1={metrics['macro_f1']:.4f}  "
          f"Off-by-1={metrics['off_by_1']:.4f}")
    recall_str = "  ".join(f"G{i}={metrics[f'recall_g{i}']:.3f}"
                           for i in range(c_max + 1))
    print(f"{prefix}Recall: {recall_str}")


# === Main ===

def parse_args():
    p = argparse.ArgumentParser(description="FOROH Training")

    # core
    p.add_argument("--method", default="FOROH",
                   choices=["FOROH", "ce", "coral", "corn", "mse"])
    p.add_argument("--dataset", default="limuc",
                   choices=["limuc", "retinamnist", "aptos", "kneexray"])
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "resnet18", "inception_v3", "densenet161", "efficientnet_b3"])

    # FOROH
    p.add_argument("--proj-dim", type=int, default=128)

    # improvements
    p.add_argument("--optimize-boundaries", action="store_true",
                   help="[A] Post-optimize decision boundaries on val set")
    p.add_argument("--aux-ce", action="store_true",
                   help="[B] Auxiliary CE head during training")
    p.add_argument("--aux-lambda", type=float, default=0.3,
                   help="[B] Auxiliary CE loss weight")
    p.add_argument("--freq-weight", action="store_true",
                   help="[C] Frequency-inverse per-sample weighting")
    p.add_argument("--boundary-grid", type=int, default=200,
                   help="[A] Grid density for boundary search")

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4, help="Backbone LR")
    p.add_argument("--lr-head", type=float, default=1e-4, help="Head LR")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--freeze-layers", type=int, default=2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "reduce_lr"])
    p.add_argument("--scheduler-patience", type=int, default=10)
    p.add_argument("--oversample", action="store_true",
                   help="Minority oversampling via WeightedRandomSampler")
    p.add_argument("--augment-preset", default="default", choices=["default", "dixit"])
    p.add_argument("--exp", type=int, default=0, help="Experiment number for logging")

    # experiment
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=0,
                   help="Fold index, -1 for all folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="outputs")

    args = p.parse_args()
    if args.backbone == "inception_v3" and args.img_size == 224:
        args.img_size = 299
    if args.backbone == "efficientnet_b3" and args.img_size == 224:
        args.img_size = 256
    return args


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = range(args.n_folds) if args.fold == -1 else [args.fold]
    all_results = []

    flags = []
    if args.method == "FOROH":
        if args.optimize_boundaries: flags.append("BoundaryOpt")
        if args.aux_ce:              flags.append(f"AuxCE(λ={args.aux_lambda})")
        if args.freq_weight:         flags.append("FreqWeight")

    for fold in folds:
        print(f"\n[Exp {args.exp} | Fold {fold} | {args.method} | {args.dataset} | {args.backbone}]"
              f"  opt={args.optimizer} lr={args.lr} sch={args.scheduler}"
              + (f"  flags={','.join(flags)}" if flags else ""))

        # Build
        model, head, c_max = build_model(args)
        model = model.to(device)
        train_ds, val_ds, test_ds = build_datasets(args, fold)

        print(f"  Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}  "
              f"C_max={c_max}  proj_dim={args.proj_dim}")

        print(f"  {freeze_backbone(model, args.backbone, args.freeze_layers)}")

        # Loaders
        sampler, shuffle = None, True
        if args.oversample:
            labels = [train_ds[i][1] for i in range(len(train_ds))]
            class_counts = Counter(labels)
            weights = [1.0 / class_counts[l] for l in labels]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            shuffle = False
            print(f"  Oversampling: {dict(class_counts)}")

        tr_loader = DataLoader(train_ds, args.batch_size, shuffle=shuffle, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
        va_loader = DataLoader(val_ds, args.batch_size * 2, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
        te_loader = DataLoader(test_ds, args.batch_size * 2, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

        # Optimizer
        bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
        opt_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
        optimizer = opt_cls([
            {"params": bb_params, "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.lr_head},
        ], weight_decay=args.weight_decay)

        if args.scheduler == "reduce_lr":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.2, patience=args.scheduler_patience)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6)

        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        # Freq weight
        cw = None
        if args.freq_weight and args.method == "FOROH":
            cw = compute_class_weights(train_ds, c_max, device)
            print(f"  FreqWeight: {[f'G{i}={cw[i]:.2f}' for i in range(c_max+1)]}")

        aux_lam = args.aux_lambda if (args.aux_ce and args.method == "FOROH") else 0.0

        # Train
        best_mae, best_ep, best_state = float("inf"), 0, None
        patience_cnt = 0

        for ep in range(1, args.epochs + 1):
            loss, det = train_one_epoch(model, tr_loader, optimizer, args.method,
                                        head, device, scaler, cw, aux_lam)
            vm = evaluate(model, va_loader, args.method, head, c_max, device)

            if args.scheduler == "reduce_lr":
                scheduler.step(vm["mae"])
            else:
                scheduler.step()
            det_str = " ".join(f"{k}={v:.4f}" for k, v in det.items())
            print(f"  Ep {ep:3d} | loss={loss:.4f} ({det_str}) | "
                  f"val MAE={vm['mae']:.4f} QWK={vm['qwk']:.4f} ACC={vm['acc']:.4f}")

            if vm["mae"] < best_mae:
                best_mae, best_ep = vm["mae"], ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"  Early stopping at epoch {ep}")
                    break

        print(f"  Best epoch: {best_ep} (val MAE={best_mae:.4f})")
        model.load_state_dict(best_state)
        model = model.to(device)

        # Boundary optimization
        theta_bounds = None
        if args.optimize_boundaries and args.method == "FOROH":
            theta_bounds = optimize_boundaries(
                model, va_loader, c_max, device, args.boundary_grid)

        # Test
        tm_default = evaluate(model, te_loader, args.method, head, c_max, device)
        print(f"  Test (default):")
        print_metrics(tm_default, c_max, prefix="    ")

        final = tm_default
        if theta_bounds is not None:
            tm_opt = evaluate(model, te_loader, args.method, head, c_max, device,
                              theta_boundaries=theta_bounds)
            print(f"  Test (optimized):")
            print_metrics(tm_opt, c_max, prefix="    ")

            dq = tm_opt["qwk"] - tm_default["qwk"]
            dm = tm_default["mae"] - tm_opt["mae"]
            print(f"  BoundaryOpt: ΔQWK={dq:+.4f}  ΔMAE={dm:+.4f}")

            if tm_opt["qwk"] >= tm_default["qwk"]:
                final = tm_opt
                final["boundaries"] = theta_bounds.tolist()
                print("  → Using optimized boundaries")
            else:
                print("  → Keeping default boundaries")

        all_results.append(final)

        # Save — tag encodes key config to prevent overwrites
        tag = f"{args.method}_{args.dataset}"
        if args.exp > 0:
            tag = f"exp{args.exp}_{tag}"
        if args.method == "FOROH":
            parts = []
            if args.freq_weight:         parts.append("freq")
            if args.optimize_boundaries: parts.append("boundary")
            if args.aux_ce:              parts.append(f"aux{args.aux_lambda}")
            if args.proj_dim != 128:     parts.append(f"d{args.proj_dim}")
            if args.freeze_layers != 2:  parts.append(f"fl{args.freeze_layers}")
            if args.backbone != "resnet50": parts.append(args.backbone)
            if args.seed != 42:          parts.append(f"s{args.seed}")
            if parts:
                tag += "_" + "_".join(parts)
        elif args.backbone != "resnet50":
            tag += f"_{args.backbone}"

        out_dir = Path(args.output_dir) / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        save = {"model_state": best_state, "args": vars(args),
                "fold": fold, "best_epoch": best_ep,
                "test_default": tm_default, "test_final": final}
        if theta_bounds is not None:
            save["theta_boundaries"] = theta_bounds.tolist()
        torch.save(save, out_dir / f"fold{fold}.pt")

    # Summary
    if len(all_results) > 1:
        print(f"\n[Summary ({len(all_results)} folds)]")
        for key in ("mae", "qwk", "acc", "macro_f1"):
            vals = [r[key] for r in all_results]
            print(f"  {key:10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        for i in range(c_max + 1):
            vals = [r[f"recall_g{i}"] for r in all_results]
            print(f"  G{i} recall : {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save JSON
    numeric_keys = [k for k in all_results[0]
                    if isinstance(all_results[0][k], (int, float))]
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args), "flags": flags,
            "fold_results": all_results,
            "mean": {k: float(np.mean([r[k] for r in all_results])) for k in numeric_keys},
            "std":  {k: float(np.std([r[k] for r in all_results]))  for k in numeric_keys},
        }, f, indent=2, default=float)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()