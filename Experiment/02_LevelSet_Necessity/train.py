"""Standalone E02 trainer following the same official five-fold protocol as E01."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import LIMUCDataset, get_transforms
from metrics import compute_metrics, print_metrics
from models import PointPrototypeHead, build_model


HUBER_DELTA = 0.5
C_MAX = 3
N_FOLDS = 5
SPLIT_SEED = 1
FOLD_SEEDS = (1, 2, 3, 4, 5)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "limuc"


def set_experiment_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def huber_per(error):
    ad = error.abs()
    return torch.where(
        ad <= HUBER_DELTA,
        0.5 * error.square() / HUBER_DELTA,
        ad - 0.5 * HUBER_DELTA,
    )


def freeze_backbone(model, n_layers=2):
    if n_layers <= 0:
        return "none"
    groups = [["conv1", "bn1"], ["layer1"], ["layer2"], ["layer3"], ["layer4"]]
    targets = [t for g in groups[:n_layers] for t in g]
    frozen = 0
    for name, p in model.backbone.named_parameters():
        if any(name.startswith(t) for t in targets):
            p.requires_grad = False
            frozen += p.numel()
    total = sum(p.numel() for p in model.backbone.parameters())
    return (
        f"Freeze {targets} — {frozen/1e6:.1f}M/{total/1e6:.1f}M "
        f"({frozen/total*100:.0f}%), trainable {(total-frozen)/1e6:.1f}M"
    )


def build_datasets(fold_index, img_size=224):
    tr_tf = get_transforms("train", img_size)
    va_tf = get_transforms("val", img_size)
    kw = dict(fold_index=fold_index, n_folds=N_FOLDS, split_seed=SPLIT_SEED)
    train_ds = LIMUCDataset(DATA_ROOT, "train", tr_tf, **kw)
    val_ds = LIMUCDataset(DATA_ROOT, "val", va_tf, **kw)
    test_ds = LIMUCDataset(DATA_ROOT, "test", va_tf)
    return train_ds, val_ds, test_ds


def compute_loss(output, labels, head):
    score, u, _, prototypes = output
    if isinstance(head, PointPrototypeHead):
        target = prototypes[labels]
        cos = torch.clamp((u.float() * target.float()).sum(-1), -1 + 1e-7, 1 - 1e-7)
        error = torch.acos(cos) * (C_MAX / math.pi)
    else:
        error = score - labels.float()
    return huber_per(error).mean()


def evaluate(model, loader, head, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for imgs, labels in loader:
            score, u, _, prototypes = model(imgs.to(device))
            if isinstance(head, PointPrototypeHead):
                cos = torch.clamp(u.float() @ prototypes.float().T, -1 + 1e-7, 1 - 1e-7)
                preds = torch.acos(cos).argmin(dim=-1).cpu()
            else:
                preds = score.round().clamp(0, C_MAX).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return compute_metrics(all_labels, all_preds, C_MAX)


def train_one_epoch(model, loader, optimizer, head, device, scaler):
    model.train()
    total, n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            output = model(imgs)
            loss = compute_loss(output, labels, head)
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def run_variant(
    variant,
    output_dir,
    *,
    proj_dim=128,
    epochs=50,
    batch_size=128,
    lr=1e-4,
    lr_head=1e-4,
    weight_decay=1e-4,
    img_size=224,
    freeze_layers=2,
    patience=10,
    num_workers=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(output_dir) / f"{variant}_limuc"
    run_dir.mkdir(parents=True, exist_ok=True)
    all_results, fold_meta = [], []

    for fold_id, experiment_seed in zip(range(1, N_FOLDS + 1), FOLD_SEEDS):
        if fold_id != experiment_seed:
            raise RuntimeError("Official protocol requires fold_id == experiment_seed")
        fold_index = fold_id - 1
        set_experiment_seed(experiment_seed)

        print(
            f"\n[E02 {variant} | Fold {fold_id}/5 | seed={experiment_seed} | "
            f"split_seed={SPLIT_SEED} | limuc | resnet50]"
        )
        model, head = build_model(variant, proj_dim=proj_dim, c_max=C_MAX)
        model = model.to(device)
        train_ds, val_ds, test_ds = build_datasets(fold_index, img_size)
        print(f"  Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")
        print(f"  {freeze_backbone(model, freeze_layers)}")

        train_loader = DataLoader(
            train_ds, batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size * 2, shuffle=False, num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size * 2, shuffle=False, num_workers=num_workers,
            pin_memory=True,
        )

        bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": bb_params, "lr": lr},
                {"params": model.head.parameters(), "lr": lr_head},
            ],
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        best_mae, best_epoch, best_state = float("inf"), 0, None
        patience_count = 0
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, head, device, scaler)
            val_metrics = evaluate(model, val_loader, head, device)
            scheduler.step()
            print(
                f"  Ep {epoch:3d} | loss={loss:.4f} | "
                f"val MAE={val_metrics['mae']:.4f} "
                f"QWK={val_metrics['qwk']:.4f} ACC={val_metrics['acc']:.4f}"
            )
            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        model = model.to(device)
        test_metrics = evaluate(model, test_loader, head, device)
        print(f"  Best epoch: {best_epoch} (val MAE={best_mae:.4f})")
        print("  Test:")
        print_metrics(test_metrics, C_MAX, prefix="    ")

        checkpoint = {
            "model_state": best_state,
            "variant": variant,
            "fold": fold_id,
            "fold_index": fold_index,
            "experiment_seed": experiment_seed,
            "split_seed": SPLIT_SEED,
            "best_epoch": best_epoch,
            "test_final": test_metrics,
            "config": {
                "dataset": "limuc", "backbone": "resnet50",
                "proj_dim": proj_dim, "epochs": epochs,
                "batch_size": batch_size, "lr": lr, "lr_head": lr_head,
                "weight_decay": weight_decay, "img_size": img_size,
                "freeze_layers": freeze_layers, "patience": patience,
                "optimizer": "adamw", "scheduler": "cosine",
                "n_folds": N_FOLDS, "split_seed": SPLIT_SEED,
                "fold_seed_rule": "experiment_seed == fold_id",
                "fold_seeds": list(FOLD_SEEDS),
            },
        }
        torch.save(checkpoint, run_dir / f"fold{fold_id}.pt")
        all_results.append(test_metrics)
        fold_meta.append({
            "fold": fold_id, "fold_index": fold_index,
            "experiment_seed": experiment_seed, "split_seed": SPLIT_SEED,
            "best_epoch": best_epoch, "best_val_mae": best_mae,
        })

    numeric_keys = list(all_results[0].keys())
    summary = {
        "variant": variant,
        "protocol": {
            "n_folds": N_FOLDS,
            "fold_ids": [1, 2, 3, 4, 5],
            "experiment_seeds": list(FOLD_SEEDS),
            "fold_seed_rule": "experiment_seed == fold_id",
            "split_seed": SPLIT_SEED,
            "single_command_runs_all_folds": True,
        },
        "fold_meta": fold_meta,
        "fold_results": all_results,
        "mean": {k: float(np.mean([r[k] for r in all_results])) for k in numeric_keys},
        "std": {k: float(np.std([r[k] for r in all_results])) for k in numeric_keys},
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n[E02 {variant} 5-fold summary]")
    for key in ("mae", "qwk", "acc", "macro_f1"):
        print(f"  {key:10s}: {summary['mean'][key]:.4f} ± {summary['std'][key]:.4f}")
    print(f"Saved to {run_dir}/")
    return summary
