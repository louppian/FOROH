"""Shared training engine for current FOROH experiments.

Each experiment defines its settings directly in its own ``run.py`` and calls
``run_experiment``. No YAML/config-file layer is used for new experiments.
Historical trainers remain untouched.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Dataset import C_MAX, build_datasets
from Model import build_model, freeze_backbone
from Experiment.common.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = dict(
    dataset="limuc",
    backbone="resnet50",
    proj_dim=128,
    dropout=0.3,
    fold=0,
    n_folds=10,
    split_seed=42,
    seed=42,
    optimizer="adamw",
    lr_backbone=1e-4,
    lr_head=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    epochs=50,
    scheduler="cosine",
    scheduler_patience=10,
    patience=10,
    freeze_layers=2,
    img_size=224,
    huber_delta=0.5,
    num_workers=4,
    class_weighting=False,
    output_dir="Result",
)


def _prepare_config(overrides):
    cfg = {**DEFAULTS, **overrides}
    for key in ("experiment_id", "method"):
        if key not in cfg:
            raise ValueError(f"Missing required setting: {key}")
    return cfg


def _huber(error, delta):
    ad = error.abs()
    return torch.where(ad <= delta, 0.5 * error.square() / delta, ad - 0.5 * delta)


def compute_loss(output, labels, cfg, class_weights=None):
    if cfg["method"] == "point_prototype":
        u = output["u"].float()
        target = output["prototypes"][labels].float()
        cos = torch.clamp((u * target).sum(-1), -1 + 1e-7, 1 - 1e-7)
        error = torch.acos(cos) * (C_MAX[cfg["dataset"]] / math.pi)
    else:
        error = output["score"] - labels.float()

    per = _huber(error, cfg["huber_delta"])
    if class_weights is not None:
        per = per * class_weights[labels]
    return per.mean()


def evaluate(model, loader, c_max, device):
    model.eval()
    scores, labels, thetas = [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for images, y in loader:
            out = model(images.to(device))
            scores.extend(out["score"].detach().cpu().numpy())
            labels.extend(y.numpy())
            if "theta" in out:
                thetas.extend(out["theta"].detach().cpu().numpy())

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    preds = np.clip(np.rint(scores), 0, c_max).astype(int)
    metrics = compute_metrics(labels, preds, c_max)
    return metrics, dict(
        scores=scores,
        labels=labels,
        preds=preds,
        thetas=np.asarray(thetas, dtype=float) if thetas else None,
    )


def _class_weights(dataset, c_max, device):
    labels = [sample[1] for sample in dataset.samples]
    counts = Counter(labels)
    n, C = len(labels), c_max + 1
    weights = torch.tensor(
        [n / (C * max(counts.get(k, 1), 1)) for k in range(C)],
        dtype=torch.float32,
        device=device,
    )
    return weights / weights.mean()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _run_dir(cfg):
    bb = cfg["backbone"].replace("resnet50", "r50").replace("resnet18", "r18")
    return (
        Path(cfg["output_dir"])
        / f"{cfg['experiment_id']}_{cfg['method']}"
        / f"{cfg['dataset']}_{bb}"
        / f"fold{cfg['fold']:02d}_seed{cfg['seed']}"
    )


def _save(run_dir, cfg, model, best_epoch, val_metrics, test_metrics, raw, history):
    run_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}

    with open(run_dir / "config.json", "w") as f:
        json.dump(clean_cfg, f, indent=2)

    manifest = dict(
        experiment_id=cfg["experiment_id"],
        git_commit=commit,
        git_branch="reproduce-paper",
        hostname=platform.node(),
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        dataset=cfg["dataset"],
        fold=cfg["fold"],
        n_folds=cfg["n_folds"],
        split_seed=cfg["split_seed"],
        seed=cfg["seed"],
        head_trainable_parameters=sum(p.numel() for p in model.head.parameters() if p.requires_grad),
        started_at=cfg["_started_at"],
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
    )
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(run_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_idx", "label", "prediction", "score"]
        if raw["thetas"] is not None:
            header.append("theta")
        writer.writerow(header)
        for i in range(len(raw["labels"])):
            row = [i, int(raw["labels"][i]), int(raw["preds"][i]), float(raw["scores"][i])]
            if raw["thetas"] is not None:
                row.append(float(raw["thetas"][i]))
            writer.writerow(row)

    torch.save(dict(
        model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        config=clean_cfg,
        best_epoch=best_epoch,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        git_commit=commit,
    ), run_dir / "checkpoint.pt")


def run_experiment(settings, force=False):
    cfg = _prepare_config(settings)
    run_dir = _run_dir(cfg)
    if (run_dir / "metrics.json").exists() and not force:
        print(f"SKIP {cfg['experiment_id']}: {run_dir} already exists")
        return None

    cfg["_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c_max = C_MAX[cfg["dataset"]]
    print(f"\n=== {cfg['experiment_id']} | {cfg['method']} | fold {cfg['fold']} | seed {cfg['seed']} ===")

    model, _ = build_model(
        cfg["method"], cfg["backbone"], c_max,
        proj_dim=cfg["proj_dim"], dropout=cfg["dropout"],
    )
    model = model.to(device)
    frozen, total = freeze_backbone(model.backbone, cfg["backbone"], cfg["freeze_layers"])
    head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(f"Backbone frozen: {frozen/1e6:.1f}M / {total/1e6:.1f}M")
    print(f"Head trainable params: {head_params:,}")

    train_ds, val_ds, test_ds = build_datasets(
        cfg["dataset"], fold=cfg["fold"], n_folds=cfg["n_folds"],
        seed=cfg["split_seed"], img_size=cfg["img_size"],
    )
    print(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, cfg["batch_size"] * 2, shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, cfg["batch_size"] * 2, shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)

    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = opt_cls([
        {"params": bb_params, "lr": cfg["lr_backbone"]},
        {"params": model.head.parameters(), "lr": cfg["lr_head"]},
    ], weight_decay=cfg["weight_decay"])

    if cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=cfg["scheduler_patience"])

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    class_weights = _class_weights(train_ds, c_max, device) if cfg["class_weighting"] else None
    print(f"Class weighting: {'ON' if class_weights is not None else 'OFF'}")

    best_mae, best_epoch, best_state = float("inf"), 0, None
    stale, history = 0, []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for images, y in train_loader:
            images, y = images.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                loss = compute_loss(model(images), y, cfg, class_weights)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses.append(loss.item())

        val_metrics, _ = evaluate(model, val_loader, c_max, device)
        if cfg["scheduler"] == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_metrics["mae"])

        train_loss = float(np.mean(losses))
        history.append(dict(epoch=epoch, train_loss=train_loss,
                            val_mae=val_metrics["mae"], val_qwk=val_metrics["qwk"],
                            val_macro_f1=val_metrics["macro_f1"]))
        print(f"Ep {epoch:03d} loss={train_loss:.4f} val_MAE={val_metrics['mae']:.4f} val_QWK={val_metrics['qwk']:.4f}")

        if val_metrics["mae"] < best_mae:
            best_mae, best_epoch, stale = val_metrics["mae"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(device)

    # Validation selected the checkpoint. Test is observed once afterward.
    val_metrics, _ = evaluate(model, val_loader, c_max, device)
    test_metrics, raw = evaluate(model, test_loader, c_max, device)
    print(f"TEST MAE={test_metrics['mae']:.4f} QWK={test_metrics['qwk']:.4f} F1={test_metrics['macro_f1']:.4f}")

    _save(run_dir, cfg, model, best_epoch, val_metrics, test_metrics, raw, history)
    print(f"Saved: {run_dir}")
    return test_metrics
