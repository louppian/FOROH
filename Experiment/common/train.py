"""Config-driven training engine for current FOROH experiments.

Usage:
  python Experiment/common/train.py --config Experiment/01_.../configs/E01C_foroh.yaml

Scientific rules enforced here:
- model selection uses validation only
- test set is evaluated once after model selection
- split seed is separate from training seed
- class weighting is opt-in, not silently enabled
- point-prototype control uses full geodesic supervision instead of an
  extra weighted auxiliary loss
"""

import argparse
import csv
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Dataset import build_datasets, C_MAX
from Model import build_model, freeze_backbone


DEFAULTS = {
    "method": "foroh",
    "dataset": "limuc",
    "backbone": "resnet50",
    "proj_dim": 128,
    "dropout": 0.3,
    "fold": 0,
    "n_folds": 10,
    "seed": 42,
    "split_seed": 42,
    "optimizer": "adamw",
    "lr_backbone": 1e-4,
    "lr_head": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 50,
    "scheduler": "cosine",
    "patience": 10,
    "freeze_layers": 2,
    "img_size": 224,
    "huber_delta": 0.5,
    "class_weighting": False,
    "num_workers": 4,
    "output_dir": "Result",
    "experiment_id": "E00",
}


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    merged = {**DEFAULTS, **cfg}
    merged["_config_path"] = str(path)
    return merged


def huber_per_sample(pred, target, delta):
    diff = pred - target
    ad = diff.abs()
    return torch.where(ad <= delta, 0.5 * diff ** 2 / delta, ad - 0.5 * delta)


def compute_loss(output, labels, cfg, class_weights=None):
    """Matched Huber objective for Phase-1 controls.

    Euclidean/cosine/FOROH use Huber(score, grade).
    Point-prototype uses Huber(full spherical geodesic error, 0), expressed
    in grade units. This changes only the point-vs-level-set constraint and
    avoids introducing a tunable auxiliary-loss coefficient.
    """
    y = labels.float()
    delta = cfg["huber_delta"]

    if cfg["method"] == "point_prototype":
        u = output["u"].float()
        target_protos = output["prototypes"][labels].float()
        dot = torch.clamp((u * target_protos).sum(dim=-1), -1 + 1e-7, 1 - 1e-7)
        geodesic = torch.acos(dot)
        point_error = (geodesic / math.pi) * C_MAX[cfg["dataset"]]
        per = huber_per_sample(point_error, torch.zeros_like(point_error), delta)
        details_name = "prototype_geodesic_huber"
    else:
        per = huber_per_sample(output["score"], y, delta)
        details_name = "huber"

    if class_weights is not None:
        loss = (per * class_weights[labels]).mean()
    else:
        loss = per.mean()
    return loss, {details_name: loss.item()}


def evaluate(model, loader, c_max, device):
    from Experiment.common.metrics import compute_metrics

    model.eval()
    all_scores, all_labels, all_thetas = [], [], []

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for imgs, y in loader:
            out = model(imgs.to(device))
            all_scores.extend(out["score"].detach().cpu().numpy())
            all_labels.extend(y.numpy())
            if "theta" in out:
                all_thetas.extend(out["theta"].detach().cpu().numpy())

    preds = np.clip(np.round(all_scores), 0, c_max).astype(int)
    labels = np.array(all_labels, dtype=int)
    metrics = compute_metrics(labels, preds, c_max)

    sample_paths = None
    if hasattr(loader.dataset, "samples") and len(loader.dataset.samples) == len(labels):
        sample_paths = [str(s[0]) for s in loader.dataset.samples]

    patient_ids = None
    if getattr(loader.dataset, "patient_ids", None) is not None:
        if len(loader.dataset.patient_ids) == len(labels):
            patient_ids = list(loader.dataset.patient_ids)

    return metrics, {
        "scores": np.array(all_scores),
        "labels": labels,
        "preds": preds,
        "thetas": np.array(all_thetas) if all_thetas else None,
        "sample_paths": sample_paths,
        "patient_ids": patient_ids,
    }


def train_one_epoch(model, loader, optimizer, cfg, device, scaler, class_weights):
    model.train()
    total_loss = 0.0
    details_acc = defaultdict(float)
    n_batches = 0

    for imgs, y in loader:
        imgs, y = imgs.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            out = model(imgs)
            loss, det = compute_loss(out, y, cfg, class_weights)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        for k, v in det.items():
            details_acc[k] += v
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, {k: v / n for k, v in details_acc.items()}


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


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_results(out_dir, cfg, fold, best_epoch, best_state,
                 val_metrics, test_metrics, raw_preds, history):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                  f, default_flow_style=False, allow_unicode=True)

    manifest = {
        "experiment_id": cfg["experiment_id"],
        "git_commit": get_git_commit(),
        "git_branch": "reproduce-paper",
        "hostname": platform.node(),
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "started_at": cfg.get("_started_at", ""),
        "dataset": cfg["dataset"],
        "dataset_split": f"v1_patient{cfg['n_folds']}fold_seed{cfg['split_seed']}",
        "fold": fold,
        "seed": cfg["seed"],
        "split_seed": cfg["split_seed"],
        "class_weighting": cfg["class_weighting"],
        "head_trainable_params": cfg.get("_head_trainable_params"),
        "model_trainable_params": cfg.get("_model_trainable_params"),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)

    if raw_preds is not None:
        with open(out_dir / "predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            header = ["sample_idx", "sample_path", "patient_id", "label", "prediction", "score"]
            if raw_preds["thetas"] is not None:
                header.append("theta")
            writer.writerow(header)
            for i in range(len(raw_preds["labels"])):
                sample_path = raw_preds["sample_paths"][i] if raw_preds["sample_paths"] else ""
                patient_id = raw_preds["patient_ids"][i] if raw_preds["patient_ids"] else ""
                row = [
                    i,
                    sample_path,
                    patient_id,
                    int(raw_preds["labels"][i]),
                    int(raw_preds["preds"][i]),
                    f"{raw_preds['scores'][i]:.6f}",
                ]
                if raw_preds["thetas"] is not None:
                    row.append(f"{raw_preds['thetas'][i]:.6f}")
                writer.writerow(row)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save({
        "model_state": best_state,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "fold": fold,
        "seed": cfg["seed"],
        "split_seed": cfg["split_seed"],
        "best_epoch": best_epoch,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "git_commit": manifest["git_commit"],
    }, out_dir / "checkpoint.pt")


def _run_dir(cfg):
    eid = cfg["experiment_id"]
    method_tag = cfg["method"]
    ds = cfg["dataset"]
    bb = cfg["backbone"].replace("resnet50", "r50").replace("resnet18", "r18")
    fold = cfg["fold"]
    seed = cfg["seed"]
    return (
        Path(cfg["output_dir"])
        / f"{eid}_{method_tag}"
        / f"{ds}_{bb}"
        / f"fold{fold:02d}_seed{seed}"
    )


def run(cfg, force=False):
    run_dir = _run_dir(cfg)
    if not force and (run_dir / "metrics.json").exists():
        print(f"\n  SKIP {cfg['experiment_id']} — results already exist at {run_dir}/")
        print("  (use --force to re-run)")
        return None

    cfg["_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c_max = C_MAX[cfg["dataset"]]

    print(f"\n{'='*70}")
    print(
        f"  {cfg['experiment_id']} | {cfg['method']} | {cfg['dataset']} | "
        f"{cfg['backbone']} | fold {cfg['fold']} | split_seed {cfg['split_seed']}"
    )
    print(f"{'='*70}")

    model, head = build_model(
        cfg["method"], cfg["backbone"], c_max,
        proj_dim=cfg["proj_dim"], dropout=cfg["dropout"],
    )
    model = model.to(device)

    frozen, total = freeze_backbone(model.backbone, cfg["backbone"], cfg["freeze_layers"])
    cfg["_head_trainable_params"] = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    cfg["_model_trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone: {frozen/1e6:.1f}M/{total/1e6:.1f}M frozen ({frozen/total*100:.0f}%)")
    print(f"  Head trainable params: {cfg['_head_trainable_params']:,}")

    train_ds, val_ds, test_ds = build_datasets(
        cfg["dataset"],
        fold=cfg["fold"],
        n_folds=cfg["n_folds"],
        seed=cfg["split_seed"],
        img_size=cfg["img_size"],
    )
    print(f"  Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    tr_loader = DataLoader(
        train_ds, cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    va_loader = DataLoader(
        val_ds, cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    te_loader = DataLoader(
        test_ds, cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt_cls = torch.optim.Adam if cfg["optimizer"] == "adam" else torch.optim.AdamW
    optimizer = opt_cls([
        {"params": bb_params, "lr": cfg["lr_backbone"]},
        {"params": model.head.parameters(), "lr": cfg["lr_head"]},
    ], weight_decay=cfg["weight_decay"])

    if cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2,
            patience=cfg.get("scheduler_patience", 10),
        )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    class_weights = None
    if cfg["class_weighting"]:
        class_weights = compute_class_weights(train_ds, c_max, device)
        print(f"  Class weighting ON: {[f'G{i}={class_weights[i]:.2f}' for i in range(c_max+1)]}")
    else:
        print("  Class weighting OFF")

    best_mae, best_ep, best_state = float("inf"), 0, None
    patience_cnt = 0
    history = []

    for ep in range(1, cfg["epochs"] + 1):
        loss, det = train_one_epoch(
            model, tr_loader, optimizer, cfg, device, scaler, class_weights
        )
        val_metrics, _ = evaluate(model, va_loader, c_max, device)

        if cfg["scheduler"] == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_metrics["mae"])

        lr_now = optimizer.param_groups[0]["lr"]
        det_str = " ".join(f"{k}={v:.4f}" for k, v in det.items())
        print(
            f"  Ep {ep:3d} | loss={loss:.4f} ({det_str}) | "
            f"val MAE={val_metrics['mae']:.4f} QWK={val_metrics['qwk']:.4f} "
            f"F1={val_metrics['macro_f1']:.4f} | lr={lr_now:.2e}"
        )

        history.append({
            "epoch": ep,
            "train_loss": loss,
            "val_mae": val_metrics["mae"],
            "val_qwk": val_metrics["qwk"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": lr_now,
        })

        if val_metrics["mae"] < best_mae:
            best_mae, best_ep = val_metrics["mae"], ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"  Early stopping at epoch {ep}")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint; check the data loader and batch size.")

    print(f"  Best epoch: {best_ep} (val MAE={best_mae:.4f})")
    model.load_state_dict(best_state)
    model = model.to(device)

    # Validation is recomputed after restoring the selected checkpoint.
    val_final, _ = evaluate(model, va_loader, c_max, device)
    # Test is touched only here, once, after model selection is complete.
    test_metrics, test_raw = evaluate(model, te_loader, c_max, device)

    print("\n  Test results:")
    print(
        f"    MAE={test_metrics['mae']:.4f}  QWK={test_metrics['qwk']:.4f}  "
        f"ACC={test_metrics['acc']:.4f}  F1={test_metrics['macro_f1']:.4f}"
    )
    recall_str = "  ".join(
        f"G{i}={test_metrics[f'recall_g{i}']:.3f}" for i in range(c_max + 1)
    )
    print(f"    Recall: {recall_str}")

    save_results(
        run_dir, cfg, cfg["fold"], best_ep, best_state,
        val_final, test_metrics, test_raw, history,
    )
    print(f"\n  Saved to {run_dir}/")
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="FOROH validated experiment training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--fold", type=int, default=None, help="Override fold")
    parser.add_argument("--seed", type=int, default=None, help="Override training seed")
    parser.add_argument("--split-seed", type=int, default=None, help="Override split seed")
    parser.add_argument("--output-dir", default=None, help="Override output dir")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.fold is not None:
        cfg["fold"] = args.fold
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.split_seed is not None:
        cfg["split_seed"] = args.split_seed
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    run(cfg, force=args.force)


if __name__ == "__main__":
    main()
