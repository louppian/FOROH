import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Dataset.dataset import LIMUCDataset, N_FOLDS, NUM_CLASSES, load_official_splits, make_transforms
from Model.model import SUPPORTED_BACKBONES, SUPPORTED_LOSSES, build_model
from losses import CDW_ALPHA, build_loss
from output import compute_confusion, compute_metrics, write_predictions, write_results_csv

DATA_ROOT = ROOT / "Dataset" / "LIMUC"
RESULT_ROOT = ROOT / "Result" / "01_LIMUC_Benchmark"
EPOCHS = 100
EARLY_STOP = 10
BATCH_SIZE = 32
NUM_WORKERS = 4
PROTOCOL = {"seed": 42, "optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.01, "scheduler": "cosine", "min_lr": 1e-6}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(outputs):
    if "logits" in outputs:
        return outputs["logits"].argmax(1)
    return outputs["score"].round().clamp(0, NUM_CLASSES - 1).long()


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        count += images.size(0)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, preds, names = [], [], []
    for images, batch_labels, batch_names in loader:
        outputs = model(images.to(device))
        batch_preds = predict(outputs).cpu().numpy()
        labels.extend(batch_labels.numpy().tolist())
        preds.extend(batch_preds.tolist())
        names.extend(batch_names)
    return np.asarray(labels), np.asarray(preds), names


def make_loader(samples, transform, shuffle=False):
    return DataLoader(LIMUCDataset(samples, transform), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weight_decay"])


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=PROTOCOL["min_lr"])


def train_fold(fold_id, samples, backbone, loss_name, output_dir, device):
    set_seed(PROTOCOL["seed"])
    train_tf, eval_tf, transform_info = make_transforms(backbone)
    train_loader = make_loader(samples["train"], train_tf, shuffle=True)
    val_loader = make_loader(samples["val"], eval_tf)
    test_loader = make_loader(samples["test"], eval_tf)
    model = build_model(backbone, loss_name).to(device)
    criterion = build_loss(loss_name)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    best_qwk, best_epoch, best_state, stale = -1e9, 0, None, 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_labels, val_preds, _ = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_preds)
        val_qwk = val_metrics["qwk"]
        scheduler.step()
        if val_qwk > best_qwk:
            best_qwk, best_epoch, stale = val_qwk, epoch, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"fold {fold_id:02d} epoch {epoch:03d} loss {train_loss:.4f} val_acc {val_metrics['accuracy']:.4f} val_qwk {val_qwk:.4f}")
        if stale >= EARLY_STOP:
            break
    model.load_state_dict(best_state)
    model.to(device)
    test_labels, test_preds, names = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_preds)
    checkpoint = {"model_state": best_state, "backbone": backbone, "loss": loss_name, "fold": fold_id, "seed": PROTOCOL["seed"], "best_epoch": best_epoch, "selection_metric": "val_qwk", "protocol": PROTOCOL, "transform": transform_info, "metrics": test_metrics}
    torch.save(checkpoint, output_dir / f"fold{fold_id}.pt")
    write_predictions(output_dir / f"fold{fold_id}_predictions.csv", names, test_labels, test_preds)
    return {"fold": fold_id, "best_epoch": best_epoch, **test_metrics, "confusion_matrix": compute_confusion(test_labels, test_preds).tolist()}


def write_config(path, backbone, loss_name):
    config = {"root": ".", "data_root": str(DATA_ROOT.relative_to(ROOT)), "result_root": str(RESULT_ROOT.relative_to(ROOT)), "backbone": backbone, "loss": loss_name, "protocol": PROTOCOL, "n_folds": N_FOLDS, "fold_source": "Dataset/LIMUC/cross_validation_folds_train_val_info", "split_seed": None, "selection_metric": "val_qwk", "epochs": EPOCHS, "batch_size": BATCH_SIZE, "cdw_alpha": CDW_ALPHA if loss_name == "cdw_ce" else None}
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=SUPPORTED_BACKBONES)
    parser.add_argument("--loss", required=True, choices=SUPPORTED_LOSSES)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = RESULT_ROOT / args.backbone / args.loss
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config(output_dir / "config.json", args.backbone, args.loss)
    splits = load_official_splits(DATA_ROOT)
    fold_results = []
    for fold_idx in range(N_FOLDS):
        fold_id = fold_idx + 1
        fold_samples = {**splits["folds"][fold_idx], "test": splits["test"]}
        print(f"\n{args.backbone} {args.loss} fold {fold_id}/10 train {len(fold_samples['train'])} val {len(fold_samples['val'])} test {len(fold_samples['test'])}")
        fold_results.append(train_fold(fold_id, fold_samples, args.backbone, args.loss, output_dir, device))
    write_results_csv(output_dir / "results.csv", fold_results)
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2)


if __name__ == "__main__":
    main()
