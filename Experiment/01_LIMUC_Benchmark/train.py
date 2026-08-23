import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Dataset.dataset import LIMUCDataset, N_FOLDS, NUM_CLASSES, load_official_splits, make_transforms
from Model.model import SUPPORTED_BACKBONES, SUPPORTED_LOSSES, build_model
from losses import CDW_ALPHA, build_loss
from output import compute_confusion, compute_metrics, write_predictions, write_results_csv

DATA_ROOT = ROOT / "Dataset" / "LIMUC"
RESULT_ROOT = ROOT / "Result" / "01_LIMUC_Benchmark"
EPOCHS = 200
BATCH_SIZE = 32
NUM_WORKERS = 4
PROTOCOLS = {
    "inception_v3": {"name": "polat", "seed": 35, "optimizer": "adam", "lr": 2e-4, "weight_decay": 0.0, "scheduler": "plateau", "factor": 0.2, "patience": 15, "early_stop": 25, "weighted_sampler": True},
    "resnet18": {"name": "polat", "seed": 35, "optimizer": "adam", "lr": 2e-4, "weight_decay": 0.0, "scheduler": "plateau", "factor": 0.2, "patience": 15, "early_stop": 25, "weighted_sampler": True},
    "coatnet_2": {"name": "nie_zhang", "seed": 0, "optimizer": "adamw", "lr": 5e-4, "weight_decay": 0.005, "scheduler": "cosine", "warmup_epochs": 5, "warmup_lr": 1e-6, "min_lr": 1e-5, "early_stop": None, "weighted_sampler": False},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(outputs):
    if "logits" in outputs:
        return outputs["logits"].argmax(1)
    return outputs["score"].round().clamp(0, 3).long()


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


def make_sampler(samples):
    counts = np.bincount([label for _, label in samples], minlength=NUM_CLASSES).astype(np.float64)
    weights = [1.0 / max(counts[label], 1.0) for _, label in samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_loader(samples, transform, shuffle=False, sampler=None):
    return DataLoader(LIMUCDataset(samples, transform), batch_size=BATCH_SIZE, shuffle=shuffle, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)


def build_optimizer(model, protocol):
    params = model.parameters()
    if protocol["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=protocol["lr"], weight_decay=protocol["weight_decay"])
    return torch.optim.AdamW(params, lr=protocol["lr"], weight_decay=protocol["weight_decay"])


def build_scheduler(optimizer, protocol):
    if protocol["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=protocol["factor"], patience=protocol["patience"], threshold=0.0001)
    base_lr = protocol["lr"]
    warmup_lr = protocol["warmup_lr"]
    min_lr = protocol["min_lr"]
    warmup_epochs = protocol["warmup_epochs"]

    def lr_scale(epoch):
        if epoch < warmup_epochs:
            return (warmup_lr + (base_lr - warmup_lr) * (epoch + 1) / warmup_epochs) / base_lr
        progress = min((epoch + 1 - warmup_epochs) / max(EPOCHS - warmup_epochs, 1), 1.0)
        cosine_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
        return cosine_lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


def train_fold(fold_id, samples, backbone, loss_name, output_dir, device):
    protocol = PROTOCOLS[backbone]
    set_seed(protocol["seed"])
    train_tf, eval_tf, transform_info = make_transforms(backbone, samples["train"])
    sampler = make_sampler(samples["train"]) if protocol["weighted_sampler"] else None
    train_loader = make_loader(samples["train"], train_tf, shuffle=sampler is None, sampler=sampler)
    val_loader = make_loader(samples["val"], eval_tf)
    test_loader = make_loader(samples["test"], eval_tf)
    model = build_model(backbone, loss_name).to(device)
    criterion = build_loss(loss_name)
    optimizer = build_optimizer(model, protocol)
    scheduler = build_scheduler(optimizer, protocol)
    best_acc, best_epoch, stale, best_state = -1e9, 0, 0, None
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_labels, val_preds, _ = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_preds)
        val_acc = val_metrics["accuracy"]
        if protocol["scheduler"] == "plateau":
            scheduler.step(val_acc)
        else:
            scheduler.step()
        if val_acc > best_acc:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"fold {fold_id:02d} epoch {epoch:03d} loss {train_loss:.4f} val_acc {val_acc:.4f} val_qwk {val_metrics['qwk']:.4f}")
        if protocol["early_stop"] is not None and stale >= protocol["early_stop"]:
            break
    model.load_state_dict(best_state)
    model.to(device)
    test_labels, test_preds, names = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_preds)
    checkpoint = {"model_state": best_state, "backbone": backbone, "loss": loss_name, "fold": fold_id, "seed": protocol["seed"], "best_epoch": best_epoch, "selection_metric": "val_accuracy", "protocol": protocol["name"], "transform": transform_info, "metrics": test_metrics}
    torch.save(checkpoint, output_dir / f"fold{fold_id}.pt")
    write_predictions(output_dir / f"fold{fold_id}_predictions.csv", names, test_labels, test_preds)
    return {"fold": fold_id, "best_epoch": best_epoch, **test_metrics, "confusion_matrix": compute_confusion(test_labels, test_preds).tolist()}


def write_config(path, backbone, loss_name):
    protocol = PROTOCOLS[backbone]
    config = {"root": ".", "data_root": str(DATA_ROOT.relative_to(ROOT)), "result_root": str(RESULT_ROOT.relative_to(ROOT)), "backbone": backbone, "loss": loss_name, "protocol": protocol, "n_folds": N_FOLDS, "fold_source": "Dataset/LIMUC/cross_validation_folds_train_val_info", "split_seed": None, "selection_metric": "val_accuracy", "epochs": EPOCHS, "batch_size": BATCH_SIZE, "cdw_alpha": CDW_ALPHA if loss_name == "cdw_ce" else None}
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
