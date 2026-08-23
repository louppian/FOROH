import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from output import METRIC_KEYS, compute_confusion, compute_metrics, write_results_csv
from Dataset.dataset import NUM_CLASSES


BACKBONE = "inception_v3"
LOSSES = ("ce", "cdw_ce", "foroh")
RESULT_ROOT = ROOT / "Result" / "01_LIMUC_Benchmark"
FOLD_RE = re.compile(r"fold(\d+)_predictions\.csv$")


def read_predictions(path):
    labels, preds, names = [], [], []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["sample_id"])
            labels.append(int(row["label"]))
            preds.append(int(row["prediction"]))
    return np.asarray(labels), np.asarray(preds), names


def find_prediction_files(loss_dir):
    files = {}
    for path in sorted(loss_dir.glob("fold*_predictions.csv")):
        match = FOLD_RE.match(path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def aggregate_loss(loss_dir):
    prediction_files = find_prediction_files(loss_dir)
    fold_results = []
    for fold in range(1, 11):
        path = prediction_files.get(fold)
        if path is None:
            continue
        labels, preds, _ = read_predictions(path)
        metrics = compute_metrics(labels, preds)
        fold_results.append({"fold": fold, "best_epoch": "", **metrics, "confusion_matrix": compute_confusion(labels, preds).tolist()})
    if fold_results:
        write_results_csv(loss_dir / "results.csv", fold_results)
        with (loss_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(fold_results, f, indent=2)
    return fold_results


def summarize_loss(loss, fold_results):
    metric_keys = list(METRIC_KEYS) + [f"recall_mayo{i}" for i in range(NUM_CLASSES)]
    row = {"loss": loss, "completed_folds": len(fold_results), "status": "complete" if len(fold_results) == 10 else "incomplete"}
    for key in metric_keys:
        values = [result[key] for result in fold_results]
        row[f"{key}_mean"] = float(np.mean(values)) if values else ""
        row[f"{key}_std"] = float(np.std(values)) if values else ""
    return row


def write_summary(path, rows):
    metric_keys = list(METRIC_KEYS) + [f"recall_mayo{i}" for i in range(NUM_CLASSES)]
    fieldnames = ["loss", "completed_folds", "status"]
    for key in metric_keys:
        fieldnames.extend([f"{key}_mean", f"{key}_std"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.4f}" if isinstance(value, float) else value for key, value in row.items()})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    return parser.parse_args()


def main():
    args = parse_args()
    backbone_dir = Path(args.result_root) / BACKBONE
    rows = []
    for loss in LOSSES:
        loss_dir = backbone_dir / loss
        if not loss_dir.exists():
            rows.append(summarize_loss(loss, []))
            print(f"{BACKBONE} {loss}: missing")
            continue
        fold_results = aggregate_loss(loss_dir)
        rows.append(summarize_loss(loss, fold_results))
        print(f"{BACKBONE} {loss}: {len(fold_results)}/10 folds")
    summary_path = backbone_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary(summary_path, rows)
    print(f"wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
