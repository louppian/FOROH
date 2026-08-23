import csv

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error, recall_score

from Dataset.dataset import NUM_CLASSES


METRIC_KEYS = ("accuracy", "macro_f1", "qwk", "mae", "remission_kappa", "remission_f1")


def compute_metrics(labels, preds):
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    rem_true = (labels >= 2).astype(int)
    rem_pred = (preds >= 2).astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "qwk": cohen_kappa_score(labels, preds, weights="quadratic"),
        "mae": mean_absolute_error(labels, preds),
        "remission_kappa": cohen_kappa_score(rem_true, rem_pred),
        "remission_f1": f1_score(rem_true, rem_pred, zero_division=0),
    }
    recalls = recall_score(labels, preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    for idx, value in enumerate(recalls):
        metrics[f"recall_mayo{idx}"] = value
    return metrics


def compute_confusion(labels, preds):
    return confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))


def write_predictions(path, names, labels, preds):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "prediction"])
        writer.writerows(zip(names, labels, preds))


def write_results_csv(path, fold_results):
    metric_keys = list(METRIC_KEYS) + [f"recall_mayo{i}" for i in range(NUM_CLASSES)]
    rows = []
    for result in fold_results:
        rows.append(["fold", result["fold"], result["best_epoch"], *[result[k] for k in metric_keys]])
    for label, fn in (("mean", np.mean), ("std", np.std)):
        rows.append([label, "", "", *[fn([r[k] for r in fold_results]) for k in metric_keys]])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "fold", "best_epoch", *metric_keys])
        for row in rows:
            writer.writerow([f"{x:.4f}" if isinstance(x, (float, np.floating)) else x for x in row])
