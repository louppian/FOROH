"""Evaluation metrics split from root 3_train.py for E01."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    recall_score,
)


def compute_metrics(labels, preds, c_max=3):
    y = np.asarray(labels).astype(int)
    p = np.asarray(preds).astype(int)
    metrics = {
        "mae": mean_absolute_error(y, p),
        "qwk": cohen_kappa_score(y, p, weights="quadratic"),
        "acc": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "off_by_1": float(np.mean(np.abs(p - y) <= 1)),
    }
    recalls = recall_score(
        y, p, average=None, labels=list(range(c_max + 1)), zero_division=0
    )
    for i, value in enumerate(recalls):
        metrics[f"recall_g{i}"] = value
    return metrics


def print_metrics(metrics, c_max=3, prefix=""):
    print(
        f"{prefix}MAE={metrics['mae']:.4f}  QWK={metrics['qwk']:.4f}  "
        f"ACC={metrics['acc']:.4f}  F1={metrics['macro_f1']:.4f}  "
        f"Off-by-1={metrics['off_by_1']:.4f}"
    )
    recall_str = "  ".join(
        f"G{i}={metrics[f'recall_g{i}']:.3f}" for i in range(c_max + 1)
    )
    print(f"{prefix}Recall: {recall_str}")
