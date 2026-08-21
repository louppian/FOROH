"""Evaluation metrics for ordinal regression."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    recall_score,
)


def compute_metrics(labels, preds, c_max):
    Y = np.asarray(labels, dtype=int)
    P = np.asarray(preds, dtype=int)
    metrics = {
        "mae": float(mean_absolute_error(Y, P)),
        "qwk": float(cohen_kappa_score(Y, P, weights="quadratic")),
        "acc": float(accuracy_score(Y, P)),
        "macro_f1": float(f1_score(Y, P, average="macro", zero_division=0)),
        "off_by_1": float(np.mean(np.abs(P - Y) <= 1)),
    }
    per_class = recall_score(
        Y, P, average=None, labels=list(range(c_max + 1)), zero_division=0,
    )
    for i, r in enumerate(per_class):
        metrics[f"recall_g{i}"] = float(r)
    return metrics
