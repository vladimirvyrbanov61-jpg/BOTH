"""Classification metrics for neural distinguisher evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute accuracy, AUC-ROC, TPR, TNR, and Gohr-style advantage."""
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_pred = (y_score >= threshold).astype(np.int64)

    acc = float(np.mean(y_pred == y_true))

    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, y_score))

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    advantage_abs = abs(acc - 0.5)
    advantage_edge = abs(2.0 * acc - 1.0)
    youden_j = tpr + tnr - 1.0

    return {
        "accuracy": acc,
        "auc_roc": auc,
        "tpr": float(tpr),
        "tnr": float(tnr),
        "advantage_abs": float(advantage_abs),
        "advantage_edge": float(advantage_edge),
        "youden_j": float(youden_j),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def metrics_row(
    *,
    cipher: str,
    rounds: int,
    split: str,
    n_samples: int,
    metrics: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    row = {
        "cipher": cipher,
        "rounds": rounds,
        "split": split,
        "n_samples": n_samples,
        **{k: metrics[k] for k in ("accuracy", "auc_roc", "tpr", "tnr", "advantage_abs", "advantage_edge", "youden_j")},
    }
    if seed is not None:
        row["seed"] = seed
    return row
