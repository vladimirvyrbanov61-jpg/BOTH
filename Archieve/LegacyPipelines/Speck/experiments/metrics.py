"""Classification metrics for distinguisher experiments."""

from __future__ import annotations

import numpy as np


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """TPR/TNR/accuracy for labels 1=positive (real Speck), 0=negative (random)."""
    y = y_true.astype(np.int8)
    p = y_pred.astype(np.int8)
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    tpr = tp / n_pos
    tnr = tn / n_neg
    accuracy = (tp + tn) / max(len(y), 1)
    return {
        "accuracy": float(accuracy),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
