"""ml/metrics.py — Evaluation metrics, threshold tuning, and fault-breakdown analysis."""

from __future__ import annotations

from typing import Any

import numpy as np


def find_threshold_at_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    normal_scores = scores[labels == 0]
    if len(normal_scores) == 0:
        return float(scores.max())
    return float(np.quantile(normal_scores, 1.0 - target_fpr))


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    preds = (scores > threshold).astype(np.int8)
    y = labels.astype(np.int8)

    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    fnr = fn / max(n_pos, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(len(y), 1)

    try:
        auc = float(roc_auc_score(y, scores))
    except Exception:
        auc = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "accuracy": accuracy,
        "auc_roc": auc,
        "threshold": threshold,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_tp": tp,
        "n_fp": fp,
        "n_fn": fn,
        "n_tn": tn,
    }


def fault_breakdown(
    scores: np.ndarray,
    labels: np.ndarray,
    meta: list[dict[str, Any]],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    preds = (scores > threshold).astype(np.int8)
    faults = np.array([m.get("fault", "unknown") for m in meta])
    result: dict[str, dict[str, Any]] = {}

    for fault_name in np.unique(faults):
        mask = faults == fault_name
        n = int(mask.sum())
        if n == 0:
            continue
        fault_labels = labels[mask]
        fault_preds = preds[mask]
        fault_scores = scores[mask]

        if fault_name == "normal":
            correct = int(((fault_labels == 0) & (fault_preds == 0)).sum())
            recall = correct / max(n, 1)
        else:
            detected = int(((fault_labels == 1) & (fault_preds == 1)).sum())
            n_anom = int((fault_labels == 1).sum())
            recall = detected / max(n_anom, 1)
            correct = detected

        result[str(fault_name)] = {
            "n": n,
            "detected": correct,
            "recall": recall,
            "mean_score": float(fault_scores.mean()),
            "std_score": float(fault_scores.std()) if n > 1 else 0.0,
        }

    return result


def format_metrics_table(metrics: dict[str, float]) -> str:
    lines = ["  ── Metrics ────────────────────────────"]
    for key in ["auc_roc", "precision", "recall", "f1", "fpr", "fnr", "accuracy"]:
        v = metrics.get(key, float("nan"))
        lines.append(f"  {key:<12}: {v:.4f}")
    lines.append(f"  threshold   : {metrics.get('threshold', 0):.6f}")
    lines.append(
        f"  TP={metrics.get('n_tp', 0)}  FP={metrics.get('n_fp', 0)}  "
        f"FN={metrics.get('n_fn', 0)}  TN={metrics.get('n_tn', 0)}"
    )
    return "\n".join(lines)


def format_fault_table(breakdown: dict[str, dict[str, Any]]) -> str:
    lines = ["  ── Fault Breakdown ─────────────────────────────────────────"]
    header = f"  {'fault':<22}{'n':>6}  {'detected':>8}  {'recall':>8}  {'mean_score':>12}"
    lines.append(header)
    lines.append("  " + "-" * 62)
    for fault, info in sorted(breakdown.items()):
        lines.append(
            f"  {fault:<22}{info['n']:>6}  {info['detected']:>8}  "
            f"{info['recall']:>8.3f}  {info['mean_score']:>12.6f}"
        )
    return "\n".join(lines)
