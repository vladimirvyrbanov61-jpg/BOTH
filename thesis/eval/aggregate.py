"""Aggregate neural-distinguisher metrics across independent random seeds."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from thesis.eval.metrics import accuracy_null_p_value

METRICS = (
    "accuracy",
    "auc_roc",
    "tpr",
    "tnr",
    "accuracy_advantage",
    "advantage_edge",
    "auc_advantage",
    "youden_j",
)

METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "accuracy": (0.0, 1.0),
    "auc_roc": (0.0, 1.0),
    "tpr": (0.0, 1.0),
    "tnr": (0.0, 1.0),
    "accuracy_advantage": (-0.5, 0.5),
    "advantage_edge": (-1.0, 1.0),
    "auc_advantage": (-1.0, 1.0),
    "youden_j": (-1.0, 1.0),
}

_T_CRITICAL_95 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)

AGGREGATE_FIELDS = [
    "cipher",
    "rounds",
    "split",
    "n_seeds",
    "seeds",
    "n_samples_total",
    "accuracy_null_p_value_fisher",
    *[
        f"{metric}_{suffix}"
        for metric in METRICS
        for suffix in ("mean", "std", "sem", "ci95_low", "ci95_high", "min", "max")
    ],
]


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("accuracy") not in (None, ""):
            accuracy = float(row["accuracy"])
            row.setdefault("accuracy_advantage", str(accuracy - 0.5))
            row["accuracy_advantage"] = row.get("accuracy_advantage") or str(
                accuracy - 0.5
            )
            row["advantage_edge"] = row.get("advantage_edge") or str(
                2.0 * (accuracy - 0.5)
            )
            if "advantage_abs" in row:
                row.pop("advantage_abs", None)
            if row.get("accuracy_null_p_value") in (None, ""):
                n_samples = int(row["n_samples"])
                correct = int(round(accuracy * n_samples))
                row["accuracy_null_p_value"] = str(
                    accuracy_null_p_value(correct, n_samples)
                )
        if row.get("auc_roc") not in (None, ""):
            auc = float(row["auc_roc"])
            row["auc_advantage"] = row.get("auc_advantage") or str(
                2.0 * (auc - 0.5)
            )
    return rows


def student_t_critical_95(n_samples: int) -> float:
    """Two-sided 95% Student-t critical value for a sample mean."""
    if n_samples <= 1:
        return 0.0
    degrees_of_freedom = n_samples - 1
    if degrees_of_freedom <= len(_T_CRITICAL_95):
        return _T_CRITICAL_95[degrees_of_freedom - 1]
    return 1.96


def summarize_values(
    values: list[float],
    *,
    bounds: tuple[float, float] | None = None,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sample")
    if not np.isfinite(array).all():
        raise ValueError("summary values must be finite")
    if bounds is not None and not (
        ((bounds[0] <= array) & (array <= bounds[1])).all()
    ):
        raise ValueError(f"summary values must be within {bounds}")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    sem = std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    margin = student_t_critical_95(len(array)) * sem
    ci_low = mean - margin
    ci_high = mean + margin
    if bounds is not None:
        ci_low = max(bounds[0], ci_low)
        ci_high = min(bounds[1], ci_high)
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    split: str = "test",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("split", "test")) != split:
            continue
        key = (str(row["cipher"]), int(row["rounds"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (cipher, rounds), group in sorted(grouped.items()):
        seeds = sorted({int(row["seed"]) for row in group})
        if len(seeds) != len(group):
            raise ValueError(
                f"Duplicate seed rows for {cipher} round {rounds}; "
                "start with --fresh-csv or remove duplicate runs before aggregation."
            )
        aggregate: dict[str, Any] = {
            "cipher": cipher,
            "rounds": rounds,
            "split": split,
            "n_seeds": len(seeds),
            "seeds": ";".join(str(seed) for seed in seeds),
            "n_samples_total": sum(int(row["n_samples"]) for row in group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            if not values:
                continue
            for suffix, value in summarize_values(
                values,
                bounds=METRIC_BOUNDS[metric],
            ).items():
                aggregate[f"{metric}_{suffix}"] = value
        p_values = [
            float(row["accuracy_null_p_value"])
            for row in group
            if row.get("accuracy_null_p_value") not in (None, "")
        ]
        if p_values:
            if not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in p_values
            ):
                raise ValueError(
                    f"Invalid null p-value for {cipher} round {rounds}"
                )
            from scipy.stats import combine_pvalues

            aggregate["accuracy_null_p_value_fisher"] = float(
                combine_pvalues(p_values, method="fisher").pvalue
            )
        output.append(aggregate)
    return output


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=AGGREGATE_FIELDS,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def aggregate_csv(raw_path: Path, output_path: Path | None = None) -> tuple[list[dict[str, Any]], Path]:
    rows = aggregate_rows(read_metric_rows(raw_path))
    target = output_path or raw_path.with_name(raw_path.name.replace("_multi_seed_raw", "_aggregate"))
    write_aggregate_csv(target, rows)
    return rows, target
