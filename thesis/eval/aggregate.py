"""Aggregate neural-distinguisher metrics across independent random seeds."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

METRICS = (
    "accuracy",
    "auc_roc",
    "tpr",
    "tnr",
    "advantage_abs",
    "advantage_edge",
    "youden_j",
)

AGGREGATE_FIELDS = [
    "cipher",
    "rounds",
    "split",
    "n_seeds",
    "seeds",
    "n_samples_total",
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
        return list(csv.DictReader(handle))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    sem = std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    margin = 1.96 * sem
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
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
            for suffix, value in _summary(values).items():
                aggregate[f"{metric}_{suffix}"] = value
        output.append(aggregate)
    return output


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_csv(raw_path: Path, output_path: Path | None = None) -> tuple[list[dict[str, Any]], Path]:
    rows = aggregate_rows(read_metric_rows(raw_path))
    target = output_path or raw_path.with_name(raw_path.name.replace("_multi_seed_raw", "_aggregate"))
    write_aggregate_csv(target, rows)
    return rows, target
