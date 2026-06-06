"""Tests for deterministic aggregation across experiment seeds."""

from __future__ import annotations

import pytest

from thesis.eval.aggregate import aggregate_rows


def test_aggregate_rows_reports_sample_uncertainty() -> None:
    base = {
        "cipher": "simon",
        "rounds": 3,
        "split": "test",
        "n_samples": 100,
        "auc_roc": 0.7,
        "tpr": 0.7,
        "tnr": 0.7,
        "advantage_abs": 0.2,
        "advantage_edge": 0.4,
        "youden_j": 0.4,
    }
    rows = [
        {**base, "seed": 1, "accuracy": 0.6},
        {**base, "seed": 2, "accuracy": 0.8},
        {**base, "seed": 99, "split": "val", "accuracy": 1.0},
    ]

    aggregate = aggregate_rows(rows)

    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["n_seeds"] == 2
    assert row["seeds"] == "1;2"
    assert row["n_samples_total"] == 200
    assert row["accuracy_mean"] == pytest.approx(0.7)
    assert row["accuracy_std"] == pytest.approx(2**0.5 / 10)
    assert row["accuracy_ci95_low"] < row["accuracy_mean"] < row["accuracy_ci95_high"]
