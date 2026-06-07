"""Tests for controlled input-difference experiment comparison."""

from __future__ import annotations

import pytest

from thesis.config.loader import config_path_for_profile, load_config
from thesis.eval.compare_experiments import build_comparison_rows


def _aggregate(mean: float, low: float, high: float) -> dict[str, float]:
    return {
        "auc_roc_mean": mean,
        "auc_roc_ci95_low": low,
        "auc_roc_ci95_high": high,
        "advantage_abs_mean": mean - 0.5,
        "advantage_abs_ci95_low": low - 0.5,
        "advantage_abs_ci95_high": high - 0.5,
    }


def test_build_comparison_rows_detects_interval_overlap() -> None:
    baseline = {"simon": {3: _aggregate(0.8, 0.75, 0.85)}}
    candidate = {"simon": {3: _aggregate(0.9, 0.86, 0.94)}}

    rows = build_comparison_rows(baseline, candidate, metrics=("auc_roc",))

    assert len(rows) == 1
    assert rows[0]["mean_difference"] == pytest.approx(0.1)
    assert rows[0]["ci95_overlap"] is False


def test_delta_profile_only_changes_controlled_fields() -> None:
    baseline = load_config(config_path_for_profile("full"))
    candidate = load_config("thesis/config/thesis_delta_0040.yaml")
    changed = {
        key
        for key in set(baseline) | set(candidate)
        if baseline.get(key) != candidate.get(key)
    }

    assert changed == {"input_delta", "model_dir", "results_dir"}
    assert candidate["input_delta"] == [0x0040, 0]
