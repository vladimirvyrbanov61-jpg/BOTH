"""Minimal integration: tiny dataset + short training + CSV/checkpoint schema."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis.config.loader import config_path_for_profile, load_config
from thesis.eval.round_sweep import CSV_FIELDS, run_round_sweep
from thesis.models.train import TrainConfig

pytest.importorskip("torch")


@pytest.fixture
def smoke_dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "data": tmp_path / "cache",
        "models": tmp_path / "models",
        "results": tmp_path / "results",
    }


def test_round_sweep_smoke_simon_r3(smoke_dirs: dict[str, Path]) -> None:
    cfg_path = config_path_for_profile("quick")
    cfg = load_config(cfg_path)
    train_overrides = {**cfg.get("training", {}), "epochs": 1, "patience": 1, "batch_size": 64}

    rows, results_path = run_round_sweep(
        cfg_path,
        ciphers=["simon"],
        rounds_list=[3],
        n_samples=200,
        data_dir=smoke_dirs["data"],
        model_dir=smoke_dirs["models"],
        results_dir=smoke_dirs["results"],
        force_regen=True,
        fresh_csv=True,
        training_overrides=train_overrides,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["cipher"] == "simon"
    assert row["rounds"] == 3
    assert row["split"] == "test"
    assert 0.0 <= row["accuracy"] <= 1.0
    assert row["advantage_abs"] >= 0.0
    assert results_path.exists()
    assert results_path.parent == smoke_dirs["results"]

    csv_path = results_path / "simon_round_sweep.csv"
    assert csv_path.exists()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDS
        data_rows = list(reader)
    assert len(data_rows) == 1

    aggregate_path = results_path / "simon_aggregate.csv"
    assert aggregate_path.exists()
    with open(aggregate_path, encoding="utf-8") as f:
        aggregate_rows = list(csv.DictReader(f))
    assert len(aggregate_rows) == 1
    assert aggregate_rows[0]["n_seeds"] == "1"
    assert float(aggregate_rows[0]["accuracy_std"]) == 0.0
    assert (results_path / "simon_accuracy.png").exists()

    ckpt = smoke_dirs["models"] / "seed_1" / "simon" / "R3.pt"
    assert ckpt.exists()
    assert ckpt.stat().st_size > 0


def test_train_config_from_dict_channels() -> None:
    cfg = TrainConfig.from_dict({"channels": [16, 32], "epochs": 2})
    assert cfg.channels == (16, 32)
    assert cfg.epochs == 2
