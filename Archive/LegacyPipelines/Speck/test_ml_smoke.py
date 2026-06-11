"""Smoke tests for ml/ pipeline (fast)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ml.config import load_config
from ml.data import DataSplit, generate_or_load_dataset
from ml.features import build_feature_matrix
from ml.models import IsolationForestModel, NumpyAutoencoder


@pytest.fixture
def small_cfg():
    from ml.config import ExperimentConfig, DataConfig

    cfg = ExperimentConfig()
    cfg.data = DataConfig(n_samples=200, anomaly_fraction=0.2, seed=0)
    cfg.autoencoder.epochs = 3
    cfg.isolation_forest.n_estimators = 20
    return cfg


def test_feature_matrix_shape():
    blocks = np.random.default_rng(0).integers(0, 0x10000, size=(5, 2), dtype=np.uint16)
    X = build_feature_matrix(blocks)
    assert X.shape[0] == 5 and X.shape[1] == 42


def test_dataset_and_iso_forest(small_cfg, tmp_path):
    small_cfg.paths.data_dir = str(tmp_path / "data")
    ds = generate_or_load_dataset(small_cfg, force_regen=True)
    split = DataSplit(ds)
    assert split.n_features == 42
    m = IsolationForestModel(small_cfg.isolation_forest)
    m.fit(split.X_train_normal)
    scores = m.score_samples(split.X_val)
    assert scores.shape == (split.n_val,)


def test_numpy_ae_roundtrip(small_cfg, tmp_path):
    blocks = np.random.default_rng(1).integers(0, 0x10000, size=(80, 2), dtype=np.uint16)
    X = build_feature_matrix(blocks)
    m = NumpyAutoencoder(small_cfg.autoencoder)
    m.fit(X[:60])
    s = m.score_samples(X[60:])
    assert s.shape == (20,)
    p = tmp_path / "ae.pkl"
    m.save(p)
    m2 = NumpyAutoencoder.load(p)
    np.testing.assert_allclose(m2.score_samples(X[60:]), s, rtol=1e-5)


def test_load_yaml_config():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.data.n_samples == 10_000
