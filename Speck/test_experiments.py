"""Smoke tests for experiments/ layer (fast, no GPU required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.config import CryptanalysisConfig, DistinguisherConfig, load_cryptanalysis_config
from experiments.distinguisher_data import (
    generate_distinguisher_dataset,
    pair_to_features,
    stratified_split_indices,
)
from experiments.distinguisher_model import NeuralDistinguisher
from experiments.metrics import binary_classification_metrics
from experiments.round_sweep import generate_round_dataset
from experiments.run_distinguisher import evaluate_distinguisher
from speck3264.cipher import ROUNDS, Speck3264


def test_cryptanalysis_config_loads():
    cfg = load_cryptanalysis_config(_REPO / "configs" / "cryptanalysis.yaml")
    assert isinstance(cfg, CryptanalysisConfig)
    assert 22 in cfg.round_sweep.round_values
    assert cfg.distinguisher.feature_mode == "xor_bits"


def test_distinguisher_data_shapes():
    dc = DistinguisherConfig(
        n_samples_per_round=50,
        feature_mode="xor_bits",
        seed=0,
    )
    rng = np.random.default_rng(0)
    X, y = generate_distinguisher_dataset(8, dc, rng)
    assert X.shape == (50, 32)
    assert y.shape == (50,)
    assert set(np.unique(y)) == {0, 1}
    assert int((y == 1).sum()) == 25


def test_pair_to_features_modes():
    c0 = np.array([[0x1111, 0x2222]], dtype=np.uint16)
    c1 = np.array([[0x3333, 0x4444]], dtype=np.uint16)
    assert pair_to_features(c0, c1, "xor_bits").shape == (1, 32)
    assert pair_to_features(c0, c1, "concat_bits").shape == (1, 64)
    assert pair_to_features(c0, c1, "ml_features").shape[1] == 42


def test_stratified_split():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    splits = stratified_split_indices(y, seed=1)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(y)


def test_binary_classification_metrics():
    y = np.array([1, 1, 0, 0], dtype=np.int8)
    p = np.array([1, 0, 0, 1], dtype=np.int8)
    m = binary_classification_metrics(y, p)
    assert m["tpr"] == 0.5
    assert m["tnr"] == 0.5
    assert m["accuracy"] == 0.5


def test_distinguisher_train_smoke():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 16)).astype(np.float32)
    y = np.array([0] * 40 + [1] * 40, dtype=np.int8)
    dc = DistinguisherConfig(
        hidden_dims=[16, 8],
        epochs=3,
        batch_size=16,
        patience=2,
        seed=0,
    )
    model = NeuralDistinguisher(dc)
    model.fit(X[:60], y[:60], X[60:], y[60:])
    proba = model.predict_proba(X)
    assert proba.shape == (80,)
    test_metrics = evaluate_distinguisher(model, X[60:], y[60:])
    assert "tpr" in test_metrics and "tnr" in test_metrics


def test_round_sweep_smoke():
    cipher = Speck3264()
    rng = np.random.default_rng(0)
    blocks, labels = generate_round_dataset(cipher, 8, 20, rng)
    assert blocks.shape == (20, 2)
    assert int(labels[0]) == 1
    blocks22, labels22 = generate_round_dataset(cipher, ROUNDS, 20, rng)
    assert int(labels22[0]) == 0


def test_run_all_imports():
    import experiments.reporting  # noqa: F401
    import experiments.run_all  # noqa: F401
    import experiments.round_sweep  # noqa: F401
    import experiments.run_distinguisher  # noqa: F401


def test_round_sweep_csv_smoke(tmp_path):
    pytest.importorskip("torch")
    from ml.models import TorchAutoencoder
    from experiments.round_sweep import run_round_sweep

    cfg = load_cryptanalysis_config(_REPO / "configs" / "cryptanalysis.yaml")
    cfg.round_sweep.n_samples_per_round = 20
    cfg.round_sweep.results_dir = str(tmp_path / "round_sweep")

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 42))
    model = TorchAutoencoder()
    model.fit(X)
    model.save(model_dir / "torch_autoencoder.pt")
    scores = model.score_samples(X[:10])
    thresh = float(np.quantile(scores, 0.99))
    (model_dir / "thresholds.json").write_text(
        json.dumps({"torch_autoencoder": thresh}), encoding="utf-8"
    )

    run_round_sweep(cfg, quick_rounds=[8, 22], model_dir=model_dir)
    csv_path = tmp_path / "round_sweep" / "round_sweep.csv"
    assert csv_path.exists()
