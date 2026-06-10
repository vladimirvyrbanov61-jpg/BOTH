"""Unit tests for thesis CNN distinguisher and training loss."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
import torch.nn as nn

from thesis.models.cnn_distinguisher import (
    INPUT_BITS,
    WORD_CHANNELS,
    WORD_LENGTH,
    BitPairReshape,
    CnnDistinguisher,
    build_model,
)
from thesis.models.train import _resolve_device, load_distinguisher


def test_bit_pair_reshape():
    x = torch.randn(8, INPUT_BITS)
    out = BitPairReshape()(x)
    assert out.shape == (8, WORD_CHANNELS, WORD_LENGTH)


def test_cnn_forward_flat_and_structured():
    model = build_model()
    flat = torch.randn(16, INPUT_BITS)
    logits_flat = model.forward_logits(flat)
    assert logits_flat.shape == (16, 1)

    structured = flat.view(16, WORD_CHANNELS, WORD_LENGTH)
    logits_struct = model.forward_logits(structured)
    torch.testing.assert_close(logits_flat, logits_struct)

    probs = model.predict_proba(flat)
    assert probs.shape == (16,)
    assert torch.all((probs >= 0) & (probs <= 1))


def test_bce_with_logits_scalar_loss():
    model = build_model()
    x = torch.randint(0, 2, (32, INPUT_BITS), dtype=torch.float32)
    y = torch.randint(0, 2, (32,), dtype=torch.float32)
    logits = model.forward_logits(x).squeeze(-1)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_train_config_from_dict():
    from thesis.models.train import TrainConfig

    cfg = TrainConfig.from_dict({"epochs": 5, "channels": [16, 32, 64]})
    assert cfg.epochs == 5
    assert cfg.channels == (16, 32, 64)


def test_checkpoint_load_uses_tensor_only_mode(tmp_path):
    model = build_model((16, 32))
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "schema_version": 2,
            "state_dict": model.state_dict(),
            "train_config": {"channels": (16, 32)},
            "experiment": {"input_delta": [1, 0]},
            "decision_threshold": 0.42,
        },
        checkpoint,
    )

    loaded = load_distinguisher(checkpoint, device="cpu")
    assert loaded.count_parameters() == model.count_parameters()
    assert loaded.decision_threshold == pytest.approx(0.42)
    assert loaded.checkpoint_metadata["input_delta"] == [1, 0]


def test_explicit_cuda_request_fails_clearly_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        _resolve_device("cuda")
