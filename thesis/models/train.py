"""Training loop for thesis CNN distinguisher with validation AUC early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from thesis.eval.metrics import classification_metrics
from thesis.models.cnn_distinguisher import CnnDistinguisher, build_model


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    device: str = "auto"
    seed: int = 1
    channels: tuple[int, ...] = (32, 64, 128)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        if "channels" in filtered and isinstance(filtered["channels"], list):
            filtered["channels"] = tuple(filtered["channels"])
        return cls(**filtered)


def _index_subset(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X[idx], y[idx]


def _batch_iter(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(y)
    order = rng.permutation(n)
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, n, batch_size):
        sl = order[start : start + batch_size]
        batches.append((X[sl], y[sl]))
    return batches


def _forward_loss(
    model: CnnDistinguisher,
    xb: np.ndarray,
    yb: np.ndarray,
    device: str,
    loss_fn: Any,
) -> Any:
    import torch

    xt = torch.from_numpy(xb).to(device)
    yt = torch.from_numpy(yb.astype(np.float32)).to(device)
    logits = model.forward_logits(xt).squeeze(-1)
    return loss_fn(logits, yt)


def _predict_scores(model: CnnDistinguisher, X: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        scores: list[np.ndarray] = []
        bs = 4096
        for start in range(0, len(X), bs):
            xb = X[start : start + bs]
            xt = torch.from_numpy(xb.astype(np.float32)).to(device)
            prob = model.predict_proba(xt)
            scores.append(prob.cpu().numpy())
        return np.concatenate(scores, axis=0)


def train_distinguisher(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    cipher: str,
    rounds: int,
    model_dir: Path | str,
    cfg: Optional[TrainConfig] = None,
    log_dir: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Train CNN with early stopping on validation AUC; save best checkpoint.
    
    Parameters
    ----------
    log_dir : Path | str, optional
        If provided, log metrics to TensorBoard at this directory.
        Layout: log_dir/seed_{seed}/{cipher}_R{rounds}/
    """
    import torch
    import torch.nn as nn

    train_cfg = cfg or TrainConfig()
    device = _resolve_device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    model_dir = Path(model_dir)
    # New: seed-aware checkpoint path (prevents overwrites)
    seed_model_dir = model_dir / f"seed_{train_cfg.seed}" / cipher
    seed_model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = seed_model_dir / f"R{rounds}.pt"

    # Initialize TensorBoard if log_dir provided
    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(log_dir)
        tb_log_dir = log_dir / f"seed_{train_cfg.seed}" / f"{cipher}_R{rounds}"
        writer = SummaryWriter(log_dir=str(tb_log_dir))

    X_tr, y_tr = _index_subset(X, y, splits["train"])
    X_va, y_va = _index_subset(X, y, splits["val"])

    model = build_model(train_cfg.channels).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(train_cfg.seed)
    best_auc = -1.0
    patience_ctr = 0
    best_state: Optional[dict[str, Any]] = None
    history: list[dict[str, float]] = []

    for epoch in range(train_cfg.epochs):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in _batch_iter(X_tr, y_tr, train_cfg.batch_size, rng):
            opt.zero_grad()
            loss = _forward_loss(model, xb, yb, device, loss_fn)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))

        val_scores = _predict_scores(model, X_va, device)
        val_m = classification_metrics(y_va, val_scores)
        train_loss_mean = float(np.mean(epoch_losses))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss_mean,
                "val_auc": val_m["auc_roc"],
                "val_accuracy": val_m["accuracy"],
                "val_advantage_abs": val_m["advantage_abs"],
            }
        )

        # Log to TensorBoard if enabled
        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss_mean, epoch)
            writer.add_scalar("AUC/Validation", val_m["auc_roc"], epoch)
            writer.add_scalar("Accuracy/Validation", val_m["accuracy"], epoch)
            writer.add_scalar("AdvantageAbs/Validation", val_m["advantage_abs"], epoch)

        if val_m["auc_roc"] > best_auc:
            best_auc = val_m["auc_roc"]
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= train_cfg.patience and best_state is not None:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Close TensorBoard writer safely
    if writer is not None:
        writer.close()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "cipher": cipher,
            "rounds": rounds,
            "train_config": train_cfg.__dict__,
            "best_val_auc": best_auc,
            "history": history,
        },
        ckpt_path,
    )

    return {
        "model": model,
        "checkpoint_path": ckpt_path,
        "best_val_auc": best_auc,
        "history": history,
        "device": device,
    }


def load_distinguisher(
    path: Path | str,
    *,
    device: str = "auto",
) -> CnnDistinguisher:
    import torch

    dev = _resolve_device(device)
    try:
        data = torch.load(path, map_location=dev, weights_only=False)
    except TypeError:
        data = torch.load(path, map_location=dev)
    channels = tuple(data.get("train_config", {}).get("channels", (32, 64, 128)))
    model = build_model(channels).to(dev)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model
