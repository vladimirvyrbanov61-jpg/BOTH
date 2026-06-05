"""PyTorch binary classifier for Speck neural distinguisher experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from experiments.config import DistinguisherConfig


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class NeuralDistinguisher:
    """MLP binary classifier: real Speck pair vs random pair."""

    def __init__(self, cfg: Optional[DistinguisherConfig] = None) -> None:
        self.cfg = cfg or DistinguisherConfig()
        self._net = None
        self._device = "cpu"
        self.input_dim = 0
        self._train_loss_history: list[float] = []

    def _build(self, input_dim: int) -> None:
        import torch.nn as nn

        layers: list[nn.Module] = []
        dims = [input_dim] + self.cfg.hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if self.cfg.dropout > 0:
                layers.append(nn.Dropout(self.cfg.dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self._net = nn.Sequential(*layers).to(self._device)
        self.input_dim = input_dim

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "NeuralDistinguisher":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.cfg.seed)
        self._device = _resolve_device(self.cfg.device)
        X = X_train.astype(np.float32)
        y = y_train.astype(np.float32).reshape(-1, 1)
        self._build(X.shape[1])

        opt = torch.optim.Adam(
            self._net.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(X).to(self._device)
        yt = torch.from_numpy(y).to(self._device)

        N = Xt.shape[0]
        bs = self.cfg.batch_size
        best_val = np.inf
        patience = 0
        best_state = None

        for epoch in range(self.cfg.epochs):
            self._net.train()
            perm = torch.randperm(N, device=self._device)
            losses = []
            for start in range(0, N, bs):
                idx = perm[start : start + bs]
                xb = Xt[idx]
                yb = yt[idx]
                opt.zero_grad()
                logits = self._net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            self._train_loss_history.append(float(np.mean(losses)))

            if X_val is not None and y_val is not None and len(X_val) > 0:
                self._net.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(X_val.astype(np.float32)).to(self._device)
                    yv = torch.from_numpy(y_val.astype(np.float32).reshape(-1, 1)).to(
                        self._device
                    )
                    vl = float(loss_fn(self._net(xv), yv).item())
                if vl < best_val:
                    best_val = vl
                    patience = 0
                    best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                else:
                    patience += 1
                    if patience >= self.cfg.patience and best_state is not None:
                        self._net.load_state_dict(best_state)
                        break
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        self._net.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X.astype(np.float32)).to(self._device)
            logits = self._net(xt).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Probability of class 1 (real Speck pair)."""
        return self.predict_proba(X)

    def save(self, path: Union[str, Path]) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg,
                "state_dict": self._net.state_dict(),
                "input_dim": self.input_dim,
                "train_history": self._train_loss_history,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NeuralDistinguisher":
        import torch

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        obj = cls(cfg=data["cfg"])
        obj._device = _resolve_device(obj.cfg.device)
        obj.input_dim = data["input_dim"]
        obj._train_loss_history = data.get("train_history", [])
        obj._build(obj.input_dim)
        obj._net.load_state_dict(data["state_dict"])
        obj._net.eval()
        return obj
