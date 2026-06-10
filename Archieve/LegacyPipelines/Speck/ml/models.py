"""ml/models.py — Anomaly detection models (sklearn, NumPy AE, PyTorch AE)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ml.config import AutoencoderConfig, IsoForestConfig, TorchAutoencoderConfig


class IsolationForestModel:
    def __init__(self, cfg: Optional[IsoForestConfig] = None) -> None:
        self.cfg = cfg or IsoForestConfig()
        self._model: Any = None

    def fit(self, X_normal: np.ndarray) -> "IsolationForestModel":
        from sklearn.ensemble import IsolationForest

        self._model = IsolationForest(
            n_estimators=self.cfg.n_estimators,
            max_samples=self.cfg.max_samples,
            contamination=self.cfg.contamination,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )
        self._model.fit(X_normal)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return -self._model.score_samples(X)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return (self.score_samples(X) > threshold).astype(np.int8)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"cfg": self.cfg, "model": self._model}, fh)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IsolationForestModel":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls(cfg=data["cfg"])
        obj._model = data["model"]
        return obj


# ---------------------------------------------------------------------------
# NumPy autoencoder (no PyTorch)
# ---------------------------------------------------------------------------

class _Layer:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "relu",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.standard_normal((in_dim, out_dim)) * scale
        self.b = np.zeros(out_dim)
        self.activation = activation
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._X = X
        self._Z = X @ self.W + self.b
        if self.activation == "relu":
            self._A = np.maximum(0.0, self._Z)
        else:
            self._A = self._Z
        return self._A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            dZ = dA * (self._Z > 0).astype(dA.dtype)
        else:
            dZ = dA
        N = self._X.shape[0]
        self._dW = self._X.T @ dZ / N
        self._db = dZ.mean(axis=0)
        return dZ @ self.W.T

    def adam_update(self, lr: float, t: int) -> None:
        for param, grad, m, v in [
            (self.W, self._dW, self.mW, self.vW),
            (self.b, self._db, self.mb, self.vb),
        ]:
            m[:] = 0.9 * m + 0.1 * grad
            v[:] = 0.999 * v + 0.001 * grad ** 2
            m_hat = m / (1.0 - 0.9 ** t)
            v_hat = v / (1.0 - 0.999 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class NumpyAutoencoder:
    def __init__(self, cfg: Optional[AutoencoderConfig] = None) -> None:
        self.cfg = cfg or AutoencoderConfig()
        self.layers: list[_Layer] = []
        self.input_dim = 0
        self._train_loss_history: list[float] = []
        self._val_loss_history: list[float] = []

    def _build(self, input_dim: int, rng: np.random.Generator) -> None:
        dims = [input_dim] + self.cfg.hidden_dims + [self.cfg.latent_dim]
        enc = [_Layer(dims[i], dims[i + 1], "relu", rng) for i in range(len(dims) - 1)]
        dec_dims = list(reversed(dims))
        dec = []
        for i in range(len(dec_dims) - 1):
            act = "relu" if i < len(dec_dims) - 2 else "linear"
            dec.append(_Layer(dec_dims[i], dec_dims[i + 1], act, rng))
        self.layers = enc + dec
        self.input_dim = input_dim

    def _forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _backward(self, X: np.ndarray, X_hat: np.ndarray) -> float:
        grad = -2.0 * (X - X_hat) / X.shape[0]
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return float(np.mean((X - X_hat) ** 2))

    def _normalize_fit(self, X: np.ndarray) -> np.ndarray:
        self._xmin = X.min(axis=0)
        self._xmax = X.max(axis=0)
        span = self._xmax - self._xmin
        span[span == 0] = 1.0
        return (X - self._xmin) / span

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        span = self._xmax - self._xmin
        span[span == 0] = 1.0
        return (X.astype(np.float64) - self._xmin) / span

    def fit(
        self,
        X_normal: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "NumpyAutoencoder":
        rng = np.random.default_rng(self.cfg.seed)
        X_norm = self._normalize_fit(X_normal.astype(np.float64))
        self._build(X_norm.shape[1], rng)

        best_val = np.inf
        patience = 0
        best_w = None
        t = 0
        N = X_norm.shape[0]
        bs = self.cfg.batch_size

        for _ in range(self.cfg.epochs):
            perm = rng.permutation(N)
            losses = []
            for start in range(0, N, bs):
                batch = X_norm[perm[start : start + bs]]
                if len(batch) == 0:
                    continue
                hat = self._forward(batch)
                losses.append(self._backward(batch, hat))
                t += 1
                for layer in self.layers:
                    layer.adam_update(self.cfg.lr, t)
            self._train_loss_history.append(float(np.mean(losses)) if losses else 0.0)

            if X_val is not None:
                vn = self._normalize(X_val)
                if y_val is not None:
                    m = y_val == 0
                    vn = vn[m] if m.any() else vn
                if len(vn):
                    vhat = self._forward(vn)
                    vl = float(np.mean((vn - vhat) ** 2))
                    self._val_loss_history.append(vl)
                    if vl < best_val:
                        best_val = vl
                        patience = 0
                        best_w = [(l.W.copy(), l.b.copy()) for l in self.layers]
                    else:
                        patience += 1
                        if patience >= self.cfg.patience and best_w:
                            for layer, (W, b) in zip(self.layers, best_w):
                                layer.W[:] = W
                                layer.b[:] = b
                            break
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xn = self._normalize(X)
        return np.mean((Xn - self._forward(Xn)) ** 2, axis=1)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "cfg": self.cfg,
                    "input_dim": self.input_dim,
                    "xmin": self._xmin,
                    "xmax": self._xmax,
                    "weights": [(l.W.copy(), l.b.copy()) for l in self.layers],
                    "train_history": self._train_loss_history,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NumpyAutoencoder":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls(cfg=data["cfg"])
        obj._xmin, obj._xmax = data["xmin"], data["xmax"]
        obj._build(data["input_dim"], np.random.default_rng(0))
        for layer, (W, b) in zip(obj.layers, data["weights"]):
            layer.W[:] = W
            layer.b[:] = b
        return obj


# ---------------------------------------------------------------------------
# PyTorch autoencoder (Google Colab / GPU)
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class TorchAutoencoder:
    """MLP autoencoder in PyTorch — primary neural-network track for Colab."""

    def __init__(self, cfg: Optional[TorchAutoencoderConfig] = None) -> None:
        self.cfg = cfg or TorchAutoencoderConfig()
        self._net: Any = None
        self._device: str = "cpu"
        self._xmin: Optional[np.ndarray] = None
        self._xmax: Optional[np.ndarray] = None
        self._train_loss_history: list[float] = []

    def _build_net(self, input_dim: int) -> Any:
        import torch
        import torch.nn as nn

        dims = [input_dim] + self.cfg.hidden_dims + [self.cfg.latent_dim]
        layers_enc: list[Any] = []
        for i in range(len(dims) - 1):
            layers_enc.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers_enc.append(nn.ReLU())
                if self.cfg.dropout > 0:
                    layers_enc.append(nn.Dropout(self.cfg.dropout))

        dec_dims = list(reversed(dims))
        layers_dec: list[Any] = []
        for i in range(len(dec_dims) - 1):
            layers_dec.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                layers_dec.append(nn.ReLU())
                if self.cfg.dropout > 0:
                    layers_dec.append(nn.Dropout(self.cfg.dropout))

        class _AE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(*layers_enc)
                self.decoder = nn.Sequential(*layers_dec)

            def forward(self, x: Any) -> Any:
                return self.decoder(self.encoder(x))

        return _AE()

    def _normalize_fit(self, X: np.ndarray) -> np.ndarray:
        self._xmin = X.min(axis=0)
        self._xmax = X.max(axis=0)
        span = self._xmax - self._xmin
        span[span == 0] = 1.0
        return ((X - self._xmin) / span).astype(np.float32)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        span = self._xmax - self._xmin
        span[span == 0] = 1.0
        return ((X.astype(np.float64) - self._xmin) / span).astype(np.float32)

    def fit(
        self,
        X_normal: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TorchAutoencoder":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self._device = _resolve_device(self.cfg.device)
        Xn = self._normalize_fit(X_normal.astype(np.float64))
        self._net = self._build_net(Xn.shape[1]).to(self._device)
        opt = torch.optim.Adam(
            self._net.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        loss_fn = nn.MSELoss()

        X_t = torch.from_numpy(Xn).to(self._device)
        N = X_t.shape[0]
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
                batch = X_t[idx]
                opt.zero_grad()
                recon = self._net(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            self._train_loss_history.append(float(np.mean(losses)))

            if X_val is not None:
                self._net.eval()
                vn = self._normalize(X_val)
                if y_val is not None:
                    m = y_val == 0
                    vn = vn[m] if m.any() else vn
                if len(vn):
                    with torch.no_grad():
                        vt = torch.from_numpy(vn).to(self._device)
                        vhat = self._net(vt)
                        vl = float(loss_fn(vhat, vt).item())
                    if vl < best_val:
                        best_val = vl
                        patience = 0
                        best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                    else:
                        patience += 1
                        if patience >= self.cfg.patience and best_state:
                            self._net.load_state_dict(best_state)
                            break

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("Call fit() first.")
        self._net.eval()
        Xn = self._normalize(X)
        with torch.no_grad():
            xt = torch.from_numpy(Xn).to(self._device)
            xh = self._net(xt)
            err = ((xt - xh) ** 2).mean(dim=1).cpu().numpy()
        return err.astype(np.float64)

    def save(self, path: Union[str, Path]) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg,
                "state_dict": self._net.state_dict(),
                "xmin": self._xmin,
                "xmax": self._xmax,
                "input_dim": int(self._xmin.shape[0]),
                "train_history": self._train_loss_history,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TorchAutoencoder":
        import torch

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        obj = cls(cfg=data["cfg"])
        obj._xmin = data["xmin"]
        obj._xmax = data["xmax"]
        obj._device = _resolve_device(obj.cfg.device)
        obj._net = obj._build_net(data["input_dim"]).to(obj._device)
        obj._net.load_state_dict(data["state_dict"])
        obj._net.eval()
        return obj


AnyModel = Union[IsolationForestModel, NumpyAutoencoder, TorchAutoencoder]
