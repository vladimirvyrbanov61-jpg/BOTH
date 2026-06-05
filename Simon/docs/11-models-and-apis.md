# 11 — Models & API Contracts

## 11.1 Model catalog

| Class | Module | File | Paradigm |
|-------|--------|------|----------|
| `IsolationForestModel` | `ml/models.py` | `iso_forest.pkl` | One-class (normal fit) |
| `NumpyAutoencoder` | `ml/models.py` | `autoencoder.pkl` | One-class AE |
| `TorchAutoencoder` | `ml/models.py` | `torch_autoencoder.pt` | One-class AE (primary) |
| `NeuralDistinguisher` | `experiments/distinguisher_model.py` | `distinguisher_r{r}.pt` | Binary classifier |

---

## 11.2 Shared anomaly detector contract

All three anomaly models implement:

```python
model.fit(X_normal, X_val=None, y_val=None)  # IF: only X_normal
scores = model.score_samples(X)              # shape (N,), higher = more anomalous
model.save(path)
model = Model.load(path)
```

### `IsolationForestModel`

- Wraps `sklearn.ensemble.IsolationForest`
- `score_samples` = `-sklearn.score_samples` (invert so higher = more anomalous)

### `NumpyAutoencoder`

- MLP encoder–decoder, manual Adam-like updates
- Trains on min-max normalized normals
- Score = mean squared reconstruction error

### `TorchAutoencoder`

- PyTorch `nn.Sequential` encoder + decoder
- Early stopping on validation **normal** reconstruction loss
- `device`: `auto` → CUDA if available
- Checkpoint stores `xmin`, `xmax`, `cfg`, `state_dict`, `input_dim`

```python
from ml.models import TorchAutoencoder
model = TorchAutoencoder.load("models/torch_autoencoder.pt")
scores = model.score_samples(X)  # float64, shape (N,)
```

---

## 11.3 `NeuralDistinguisher` contract

```python
from experiments.distinguisher_model import NeuralDistinguisher

model = NeuralDistinguisher(cfg)
model.fit(X_train, y_train, X_val, y_val)   # y: 1=real Simon pair, 0=random
proba = model.predict_proba(X)              # P(class 1)
model.score_samples(X)                      # same as predict_proba
model.save("models/distinguisher_r8.pt")
model = NeuralDistinguisher.load(path)
```

Architecture: `Linear → ReLU → Dropout → … → Linear(1)` + `BCEWithLogitsLoss`.

---

## 11.4 Prediction semantics

| Model | Threshold rule | Positive prediction |
|-------|----------------|---------------------|
| Anomaly detectors | `score > threshold` | Anomaly |
| Distinguisher | `proba >= 0.5` | Real Simon pair |

---

## 11.5 Type and shape requirements

| Tensor | dtype | Shape |
|--------|-------|-------|
| `blocks` | uint16 | `(N, 2)` |
| `X` (features) | float64/float32 | `(N, F)` |
| `y` (labels) | int8 | `(N,)` |

---

## 11.6 Choosing a model

| Scenario | Recommendation |
|----------|----------------|
| Colab / assignment | `TorchAutoencoder` |
| CI without PyTorch | `IsolationForestModel` or `NumpyAutoencoder` |
| Maximum interpretability | IF + fault breakdown tables |
| Reduced-round pair classification | `NeuralDistinguisher` per round |

---

[← Features](10-feature-engineering.md) · [Next: CLI reference →](12-cli-reference.md)
