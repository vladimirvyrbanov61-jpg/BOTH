# 02 — System Architecture

## 2.1 Architectural principles

| Principle | How it manifests |
|-----------|------------------|
| **Layered dependencies** | `simon.py` has no ML deps; `simon3264` depends only on `simon`; `ml` depends on `simon3264`; `experiments` depends on `ml` + `simon3264` |
| **Additive assignment layer** | `experiments/` was added without refactoring core cipher or `ml/train.py` |
| **File-based state** | No database; caches and artifacts on disk under `data/`, `models/`, `results/` |
| **Configuration as data** | YAML dataclasses (`ml/config.py`, `experiments/config.py`) + CLI overrides |
| **One-class anomaly training** | Autoencoders fit on **normal rows only**; anomalies evaluated at test time |
| **Reproducibility** | Seeded RNGs; dataset cache keyed by config hash |

**Why layers instead of a monolith?** Cipher correctness can be tested in isolation; ML can evolve without touching round functions; assignment experiments can be optional for users who only need anomaly detection.

---

## 2.2 Layer diagram

```mermaid
flowchart TB
  subgraph L0 [Layer 0 — Cipher primitive]
    simon[simon.py\nSimon encrypt/decrypt\nexpand_key batch API]
  end

  subgraph L1 [Layer 1 — SIMON 32/64 profile]
    cipher[simon3264/cipher.py\nSimon3264 encrypt_rounds]
    enc[encoding.py]
    faults[faults.py]
    dataset[dataset.py]
    io[io.py]
    feat3264[features.py]
    trace[trace.py]
    simon --> cipher
    cipher --> enc
    cipher --> faults
    cipher --> dataset
    dataset --> io
  end

  subgraph L2 [Layer 2 — ML anomaly pipeline]
    mlcfg[ml/config.py]
    mldata[ml/data.py]
    mlfeat[ml/features.py\n42-dim]
    mlmodels[ml/models.py]
    mlmetrics[ml/metrics.py]
    train[ml/train.py]
    score[ml/score.py]
    dataset --> mldata
    mlcfg --> mldata
    mldata --> mlfeat --> mlmodels
    mlmodels --> train
    mlmodels --> score
    mlmetrics --> train
    mlmetrics --> score
  end

  subgraph L3 [Layer 3 — Assignment experiments]
    expcfg[experiments/config.py]
    sweep[round_sweep.py Track A]
    dist[distinguisher_* Track B]
    runall[run_all.py]
    report[reporting.py]
    mlmodels --> sweep
    cipher --> dist
    expcfg --> sweep
    expcfg --> dist
    sweep --> report
    dist --> report
    runall --> sweep
    runall --> dist
    runall --> report
  end
```

---

## 2.3 ML anomaly pipeline — data flow

```mermaid
sequenceDiagram
  participant CFG as configs/default.yaml
  participant DATA as ml/data.py
  participant DS as simon3264.dataset
  participant FEAT as ml/features.py
  participant TRAIN as ml/train.py
  participant MODEL as models/
  participant SCORE as ml/score.py

  CFG->>DATA: load_config
  DATA->>DS: generate_labeled_dataset (on cache miss)
  DS-->>DATA: blocks labels meta
  DATA->>FEAT: build_feature_matrix
  DATA->>DATA: stratified_split → DataSplit
  TRAIN->>DATA: generate_or_load_dataset
  TRAIN->>MODEL: fit TorchAutoencoder on X_train_normal
  TRAIN->>MODEL: thresholds.json from val FPR
  SCORE->>MODEL: load + score_samples
  SCORE-->>SCORE: evaluate or score-file
```

**Why cache datasets?** Generating tens of thousands of encrypted blocks with fault injection is CPU-heavy. `ml/data.py` hashes the experiment config and stores `data/dataset_<hash>.npz` plus precomputed features.

---

## 2.4 Assignment pipeline — data flow

```mermaid
sequenceDiagram
  participant RUN as experiments/run_all.py
  participant TRAIN as ml/train.py
  participant SW as round_sweep.py
  participant DIST as run_distinguisher.py
  participant REP as reporting.py

  RUN->>TRAIN: subprocess if torch_autoencoder.pt missing
  RUN->>SW: load AE + threshold sweep rounds
  SW-->>RUN: results/round_sweep/*
  RUN->>DIST: per-round distinguisher train/eval
  DIST-->>RUN: results/distinguisher/*
  RUN->>REP: merge CSV/PNG → ASSIGNMENT_SUMMARY.md
```

---

## 2.5 Dual ML paradigms (design choice)

| Paradigm | Module | Training | Inference question |
|----------|--------|----------|-------------------|
| **One-class anomaly** | `ml/models.py` | Normals only (AE/IF) | “Does this block look like training normals?” |
| **Binary distinguisher** | `experiments/distinguisher_model.py` | Real pairs vs random pairs | “Is this pair from reduced-round Simon or random?” |

They share the cipher oracle but **must not be conflated** in reports: Track A uses **42-dim block features** + reconstruction error; Track B default uses **32-bit XOR-of-pair** features.

---

## 2.6 Dependency tree (runtime packages)

```mermaid
flowchart TD
  numpy[numpy]
  yaml[pyyaml]
  sklearn[scikit-learn]
  torch[torch optional local]
  mpl[matplotlib]
  pytest[pytest dev]

  simon[simon.py] --> numpy
  s3264[simon3264] --> simon
  ml[ml] --> s3264
  ml --> sklearn
  ml --> torch
  exp[experiments] --> ml
  exp --> s3264
  exp --> mpl
  tests[pytest suite] --> pytest
```

See [09 — Configuration reference](09-configuration-reference.md) for `requirements.txt` vs `requirements-colab.txt`.

---

## 2.7 Execution model

- **No long-running server.** All entry points are **CLI scripts** invoked as `python path/to/script.py`.
- **Working directory:** Scripts insert the repository root into `sys.path` (`_REPO` / `_REPO_ROOT` pattern). Run commands from the repo root.
- **No environment variables** are read for configuration (by design). Paths come from YAML `paths:` sections.
- **GPU:** Optional; `TorchAutoencoder` and `NeuralDistinguisher` use `device: auto` → CUDA if available.

---

## 2.8 Extension points

| Extension | Hook |
|-----------|------|
| New fault type | `simon3264/faults.py` + `_apply_fault` in `dataset.py` |
| New block features | `ml/features.py` `build_feature_matrix` |
| New anomaly model | `ml/models.py` + `train.py` / `score.py` |
| New distinguisher features | `experiments/distinguisher_data.py` `pair_to_features` |
| New round-sweep metric | `experiments/round_sweep.py` `evaluate_round` |

---

[← Introduction](01-introduction-and-scope.md) · [Next: Repository layout →](03-repository-layout.md)
