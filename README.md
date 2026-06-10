# BOTH: Neural Differential Cryptanalysis

This repository contains the active thesis pipeline for neural differential
cryptanalysis of SIMON32/64 and SPECK32/64.

## Active Architecture

The supported implementation is:

- `ciphers/`: shared encoding, sampling, and cipher registry.
- `Simon/simon.py`, `Simon/simon3264/cipher.py`: SIMON primitive and profile.
- `Speck/speck.py`, `Speck/speck3264/cipher.py`: SPECK primitive and profile.
- `thesis/`: dataset generation, CNN training, classical analysis, statistics,
  plots, and experiment manifests.
- `scripts/multi_seed_sweep.py`: ten-seed experiment orchestrator.
- `tests/`: the test gate executed before a normal sweep.

Historical experiments and superseded scripts are stored under `Archieve/`.
They are not supported runtime code and must not be cited as the thesis model.

## Dataset Definition

Each neural-distinguisher dataset is balanced:

- Class 1: plaintext pairs use one fixed configured 32-bit input difference.
  Every pair receives a newly sampled random 64-bit key, then both plaintexts
  are encrypted under that same key for the selected number of rounds.
- Class 0: two independently sampled random 32-bit blocks.
- Model input: `bits(C0) || bits(C1)`, represented as 64 binary values.

Plaintexts and keys are not stored in the thesis cache. Dataset generation and
the stratified train/validation/test split are derived from the experiment seed.
Within one seed, the same sampled base pairs and split indices are reused across
round counts so the round curve isolates the effect of additional rounds.

The baseline difference `(0x0001, 0x0000)` is a simple single-bit control
difference. The second experiment `(0x0040, 0x0000)` is a predeclared
input-sensitivity comparison, not a difference selected after inspecting its
results. These choices are experimental controls rather than claims of
optimality.

## Neural Model

The supported distinguisher is a 1D CNN:

- Input reshape: `(batch, 64)` to four 16-bit word channels.
- Three Conv1D blocks with channels 32, 64, and 128.
- Batch normalization and ReLU after every convolution.
- Adaptive average pooling and one binary output logit.
- Adam optimization with binary cross-entropy loss.
- Early stopping on validation ROC AUC.
- A decision threshold selected on validation data by maximizing Youden's J.

The headline accuracy statistic is the signed edge `2 * (accuracy - 0.5)`.
Unlike `abs(accuracy - 0.5)`, its expectation under random guessing is zero.
Every seed row also reports an exact two-sided binomial null p-value. Across
seeds these p-values are combined with Fisher's method; they are not averaged.
A small positive absolute deviation alone is not evidence of a distinguisher.

## Classical Comparison

SIMON uses an exact DDT for its nonlinear round function. SPECK transition rows
are estimated with repeated Monte Carlo sampling and report confidence
intervals across repetitions. Both use top-k beam search to track the best
retained differential characteristic over the requested rounds.

The classical value is a non-exhaustive beam-search estimate, not a formal
upper or lower bound. For SPECK it additionally contains Monte Carlo sampling
uncertainty. It is not numerically equivalent to neural classification
advantage. Comparison plots show both trends but must not be interpreted as a
direct magnitude test.
Classical CSV reuse is accepted only when the cipher, input difference, sample
count, top-k setting, seed, repetition count, and schema all match.

## Installation

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-thesis.txt
```

Editable package installation is also supported:

```powershell
python -m pip install -e ".[test]"
```

## Commands

Quick pipeline with the test gate:

```powershell
python -m thesis.run_thesis --profile quick --fresh-csv
```

Full ten-seed baseline:

```powershell
python -m scripts.multi_seed_sweep --seeds 1 2 3 4 5 6 7 8 9 10 --profile full --log-dir runs/thesis --fresh-csv --force-regen --force-classical
```

Second input-difference experiment:

```powershell
python -m scripts.multi_seed_sweep --seeds 1 2 3 4 5 6 7 8 9 10 --config thesis/config/thesis_delta_0040.yaml --log-dir runs/thesis_delta_0040 --fresh-csv --force-regen --force-classical
```

TensorBoard:

```powershell
python -m tensorboard.main --logdir runs/thesis
```

Checkpoint files are local trusted artifacts. The active loader uses PyTorch's
restricted `weights_only=True` mode and should not load files from untrusted
sources.

## Artifact Policy

Dataset caches, model checkpoints, and TensorBoard events are regenerable local
artifacts and are excluded from Git. Publication CSVs, figures, and manifests
may be retained under `results/`. Manifests preserve the training completion
time separately from later post-processing, hash the maintained source tree,
and index exact dataset caches, checkpoints, and TensorBoard event files.

To compare an already completed timestamped run without retraining:

```powershell
python -m thesis.run_thesis --profile full --skip-sweep --results-dir results/thesis/run_YYYYMMDD_HHMMSS_microseconds
```

Runs created before the schema-5 hardening pass can still be inspected, but
their dirty-worktree provenance and original absolute-advantage metric mean
they should not be represented as fresh outputs of the current implementation.
