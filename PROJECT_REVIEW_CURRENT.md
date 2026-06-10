# BOTH Project Review

Review date: 2026-06-10

The maintained project implements neural statistical distinguishers for
SIMON32/64 and SPECK32/64 over rounds 3-10, multi-seed uncertainty analysis,
and classical differential-characteristic comparison.

## Current Assessment

- Cipher primitives pass official known-answer vectors and complete local
  primitive suites, including all ten official SIMON and SPECK family
  parameter variants.
- Neural inputs contain ciphertext pairs only; plaintexts and keys are not
  persisted in caches.
- The configured plaintext difference must be nonzero, preventing a trivial
  zero-difference experiment.
- Train, validation, and test indices are validated as a complete, disjoint
  partition before model construction.
- Dataset generation, splitting, initialization, minibatch order, Torch, and
  CUDA execution are seeded. Strict deterministic Torch algorithms are enabled.
- Accuracy thresholds are selected on validation data; final metrics use the
  held-out test split.
- Signed accuracy/AUC edges replace absolute deviations. Exact per-seed null
  p-values are combined across seeds with Fisher's method.
- Aggregate plots report 95% Student-t intervals across distinct seeds.
- Input-difference comparisons are paired by seed and use Holm-adjusted
  multiple-testing p-values.
- SIMON one-round transitions use an exact nonlinear-function DDT. SPECK rows
  use repeated Monte Carlo sampling. Multi-round tracking for both ciphers is
  top-k beam search and is explicitly an estimate, not a formal bound.
- Manifests record source/configuration/environment provenance and hashes for
  exact caches, checkpoints, TensorBoard events, and result artifacts.

## Scientific Limits

The project evaluates statistical distinguishing, not key recovery. Neural
edge and classical characteristic probability are different quantities and
must not be compared as a direct ratio. The two configured input differences
test controlled sensitivity, not generalization over all differences.

## Operational Status

The normal local gate includes maintained unit/integration tests, classical
tests, and complete SIMON/SPECK primitive suites. GitHub CI runs the same
scientific and primitive coverage. Generated caches, models, logs, and IDE
state remain local and outside version control. Historical result snapshots
already committed under `results/` are retained as legacy evidence and are
not treated as outputs of the current schema-5 pipeline.
