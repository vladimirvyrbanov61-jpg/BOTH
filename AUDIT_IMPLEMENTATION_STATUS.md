# Audit Implementation Status

Date: 2026-06-11

The current schema-6 maintained pipeline implements the corrective audit plan.

## Corrected

- Signed accuracy and AUC edges replace absolute accuracy deviation as the
  scientific comparison metrics.
- Per-seed accuracy includes a two-sided exact random-label p-value conditional
  on the fixed class and prediction counts.
- Decision thresholds are selected on validation data only.
- Base samples and splits are held constant across round counts within a seed.
- SPECK classical estimates support repeated Monte Carlo runs and confidence
  intervals; top-k results are explicitly labeled non-exhaustive beam-search
  estimates rather than formal bounds.
- Classical CSV reuse validates cipher, delta, sample count, top-k setting,
  seed, repetition count, and schema.
- Input-difference comparisons enforce matching controlled configuration,
  training parameters, dependency environment, and source hash.
- Paired comparison tests include Holm-adjusted p-values.
- Paired tests distinguish machine-precision constants from genuinely varying
  seed differences instead of collapsing near-constant samples to p=0.
- SIMON and SPECK batched-key cache identities include every key in the batch.
- Classical trails are reconstructed with parent pointers.
- Manifests hash both the full maintained tree and execution-relevant source,
  and separate training completion from later post-processing.
- Manifests index SHA-256 hashes for exact caches, checkpoints, and TensorBoard
  event files used by each seed/cipher/round.
- Checkpoints include experiment metadata and validation-selected thresholds.
- Checkpoint paths include the run identifier, preventing silent overwrite between runs.
- CSV, manifest, checkpoint, and cache writes use temporary-file replacement.
- TensorBoard output is isolated by run identifier.
- New multi-seed runs reject nonempty result directories, and interruptions in
  the test gate or seed subprocesses are recorded in the manifest.
- Runtime overrides are validated after resolution.
- Unknown top-level, training, and classical configuration keys are rejected.
- Zero plaintext differences, malformed feature tensors, overlapping dataset
  splits, invalid direct training settings, and mismatched key batches are
  rejected before computation.
- The complete official SIMON and SPECK family parameter tables and
  known-answer fixtures include both 48/96 variants.
- Active imports use installable packages rather than `sys.path` mutation.
- Full cipher suites are included in the local experiment test gate.
- CI, `pyproject.toml`, dependency constraints, and artifact ignore rules exist.
- `requirements-lock.txt` is a complete freeze of the tested Python 3.13
  environment rather than a direct-dependency-only list.
- Generated-directory ignore rules are root-anchored so they do not hide
  maintained source packages such as `thesis/models`.
- Generated caches, checkpoints, TensorBoard events, timestamped results, and
  IDE state are excluded from Git while remaining available locally.
- Historical code is isolated under `Archive/`, excluded from package and test
  discovery, and explicitly documented as methodologically incomparable with
  the maintained thesis pipeline.
- Numbered generated archive snapshots are ignored and no longer tracked,
  while their local files remain available to the repository owner.

## Experiment Compatibility

Existing June 2026 neural raw CSVs can be re-aggregated into signed metrics
without retraining because they contain accuracy and AUC. Their original model
training, however, predates validation-threshold calibration, shared round
samples, checkpoint schema 2, and source-tree hashing.

For final thesis publication, run fresh baseline and `(0x0040, 0x0000)`
experiments with the current code and `--force-classical`. Older runs remain
historical evidence and should not be described as schema-6 experiments.

## Remaining Scientific Boundaries

- The project implements statistical distinguishers, not key recovery.
- SPECK characteristic tracking remains a sampled, top-k approximation.
- SIMON characteristic tracking also uses top-k beam pruning after exact
  one-round transition construction, so its multi-round result is an estimate.
- Neural signed edge and classical characteristic probability are different
  quantities and remain plotted in separate panels.
- The two selected input differences do not establish generalization to all
  differences.

These are explicit thesis-scope limitations rather than unresolved code errors.
