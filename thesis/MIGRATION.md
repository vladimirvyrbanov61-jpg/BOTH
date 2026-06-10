# Maintained Thesis Architecture

The historical migration is complete. The supported execution path is:

`thesis/config` -> `thesis/data` -> `ciphers/registry.py` -> cipher profile ->
`thesis/models` -> `thesis/eval` -> `thesis/classical`.

Legacy fault-injection, autoencoder, recovery, and duplicate experiment code is
stored under `Archieve/LegacyPipelines/` and is not imported by the maintained
pipeline.

## Current Outputs

Each timestamped run contains:

- `{cipher}_multi_seed_raw.csv`: one test row per seed and round.
- `{cipher}_aggregate.csv`: seed means, sample dispersion, 95% Student-t
  intervals, and Fisher-combined null p-values.
- `{cipher}_classical_bounds.csv`: legacy filename containing schema-3
  non-exhaustive characteristic estimates and provenance.
- Neural, classical-estimate, and comparison figures.
- `external_artifacts.json`: SHA-256 records for caches, checkpoints, and
  TensorBoard events.
- `manifest.json`: resolved configuration, environment, source hash, progress,
  lifecycle timestamps, and artifact inventory.

Checkpoints are stored under:

`models/<experiment>/<run_id>/seed_<seed>/<cipher>/R<rounds>.pt`

The quick and full profiles both sweep rounds 3-10. The quick profile reduces
sample counts and epochs only; it is intended for smoke validation, not thesis
claims.
