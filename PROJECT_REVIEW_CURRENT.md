# BOTH Project Review

Review date: 2026-06-10

## Executive assessment

The maintained thesis pipeline is coherent and runnable for its stated purpose:
training multi-seed neural distinguishers for SIMON32/64 and SPECK32/64,
estimating classical differential-characteristic bounds, and plotting both
quantities over rounds 3-10.

The codebase previously mixed this pipeline with older fault-injection,
autoencoder, and per-cipher experiment trees. Those implementations are now
isolated under `Archieve/LegacyPipelines/` and are not imported by the active
pipeline.

No active arbitrary-code deserialization, plaintext/key cache leakage, or
known-answer cipher failure was found. The main remaining risks are
methodological: the SPECK classical calculation is sampled and pruned, the
neural and classical curves are not the same mathematical quantity, and ten
seeds do not remove all experiment-selection uncertainty.

## External review reassessment

| Review criticism | Current verdict | Evidence or correction |
| --- | --- | --- |
| SPECK classical analysis is incomplete | Outdated | `thesis/classical/ddt_speck.py` implements sampled round transitions and `thesis/classical/characteristic.py` tracks trails over rounds. |
| Duplicate Simon/Speck ML pipelines create ambiguity | Valid, corrected | Historical `ml/`, `experiments/`, `scripts/`, and fault-analysis utilities were moved to `Archieve/LegacyPipelines/`. |
| Plaintexts or keys may leak into training features/cache | Not found in maintained pipeline | `thesis/data/generator.py` exposes only paired ciphertext bits and labels. `thesis/data/cache.py` accepts only `X`, `y`, and round metadata and validates the schema. |
| Classical sample counts are too small | Misapplied to the full profile | Full SPECK uses 1,000,000 samples per expanded transition row. SIMON's nonlinear-function DDT is exact. Quick-profile numbers are smoke-test settings, not thesis estimates. |
| Configuration validation is weak | Valid, corrected | `thesis/config/loader.py` now validates ciphers, rounds, delta width, sample parity, split feasibility, training values, paths, and classical parameters before a run starts. |
| Random seeds are uncontrolled | Incorrect for the maintained pipeline | Python, NumPy, Torch, CUDA, data generation, splitting, and loaders are seeded. CuDNN deterministic mode is enabled. |
| CNN design is undocumented | Outdated | Root `README.md`, configuration files, and `thesis/models/cnn_distinguisher.py` document the input layout and architecture. |
| Test and IDE artifacts pollute the project | Valid, corrected | Cache/IDE patterns are ignored; generated Python caches are removed from the maintained tree. |
| Unsafe or stale model loading may be possible | Partly valid, corrected | `thesis/models/train.py` uses `torch.load(..., weights_only=True)` and rejects runtimes that cannot provide that safer mode. |
| The comparison lacks uncertainty across seeds | Valid, corrected | Aggregation uses 95% Student-t intervals. Input-difference comparison also computes paired per-seed difference intervals. |
| No cipher-reference validation exists | Valid, corrected | `tests/test_cipher_kats.py` verifies official SIMON32/64 and SPECK32/64 known-answer vectors. |

## Corrections implemented in this review

1. Added strict configuration and direct data-generator validation.
2. Added cache schema, dtype, shape, label, and round checks with atomic writes.
3. Replaced normal-approximation intervals with Student-t intervals for ten-seed summaries.
4. Added paired seed-level confidence intervals to experiment comparison.
5. Made Torch/CUDA execution more deterministic and checkpoint loading safer.
6. Added manifest schema versioning, SHA-256 artifact hashes, and recursive artifact inventory.
7. Added failure/interruption recording around plotting and classical comparison.
8. Included smoke, configuration, aggregation, comparison, cache, checkpoint, and cipher KAT tests in the normal experiment test gate.
9. Archived duplicate and historical per-cipher pipelines.
10. Added a thesis-first README, dependency list, ignore rules, and archive documentation.

## Active architecture

The maintained execution path is:

`thesis/config` -> `thesis/data` -> `ciphers/registry.py` -> cipher primitive ->
`thesis/models` -> `thesis/eval` -> `thesis/classical`.

The active cipher packages expose only the maintained SIMON32/64 and SPECK32/64
primitive profiles. Historical fault, trace, encoding, and feature modules are
archived and no longer execute as package-import side effects.

## Data and leakage review

Positive examples encrypt plaintext pairs with the configured XOR difference.
Negative examples use independently random ciphertext-sized blocks. A fresh
random key is sampled per pair. The neural input is the concatenation of two
32-bit ciphertexts, represented as 64 float32 bits.

The current cache does not store keys or plaintexts. Dataset splitting is
stratified and checked to leave both classes represented in train, validation,
and test subsets.

The negative-class construction is a standard random-permutation-style
distinguisher baseline, but the thesis should state this explicitly. It does
not by itself demonstrate key recovery or performance against every alternative
null distribution.

## Model and training review

The model is a compact one-dimensional CNN trained with Adam and binary
cross-entropy. Validation AUC controls early stopping. The test split is used
for final metrics rather than model selection.

Ten seeds vary all seeded stochastic operations: generated data, train/validation
splits, model initialization, and minibatch order. They do not vary the
architecture, hyperparameters, input difference, sample count, or round list.

## Statistical review

Seed aggregates now report means, sample standard deviations, and 95%
Student-t confidence intervals. Bounded metric intervals are clipped to their
mathematical ranges.

The controlled input-difference comparison uses matching seed identifiers and
paired differences. This is more informative than visually checking whether
two independent confidence intervals overlap. The comparison remains
exploratory across many rounds and metrics; no multiple-comparison correction
is applied.

For the existing `(0x0040, 0x0000)` experiment, paired intervals indicate:

- SIMON AUC changes outside seed uncertainty at rounds 4, 5, 6, and 8.
- SIMON absolute-advantage changes outside seed uncertainty at rounds 4, 5, and 8.
- SPECK AUC and absolute advantage improve strongly at rounds 3-6.
- SPECK rounds 7-9 show no resolved difference.
- SPECK round-10 absolute advantage is slightly but consistently lower.

These statements are conditional on the existing ten runs and are not adjusted
for testing many round/metric combinations.

## Classical-analysis review

SIMON uses an exact DDT for its nonlinear round function. SPECK uses Monte Carlo
round-transition estimates because exhaustive enumeration of the relevant ARX
state space is impractical here. Multi-round tracking retains only the configured
top candidates.

Accordingly, the reported SPECK `p_max` is an estimator under sampling and
top-k pruning, not a formal exhaustive maximum. The classical process also uses
one configured seed and currently has no repeated-sampling confidence interval.

The neural curve (`|accuracy - 0.5|`) and classical curve
(`log2(max characteristic probability)`) are intentionally shown on separate
axes. They are useful for trend comparison but are not numerically equivalent
measures and must not be presented as a direct superiority ratio.

## Security and robustness review

The active code contains no `eval`, `exec`, pickle loading, or unrestricted
Torch checkpoint loading. Cached datasets are validated before use and written
atomically. Run manifests record configuration, dependencies, Git state,
parameters, progress, artifact sizes, and SHA-256 hashes.

This is research software, not a hardened service. It does not authenticate
artifacts, sandbox third-party code, or defend against a malicious local user
who can replace source, configuration, or data before execution.

## Remaining limitations

1. SPECK classical bounds need repeated Monte Carlo runs or bootstrap intervals to quantify estimator uncertainty.
2. Top-k trail pruning can miss a stronger characteristic and should be described as an approximation.
3. The project evaluates distinguishers, not key recovery or full practical attacks.
4. Generalization is tested for two selected input differences, not a broad or held-out difference family.
5. The current comparison performs multiple exploratory tests without family-wise or false-discovery correction.
6. Existing experiment artifacts were post-processed with the corrected statistics; the neural models were not retrained after these code-maintenance changes.
7. Archived code remains available for provenance but is unsupported and should not be cited as part of the maintained methodology.
8. The directory name `Archieve` is misspelled. Renaming it would improve polish but would create broad path churn without changing scientific behavior.

## Final verdict

The project is suitable for repeatable thesis experiments after the corrections
above, provided the written thesis describes the classical SPECK result as an
approximate sampled bound, avoids equating neural advantage with characteristic
probability, and limits its claims to statistical distinguishing rather than
automated key recovery.
