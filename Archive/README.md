# Historical Material

This directory contains superseded experiments, plots, intermediate backups,
and old helper scripts. Nothing under this directory is imported by or required
for the active thesis pipeline.

The historical code is retained only for provenance. It may contain obsolete
interfaces, old methodology, and unsafe serialization patterns and must not be
treated as supported runtime code.

## Do Not Mix With Thesis Results

Legacy configurations may sweep different round ranges, including SPECK rounds
beyond the thesis experiment's configured rounds 3-10. They also use different
dataset definitions, metrics, model code, and artifact schemas. Their figures
must not be combined with or compared numerically against outputs from
`thesis/`.

Only commands documented in the root `README.md` are supported.

Numbered experiment snapshots are intentionally local-only and ignored by Git.
They contain generated results and terminal transcripts, not maintained source.
Published thesis evidence must come from a timestamped current-pipeline run with
its manifest and artifact hashes.
