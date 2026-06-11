# Unsupported Historical Helpers

These scripts predate the maintained `thesis/` pipeline. They are retained only
to explain old local work and are not imported, tested, or invoked by supported
commands.

Some helpers rewrite or delete rows in historical CSV files. They must not be
used to alter current thesis results. Current aggregation, validation, and
classical recomputation are implemented by `thesis.eval.aggregate`,
`thesis.eval.compare`, and `scripts.multi_seed_sweep`; their actions are
recorded in the run manifest.

No result produced by the maintained pipeline requires these scripts. Archived
CSVs cannot be assumed to be raw, reproducible thesis evidence.
