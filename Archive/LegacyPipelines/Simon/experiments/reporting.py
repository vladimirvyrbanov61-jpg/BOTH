"""Merge experiment outputs into assignment-ready Markdown report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional


METHODOLOGY = """
## Methodology (assignment mapping)

| Criterion | Implementation | Output |
|-----------|----------------|--------|
| SIMON 32/64 cipher | `simon.py`, `simon3264/` | KAT tests, `encrypt_rounds` |
| Deep learning on ciphertext | `TorchAutoencoder` + 42-dim features | `models/torch_autoencoder.pt` |
| Statistical anomaly detection | One-class AE on 32-round normal | Track A round sweep |
| Automated cryptanalysis | Neural distinguisher (pairs vs random) | Track B `results/distinguisher/` |
| Varying round counts | Sweeps over `round_values` in config | CSV + PNG plots |

**Track A** trains the autoencoder on genuine 32-round ciphertext features, then scores
ciphertext produced with fewer rounds (and random blocks) using the fixed validation threshold.
At `r < 32`, `detection_rate` is the fraction flagged anomalous; at `r = 32` it is specificity
(fraction correctly not flagged).

**Track B** trains a separate binary classifier on XOR-of-pair bit features to distinguish
reduced-round Simon ciphertext pairs (fixed input difference Δ) from random pairs.
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            v = row.get(col, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def generate_assignment_report(
    *,
    results_root: Path = Path("results"),
    round_sweep_dir: Optional[Path] = None,
    distinguisher_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> str:
    results_root = Path(results_root)
    round_sweep_dir = round_sweep_dir or results_root / "round_sweep"
    distinguisher_dir = distinguisher_dir or results_root / "distinguisher"
    output_path = output_path or results_root / "ASSIGNMENT_SUMMARY.md"

    rs_rows = _read_csv(round_sweep_dir / "round_sweep.csv")
    dist_rows = _read_csv(distinguisher_dir / "summary.csv")

    rs_plot = round_sweep_dir / "score_vs_rounds.png"
    dist_plot = distinguisher_dir / "advantage_vs_rounds.png"

    parts = [
        "# Assignment experiment summary\n",
        METHODOLOGY,
        "\n## Track A — Autoencoder vs round count\n\n",
        _markdown_table(
            [{k: r.get(k, "") for k in r} for r in rs_rows],
            ["rounds", "mean_score", "detection_rate", "auc_vs_random", "n_samples"],
        ),
    ]
    if rs_plot.exists():
        parts.append(f"\n![Round sweep scores]({rs_plot.as_posix()})\n")

    parts.extend(
        [
            "\n## Track B — Neural distinguisher\n\n",
            _markdown_table(
                [{k: r.get(k, "") for k in r} for r in dist_rows],
                ["rounds", "accuracy", "auc", "advantage", "tpr", "tnr", "n_test"],
            ),
        ]
    )
    if dist_plot.exists():
        parts.append(f"\n![Distinguisher advantage]({dist_plot.as_posix()})\n")

    parts.append(
        """
## Interpretation (for written report)

- **Track A:** Reconstruction error should rise as round count drops below 32, showing the AE
  flags non-standard (truncated-round) ciphertext as statistical anomalies relative to training.
- **Track B:** Classifier advantage above 0.5 accuracy at low `r` indicates automated
  distinguishing of reduced-round Simon from random; advantage typically falls as `r` increases.
- **Limitation:** Neither track performs full 32-round key recovery; baselines include random blocks/pairs.

## Reproduction

```bash
python ml/train.py --torch-only
python experiments/run_all.py --config configs/cryptanalysis.yaml
```

Figures: `results/round_sweep/score_vs_rounds.png`, `results/distinguisher/advantage_vs_rounds.png`.
"""
    )

    text = "".join(parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text
