"""Merge experiment outputs into assignment-ready Markdown report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional


METHODOLOGY = """
## Methodology

| Component | Implementation | Output |
|-----------|----------------|--------|
| SPECK 32/64 cipher | `speck.py`, `speck3264/` | KAT tests, `encrypt_rounds` |
| Deep learning on ciphertext | `TorchAutoencoder` + 42-dim features | `models/torch_autoencoder.pt` |
| Statistical anomaly detection | One-class AE on 22-round normal | Track A round sweep |
| Automated cryptanalysis | Neural distinguisher (pairs vs random) | Track B `results/distinguisher/` |
| Varying round counts | Sweeps over `round_values` in config | CSV + PNG plots |

**Track A** trains the autoencoder on genuine 22-round ciphertext features, then scores
ciphertext produced with fewer rounds (and random blocks).

**Track B** trains a binary classifier on XOR-of-pair bit features to distinguish
reduced-round Speck ciphertext pairs (fixed input difference Δ) from random pairs.
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data._\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep] + body) + "\n"


def generate_assignment_report(
    *,
    results_root: Optional[Path] = None,
) -> str:
    root = results_root or Path("results")
    parts = ["# SPECK 32/64 Cryptanalysis — Summary\n", METHODOLOGY]

    rs_csv = root / "round_sweep" / "round_sweep.csv"
    if not rs_csv.exists():
        rs_csv = Path("results/round_sweep/round_sweep.csv")
    rs_rows = _read_csv(rs_csv)
    if rs_rows:
        parts.append("\n## Track A: Round sweep\n")
        cols = [c for c in rs_rows[0].keys() if c]
        parts.append(_markdown_table(rs_rows, cols[:8]))

    dist_json = root / "distinguisher" / "summary.json"
    if not dist_json.exists():
        dist_json = Path("results/distinguisher/summary.json")
    if dist_json.exists():
        metrics = json.loads(dist_json.read_text(encoding="utf-8"))
        if metrics:
            parts.append("\n## Track B: Neural distinguisher\n")
            cols = ["rounds", "accuracy", "auc", "advantage", "tpr", "tnr"]
            parts.append(_markdown_table(metrics, cols))

    text = "\n".join(parts)
    out = root / "ASSIGNMENT_SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
