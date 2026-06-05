# 18 — Technical Debt & Roadmap

## 18.1 Documentation debt

| Item | Status | Action |
|------|--------|--------|
| Root `DOCUMENTATION.md` | Superseded by `docs/` set | Pointer only — see below |
| §4 layout in legacy doc | Missing `experiments/` | Use [03 — Layout](03-repository-layout.md) |
| Non-goals vs distinguisher | Contradictory in v1 | Resolved in [01 — Introduction](01-introduction-and-scope.md) |

---

## 18.2 Architectural inconsistencies

| Issue | Impact | Proposed fix |
|-------|--------|--------------|
| **Dual config loaders** (`ml` vs `experiments`) | User confusion | Unified schema or code-generated docs |
| **Dual split strategies** (fault strata vs label-only) | Different val/test composition | Shared split utility with mode flag |
| **Dual feature entry points** (`simon3264.features` vs `ml.features`) | Wrong features if mis-imported | Deprecate simon3264 matrix for ML path in docs only, or alias |
| **Dual metrics modules** | Name collision risk | Namespace or rename `experiments.classification_metrics` |
| **No installable package** | `sys.path` hacks in every CLI | Add `pyproject.toml` with `[project.scripts]` |
| **`score_agg` unused** | Dead config | Implement or remove from `ScoringConfig` |
| **Nested gitignore** | `.gitkeep` rules inconsistent | Single root policy |

---

## 18.3 Performance debt

| Location | Issue | Remediation |
|----------|-------|-------------|
| `simon3264/dataset.py` labeled_batch | Per-row Python encrypt loop | Batch `encrypt` where keys repeat |
| `experiments/distinguisher_data.py` | Per-row `encrypt_rounds` | Vectorize or Cython hot path |
| Large distinguisher `n_samples` | Long CPU time | Document; default quick config |

---

## 18.4 Testing & CI debt

| Gap | Priority |
|-----|----------|
| No CI pipeline | High |
| No e2e `run_all --quick` in CI | Medium |
| No golden-file metrics | Low |
| Torch tests skip without GPU package | Acceptable |

### Proposed GitHub Actions workflow (template)

```yaml
name: test
on: [push, pull_request]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest -q
      - run: python scripts/verify_assignment_setup.py
```

Optional job: `experiments/run_all.py --quick` with timeout (requires torch download).

---

## 18.5 Roadmap (suggested)

| Phase | Deliverable |
|-------|-------------|
| P0 | CI with pytest + verify script |
| P1 | `pyproject.toml` + console_scripts |
| P2 | Batch dataset generation |
| P3 | CSV ingest for `score-file` / supervised path |
| P4 | Calibration plots notebook |
| P5 | Optional CBC mode behind explicit flag |

---

## 18.6 Areas safe to refactor

| Area | Risk if changed |
|------|-----------------|
| `simon.py` KAT | High — run `test_simon.py` |
| `build_feature_matrix` width | High — breaks all models |
| `thresholds.json` key names | Medium — update round_sweep reference_model |
| `experiments/` only | Low — isolated layer |

---

## 18.7 Undocumented or lightly documented code

| Item | Notes |
|------|-------|
| `simon3264/trace.py` subkey export | Research utility — expand examples in future |
| `build_hw_reference` in ml/features | HW reference for chi-square — rarely used in default path |
| `encrypt_variant` | Used only for wrong_z faults |

---

[← Security](17-security-limitations.md) · [Appendix A →](appendices/A-glossary.md)
