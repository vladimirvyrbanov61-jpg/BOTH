# Simon Project — Master Documentation Index

**Version:** 2.0 (multi-file production documentation)  
**Repository:** SIMON 32/64 cipher + ML anomaly detection + assignment cryptanalysis experiments  
**Audience:** Developers, researchers, students, operators running batch pipelines on Colab or locally

---

## How to read this documentation

This is the **authoritative** documentation set for the repository. The legacy single-file [`DOCUMENTATION.md`](../DOCUMENTATION.md) at the repo root is retained as a **compatibility pointer**; new material lives under `docs/`.

| If you want to… | Start here |
|-----------------|------------|
| Understand what the project does | [01 — Introduction & scope](01-introduction-and-scope.md) |
| See architecture and data flows | [02 — System architecture](02-system-architecture.md) |
| Navigate the codebase | [03 — Repository layout](03-repository-layout.md) |
| Train / score anomalies | [06 — ML pipeline](06-ml-anomaly-pipeline.md) + [12 — CLI reference](12-cli-reference.md) |
| Run assignment experiments | [07 — Experiments layer](07-experiments-cryptanalysis.md) + [16 — Assignment & Colab](16-assignment-and-colab.md) |
| Configure experiments | [09 — Configuration reference](09-configuration-reference.md) |
| Onboard locally or on Colab | [14 — Onboarding & troubleshooting](14-onboarding-and-troubleshooting.md) |
| Understand limitations & debt | [17 — Security & limitations](17-security-limitations.md) + [18 — Technical debt](18-technical-debt-and-roadmap.md) |

---

## Documentation map

### Part I — Context & architecture

1. [Introduction & scope](01-introduction-and-scope.md)
2. [System architecture](02-system-architecture.md)
3. [Repository layout](03-repository-layout.md)

### Part II — Domain & core modules

4. [Cryptography: `simon.py`](04-cryptography-simon.md)
5. [`simon3264/` toolkit](05-simon3264-toolkit.md)
6. [`ml/` anomaly pipeline](06-ml-anomaly-pipeline.md)
7. [`experiments/` cryptanalysis layer](07-experiments-cryptanalysis.md)

### Part III — Data, configuration, APIs

8. [Persistence & artifacts](08-data-persistence-and-artifacts.md)
9. [Configuration reference](09-configuration-reference.md)
10. [Feature engineering](10-feature-engineering.md)
11. [Models & API contracts](11-models-and-apis.md)
12. [CLI reference](12-cli-reference.md)

### Part IV — Engineering & operations

13. [Testing & quality](13-testing-and-quality.md)
14. [Onboarding & troubleshooting](14-onboarding-and-troubleshooting.md)
15. [Operations runbook](15-operations-and-observability.md)
16. [Assignment & Colab](16-assignment-and-colab.md)

### Part V — Risk & maintenance

17. [Security & limitations](17-security-limitations.md)
18. [Technical debt & roadmap](18-technical-debt-and-roadmap.md)

### Appendices

- [A — Glossary](appendices/A-glossary.md)
- [B — File formats](appendices/B-file-formats.md)
- [C — References](appendices/C-references.md)

---

## Quick start (copy-paste)

```bash
# Local / Colab — install
pip install -r requirements-colab.txt   # or requirements.txt

# Verify structure (no training)
python scripts/verify_assignment_setup.py

# Primary ML pipeline
python ml/train.py --torch-only --n-samples 20000 --epochs 40
python ml/score.py evaluate --model torch_autoencoder

# Assignment / cryptanalysis (both tracks + report)
python experiments/run_all.py --config configs/cryptanalysis.yaml

# Fast smoke (~minutes)
python experiments/run_all.py --quick
```

---

## Companion documents (outside `docs/`)

| File | Purpose |
|------|---------|
| [`README.md`](../README.md) | Short project overview |
| [`COLAB.md`](../COLAB.md) | Minimal Colab checklist |
| [`docs/ASSIGNMENT_ALIGNMENT.md`](ASSIGNMENT_ALIGNMENT.md) | Rubric mapping & report template |
| [`configs/default.yaml`](../configs/default.yaml) | ML experiment defaults |
| [`configs/cryptanalysis.yaml`](../configs/cryptanalysis.yaml) | Full assignment sweep config |
| [`configs/cryptanalysis_quick.yaml`](../configs/cryptanalysis_quick.yaml) | Fast local verification config |

---

## Document conventions

- **Paths** are relative to the repository root unless stated otherwise.
- **CLI examples** assume the current working directory is the repo root (`Simon/`).
- **Code citations** refer to modules by path (e.g. `ml/models.py`); line numbers may shift between commits.
- **“Normal”** = label `0`, valid 32-round SIMON ciphertext (in the default ML pipeline).
- **“Anomaly”** = label `1`, synthetic fault or non-standard ciphertext.

---

*Generated as production-grade project documentation. For changes, update the relevant chapter under `docs/` and bump the version note in this file.*
