# 13 — Testing & Quality

## 13.1 Philosophy

Tests prioritize **correctness of cryptography** and **smoke coverage of ML/experiments** over exhaustive ML performance regression. There is **no** enforced coverage percentage or CI gate in-repo.

---

## 13.2 Test suite map

| File | Tests | Runtime | Dependencies |
|------|-------|---------|--------------|
| `test_simon.py` | Cipher primitives, KAT, variants | Fast | numpy, pytest |
| `test_simon3264.py` | Toolkit integration | Fast | numpy, pytest |
| `test_ml_smoke.py` | Features, dataset, IF, NumPy AE | Fast | numpy, sklearn, pyyaml |
| `test_experiments.py` | Config, distinguisher data, metrics | Fast | numpy, pytest; torch tests optional |

**Typical full run:** `pytest` → ~80+ tests, <15s without torch training.

---

## 13.3 What is covered

| Area | Coverage |
|------|----------|
| Official SIMON 32/64 vector | Yes (KAT) |
| Endian / hex parsing | Yes |
| Fault injectors | Yes |
| 42-dim feature shape | Yes |
| Dataset stratified split | Yes |
| Torch AE save/load | Smoke (if torch installed) |
| Round sweep CSV output | Smoke (if torch installed) |
| Distinguisher TPR/TNR math | Yes (unit) |

---

## 13.4 What is not covered

| Gap | Risk |
|-----|------|
| Torch AE training convergence | Medium — manual Colab check |
| Full `run_all.py` end-to-end | Slow — run manually before submission |
| Distinguisher accuracy thresholds | Research metric — not golden-file |
| GPU-specific behaviour | Environment-dependent |
| `score_agg` config field | Unused — no test |

---

## 13.5 Running tests locally

```bash
pip install -r requirements.txt
pytest -v
pytest test_experiments.py -v   # experiments only
```

**Without torch:** 2 tests in `test_experiments.py` skip (`importorskip("torch")`).

---

## 13.6 Recommended pre-commit / PR checklist

- [ ] `pytest` passes
- [ ] `python scripts/verify_assignment_setup.py` exits 0
- [ ] If touching features: `test_ml_smoke.test_feature_dim` still passes
- [ ] If touching cipher: `test_simon.py` KAT passes
- [ ] If touching experiments: `test_experiments.py` passes

---

## 13.7 CI/CD status

**No GitHub Actions or other CI is configured.** See proposed workflow in [18 — Technical debt](18-technical-debt-and-roadmap.md).

---

## 13.8 Quality practices for contributors

| Practice | Detail |
|----------|--------|
| Minimize scope | Cipher changes require KAT update |
| Match conventions | `sys.path` insert in CLIs; dataclass configs |
| Regenerate caches | Use `--force-regen` when changing dataset logic |
| Document config changes | Update [09](09-configuration-reference.md) |

---

[← CLI](12-cli-reference.md) · [Next: Onboarding →](14-onboarding-and-troubleshooting.md)
