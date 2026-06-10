"""Round-count sweep: train CNN distinguisher and log test metrics per cipher."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from thesis.config.loader import config_path_for_profile, load_config
from thesis.data.generator import DEFAULT_INPUT_DELTA, generate_or_load
from thesis.eval.manifest import artifact_inventory, build_manifest, utc_now, write_manifest
from thesis.eval.metrics import classification_metrics, metrics_row
from thesis.eval.plot_results import plot_all
from thesis.models.train import TrainConfig, _predict_scores, train_distinguisher

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = config_path_for_profile("full")

CSV_FIELDS = [
    "seed",
    "cipher",
    "rounds",
    "split",
    "n_samples",
    "accuracy",
    "auc_roc",
    "tpr",
    "tnr",
    "advantage_abs",
    "advantage_edge",
    "youden_j",
]


def load_thesis_config(path: Path | None = None) -> dict[str, Any]:
    return load_config(path or DEFAULT_CONFIG)


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def _create_timestamped_run_dir(base_results_path: Path) -> Path:
    """Create a timestamped results directory for experimental hygiene."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_results_path / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_round_sweep(
    config_path: Path | None = None,
    *,
    ciphers: list[str] | None = None,
    rounds_list: list[int] | None = None,
    n_samples: int | None = None,
    data_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
    seed_override: int | None = None,
    force_regen: bool = False,
    fresh_csv: bool = False,
    use_timestamped_dir: bool = True,
    training_overrides: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """Sweep rounds for each cipher; append test metrics to multi-seed CSV files."""
    cfg = load_thesis_config(config_path)
    base = _REPO_ROOT

    cipher_names = ciphers or cfg.get("ciphers") or [cfg.get("cipher", "simon")]
    if isinstance(cipher_names, str):
        cipher_names = [cipher_names]

    rounds_list = rounds_list or cfg.get("rounds", [3, 4, 5, 6, 7, 8, 9, 10])
    n = n_samples if n_samples is not None else int(cfg.get("n_samples_per_round", 100_000))
    seed = seed_override if seed_override is not None else int(cfg.get("seed", 1))
    delta = tuple(cfg.get("input_delta", list(DEFAULT_INPUT_DELTA)))
    data_path = _resolve_path(
        base, str(data_dir) if data_dir is not None else cfg.get("data_dir", "thesis/data/cache")
    )
    model_path = _resolve_path(
        base, str(model_dir) if model_dir is not None else cfg.get("model_dir", "models/thesis")
    )
    results_path = _resolve_path(
        base,
        str(results_dir) if results_dir is not None else cfg.get("results_dir", "results/thesis"),
    )
    log_path = None
    if log_dir is not None:
        log_path = _resolve_path(base, str(log_dir)) if not Path(log_dir).is_absolute() else Path(log_dir)
    training_values = {**cfg.get("training", {}), **(training_overrides or {})}
    train_cfg = TrainConfig.from_dict(training_values)
    train_cfg.seed = seed
    train_ratio = float(cfg.get("train_ratio", 0.7))
    val_ratio = float(cfg.get("val_ratio", 0.15))

    # Create timestamped results directory for experimental hygiene
    if use_timestamped_dir:
        results_path = _create_timestamped_run_dir(results_path)
        manifest_path = results_path / "manifest.json"
        manifest = build_manifest(
            repo_root=base,
            run_type="round_sweep",
            config_path=Path(config_path or DEFAULT_CONFIG),
            config=cfg,
            parameters={
                "seeds": [seed],
                "ciphers": cipher_names,
                "rounds": rounds_list,
                "input_delta": list(delta),
                "n_samples_per_round": n,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "training": asdict(train_cfg),
                "force_regen": force_regen,
                "fresh_csv": fresh_csv,
            },
            paths={
                "results_dir": results_path,
                "data_dir": data_path,
                "model_dir": model_path,
                "log_dir": log_path,
            },
        )
        write_manifest(manifest_path, manifest)
        print(f"[sweep] Using timestamped results directory: {results_path}")
    else:
        manifest_path = None
        manifest = None

    all_rows: list[dict[str, Any]] = []

    for cipher in cipher_names:
        csv_path = results_path / f"{cipher}_multi_seed_raw.csv"
        round_csv_path = results_path / f"{cipher}_round_sweep.csv"
        if fresh_csv:
            if csv_path.exists():
                csv_path.unlink()
            if round_csv_path.exists():
                round_csv_path.unlink()

        for rounds in rounds_list:
            print(f"[sweep] seed={seed} {cipher} R={rounds} — loading data (n={n}) …")
            data = generate_or_load(
                cipher,
                rounds=rounds,
                n_samples=n,
                input_delta=delta,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                data_dir=data_path,
                force_regen=force_regen,
            )
            X, y = data["X"], data["y"]
            splits = data["splits"]

            print(f"[sweep] seed={seed} {cipher} R={rounds} — training …")
            result = train_distinguisher(
                X,
                y,
                splits,
                cipher=cipher,
                rounds=rounds,
                model_dir=model_path,
                cfg=train_cfg,
                log_dir=log_path,
            )
            model = result["model"]
            device = result["device"]

            test_idx = splits["test"]
            X_te, y_te = X[test_idx], y[test_idx]
            scores = _predict_scores(model, X_te, device)
            test_m = classification_metrics(y_te, scores)
            row = metrics_row(
                cipher=cipher,
                rounds=rounds,
                split="test",
                n_samples=len(y_te),
                metrics=test_m,
                seed=seed,
            )
            _append_csv_row(csv_path, row)
            _append_csv_row(round_csv_path, row)
            all_rows.append(row)
            print(
                f"[sweep] seed={seed} {cipher} R={rounds} — "
                f"acc={test_m['accuracy']:.4f} auc={test_m['auc_roc']:.4f} "
                f"adv_abs={test_m['advantage_abs']:.4f}"
            )

    plot_all(results_path, cipher_names)
    if manifest is not None and manifest_path is not None:
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["progress"]["completed_seeds"] = [seed]
        manifest["artifacts"] = artifact_inventory(results_path)
        write_manifest(manifest_path, manifest)

    return all_rows, results_path


def resolve_config_path(
    config: Path | None = None,
    profile: str | None = None,
) -> Path:
    if config is not None:
        return config
    if profile in ("full", "quick", None):
        return config_path_for_profile(profile or "full")
    raise ValueError(f"unknown profile {profile!r}; use full or quick")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis neural distinguisher round sweep")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (overrides --profile)",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "quick"],
        default=None,
        help="Use thesis.yaml (full) or thesis_quick.yaml (quick)",
    )
    parser.add_argument(
        "--cipher",
        action="append",
        choices=["simon", "speck"],
        help="Restrict to cipher(s); default: all in config",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        help="Override round list from config",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override n_samples_per_round from config",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override dataset cache directory",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override results CSV directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory (e.g., runs/thesis)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from config",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Regenerate cached datasets",
    )
    parser.add_argument(
        "--fresh-csv",
        action="store_true",
        help="Delete existing per-cipher multi_seed_raw.csv before writing",
    )
    parser.add_argument(
        "--no-timestamped-dir",
        action="store_true",
        help="Disable timestamped results directory (use base results_dir directly)",
    )
    args = parser.parse_args()
    cfg_path = resolve_config_path(args.config, args.profile)
    run_round_sweep(
        cfg_path,
        ciphers=args.cipher,
        rounds_list=args.rounds,
        n_samples=args.n_samples,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        log_dir=args.log_dir,
        seed_override=args.seed,
        force_regen=args.force_regen,
        fresh_csv=args.fresh_csv,
        use_timestamped_dir=not args.no_timestamped_dir,
    )


if __name__ == "__main__":
    main()
