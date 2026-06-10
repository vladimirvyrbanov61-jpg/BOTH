"""Experiment-manifest helpers for reproducible thesis runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _dependency_versions() -> dict[str, str | None]:
    names = ("numpy", "PyYAML", "scikit-learn", "torch", "matplotlib", "seaborn")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    repo_root: Path,
    run_type: str,
    config_path: Path,
    config: dict[str, Any],
    parameters: dict[str, Any],
    paths: dict[str, Path | str | None],
) -> dict[str, Any]:
    created_at = utc_now()
    return {
        "schema_version": 3,
        "run_type": run_type,
        "status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "completed_at": None,
        "command": [sys.executable, *sys.argv],
        "working_directory": os.getcwd(),
        "git": _git_metadata(repo_root),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "dependencies": _dependency_versions(),
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": config_digest(config),
            "resolved": config,
        },
        "parameters": parameters,
        "statistics": {
            "grouping": ["cipher", "rounds", "split"],
            "dispersion": "sample standard deviation (ddof=1; zero for one seed)",
            "error_bars": (
                "95% Student-t confidence interval across seeds; clipped to "
                "the mathematical range of bounded metrics"
            ),
        },
        "paths": {key: str(value) if value is not None else None for key, value in paths.items()},
        "progress": {"completed_seeds": [], "failed_seeds": []},
        "artifacts": [],
    }


def artifact_inventory(run_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "type": path.suffix.lstrip(".") or "file",
                    "sha256": file_digest(path),
                }
            )
    return artifacts


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["schema_version"] = 3
    statistics = manifest.setdefault("statistics", {})
    statistics.update(
        {
            "grouping": ["cipher", "rounds", "split"],
            "dispersion": "sample standard deviation (ddof=1; zero for one seed)",
            "error_bars": (
                "95% Student-t confidence interval across seeds; clipped to "
                "the mathematical range of bounded metrics"
            ),
        }
    )
    manifest["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
