"""R-round maximum differential characteristic probability (max-trail composition)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

from thesis.classical.ddt_core import (
    Delta32,
    max_trail_probability,
    prune_top_k,
    row_from_conditional,
)
from thesis.classical.ddt_simon import (
    analytical_round_bound_from_f,
    compute_f_ddt_exact,
    compute_simon_round_ddt,
)
from thesis.classical.ddt_speck import compute_speck_round_ddt

CipherName = Literal["simon", "speck"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _expand_transition_rows(
    cipher: CipherName,
    delta_in: Delta32,
    transition_cache: dict[Delta32, dict[Delta32, float]],
    *,
    n_samples_row: int,
    seed: int,
    f_ddt_simon: dict[int, dict[int, float]] | None,
) -> None:
    if delta_in in transition_cache:
        return
    if cipher == "simon" and f_ddt_simon is not None:
        transition_cache[delta_in] = analytical_round_bound_from_f(delta_in, f_ddt_simon)
    elif cipher == "simon":
        transition_cache[delta_in] = compute_simon_round_ddt(
            delta_in, n_samples=n_samples_row, seed=seed
        )
    else:
        transition_cache[delta_in] = compute_speck_round_ddt(
            delta_in, n_samples=n_samples_row, seed=seed
        )


def max_characteristic_probability(
    cipher: CipherName,
    rounds: int,
    delta_in: Delta32,
    *,
    n_samples_row: int | None = None,
    top_k: int = 32,
    seed: int = 0,
    use_simon_f_exact: bool = True,
) -> tuple[float, list[Delta32]]:
    """Maximum R-round characteristic probability along a single differential trail.

    Uses repeated 1-round conditional distributions P(Δ_{r+1} | Δ_r) with
    max-product dynamic programming (best trail, not sum over all trails).
    """
    if rounds < 1:
        return 1.0, [delta_in]

    if n_samples_row is None:
        n_samples_row = 250_000 if cipher == "simon" else 1_000_000

    f_ddt = compute_f_ddt_exact() if cipher == "simon" and use_simon_f_exact else None
    transition_cache: dict[Delta32, dict[Delta32, float]] = {}

    states: dict[Delta32, float] = {delta_in: 1.0}
    trail: list[Delta32] = [delta_in]

    for r in range(rounds):
        nxt: dict[Delta32, float] = {}
        for d_in, p_in in states.items():
            _expand_transition_rows(
                cipher,
                d_in,
                transition_cache,
                n_samples_row=n_samples_row,
                seed=seed + r,
                f_ddt_simon=f_ddt,
            )
            row = row_from_conditional(transition_cache, d_in)
            for d_out, p_cond in row.items():
                cand = p_in * p_cond
                if cand > nxt.get(d_out, 0.0):
                    nxt[d_out] = cand
        if not nxt:
            return 0.0, trail
        states = prune_top_k(nxt, top_k)
        trail.append(max(states, key=states.get))

    return max(states.values()), trail


def max_characteristic_from_fixed_transition(
    delta_in: Delta32,
    transition: dict[Delta32, dict[Delta32, float]],
    rounds: int,
    *,
    top_k: int = 32,
) -> tuple[float, list[Delta32]]:
    """Compose R rounds from a precomputed sparse transition matrix."""
    return max_trail_probability(delta_in, transition, rounds, top_k=top_k)


def track_characteristic_over_rounds(
    cipher: CipherName,
    round_list: list[int],
    delta_in: Delta32,
    *,
    n_samples_row: int | None = None,
    top_k: int = 32,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in round_list:
        p_max, trail = max_characteristic_probability(
            cipher,
            r,
            delta_in,
            n_samples_row=n_samples_row,
            top_k=top_k,
            seed=seed,
        )
        rows.append(
            {
                "cipher": cipher,
                "rounds": r,
                "delta_in": delta_in,
                "max_characteristic_prob": p_max,
                "trail_len": len(trail),
            }
        )
    return rows


def save_classical_bounds_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cipher", "rounds", "max_characteristic_prob", "delta_left", "delta_right"]
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for row in rows:
            d = row["delta_in"]
            w.writerow(
                {
                    "cipher": row["cipher"],
                    "rounds": row["rounds"],
                    "max_characteristic_prob": row["max_characteristic_prob"],
                    "delta_left": d[0],
                    "delta_right": d[1],
                }
            )


def load_classical_bounds_csv(path: Path) -> dict[int, float]:
    """Map rounds → max_characteristic_prob."""
    out: dict[int, float] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(row["rounds"])] = float(row["max_characteristic_prob"])
    return out
