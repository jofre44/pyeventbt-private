"""Scoring de consistencia sintética (Etapa 3.2a, DMRI reinterpretado).

Por representante: 1 backtest real + N sintéticos. `score_percentile_rank`
por métrica con banda [low, high], quorum y bordes conservadores:
  - NaN/inf (real o sintético degenerado) → None (no vota)
  - varianza cero en sintéticos → 50 (neutral)
Semántica: rank bajo (<low) = FAIL; rank alto (>high) = ÁMBAR (overfitting O
edge real que el generador no captura; lo resuelven edge test + 3B).
"""
from __future__ import annotations

import numpy as np


def score_percentile_rank(real_val: float, synth_vals: list[float] | np.ndarray) -> float | None:
    s = np.asarray(list(synth_vals), dtype=float)
    s = s[np.isfinite(s)]
    if not np.isfinite(real_val) or len(s) == 0:
        return None
    if float(np.std(s)) == 0.0:
        return 50.0
    return float(np.mean(s <= real_val) * 100.0)


def rank_verdict(rank: float | None, low: float = 10.0, high: float = 90.0) -> str:
    if rank is None:
        return "NODATA"
    if rank < low:
        return "FAIL"  # peor que mercados plausibles: algo raro
    if rank > high:
        return "AMBER"  # overfitting o edge no capturado: decide edge test + 3B
    return "GREEN"


def consistency_verdict(ranks: dict[str, float | None], quorum: float = 0.5,
                        low: float = 10.0, high: float = 90.0) -> dict:
    votes = {m: rank_verdict(r, low, high) for m, r in ranks.items()}
    counted = [v for v in votes.values() if v != "NODATA"]
    if not counted:
        return {"verdict": "NODATA", "votes": votes}
    if any(v == "FAIL" for v in counted):
        final = "RED"
    else:
        green_share = sum(v == "GREEN" for v in counted) / len(counted)
        final = "GREEN" if green_share >= quorum else "AMBER"
    return {"verdict": final, "votes": votes}
