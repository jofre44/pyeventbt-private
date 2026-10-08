"""Envolvente MC de drawdown (Etapa 5): block bootstrap de retornos diarios.

Salidas: P(DD>X), ES, máximo periodo perdedor, recovery time, envolventes
p95/p99 (la monitorización las usa como alerta/halt en Etapa 6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _max_dd(eq: np.ndarray) -> tuple[float, int, int]:
    roll = np.maximum.accumulate(eq)
    dd = eq / np.maximum(roll, 1e-12) - 1.0
    t = int(np.argmin(dd))
    peak = int(np.argmax(eq[:t + 1])) if t else 0
    return float(-dd[t]), t - peak, t


def mc_dd_envelope(daily: pd.Series, n_paths: int = 5000, block: int = 10,
                   percentiles=(95, 99), seed: int = 1500) -> dict:
    r = daily.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 30:
        return {"verdict": "NODATA"}
    rng = np.random.default_rng(seed)
    maxdds, maxloss_streak, recovs, ann_loss = [], [], [], []
    for _ in range(n_paths):
        idx = np.concatenate([np.arange(s, min(s + block, n))
                              for s in rng.integers(0, n, size=n // block + 1)])[:n]
        path = r[idx]
        eq = np.cumprod(1 + path)
        dd, dd_len, _ = _max_dd(eq)
        maxdds.append(dd)
        # Racha perdedora máxima (días)
        neg = path < 0
        streak = cur = 0
        for v in neg:
            cur = cur + 1 if v else 0
            streak = max(streak, cur)
        maxloss_streak.append(streak)
        # Pérdida anualizada del camino
        yrs = n / 252
        ann_loss.append(float(eq[-1] ** (1 / yrs) - 1) if eq[-1] > 0 else -1.0)
        recovs.append(dd_len)
    maxdds = np.asarray(maxdds)
    env = {f"p{p}": float(np.percentile(maxdds, p)) for p in percentiles}
    dd = float(np.mean(maxdds))
    tail = maxdds[maxdds >= env[f"p{percentiles[0]}"]]
    return {
        "dd_envelope": env,
        "mean_max_dd": dd,
        "expected_shortfall_dd": float(tail.mean()) if len(tail) else dd,
        "max_losing_streak_days": int(np.percentile(maxloss_streak, percentiles[-1])),
        "max_recovery_days": int(np.percentile(recovs, percentiles[-1])),
        "p_annual_loss": float(np.mean(np.asarray(ann_loss) < 0)),
        "n_paths": n_paths,
    }
