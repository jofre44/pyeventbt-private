"""CPCV por superviviente (Etapa 3B): distribución de Sharpe en caminos CPCV.

n_groups=10, k_test=2 → 45 caminos; purga = embargo en días. Gate: mediana > 0
y ≥70% de caminos positivos. Evalúa parámetros fijos (el OOS estricto es el WF).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def cpcv_distribution(
    daily: pd.Series | pd.DataFrame,
    n_groups: int = 10,
    k_test_groups: int = 2,
    purge_days: int = 2,
) -> dict:
    s = daily.iloc[:, 0] if isinstance(daily, pd.DataFrame) else daily
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    n = len(s)
    if n < n_groups * 5:
        return {"verdict": "NODATA", "n_paths": 0}
    blocks = np.array_split(np.arange(n), n_groups)
    vals = s.to_numpy(dtype=float)
    sharpes = []
    for combo in itertools.combinations(range(n_groups), k_test_groups):
        idx = np.concatenate([blocks[g] for g in combo])
        if purge_days:
            idx = idx[purge_days:-purge_days] if len(idx) > 2 * purge_days else idx[:0]
        if len(idx) < 5:
            continue
        mu, sd = float(np.mean(vals[idx])), float(np.std(vals[idx], ddof=1))
        sharpes.append(float(np.sqrt(252) * mu / sd) if sd > 0 else float("nan"))
    sharpes = np.asarray([v for v in sharpes if np.isfinite(v)])
    if len(sharpes) == 0:
        return {"verdict": "NODATA", "n_paths": 0}
    med = float(np.median(sharpes))
    pos = float(np.mean(sharpes > 0))
    ok = med > 0 and pos >= 0.70
    return {"median": med, "frac_positive": pos, "n_paths": int(len(sharpes)),
            "verdict": "PASS" if ok else "FAIL", "sharpes": sharpes.tolist()}
