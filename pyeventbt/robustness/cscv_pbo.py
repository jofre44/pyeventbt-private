"""CSCV/PBO real (Etapa 3B): barato, sin re-optimizar.

Matriz daily_returns (fechas × trials) → 10 bloques contiguos → mitades
train/test (purga/embargo) → en cada split, el mejor de train y su rank en
test. PBO = P(rank_OOS < mediana). Gate: PBO ≤ 0.30.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _sharpe_block(X: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        sr = np.sqrt(252) * mu / np.where(sd > 0, sd, np.nan)
    return sr


def cscv_pbo(
    daily_returns,
    n_groups: int = 10,
    embargo_days: int = 2,
    max_splits: int = 500,
    seed: int = 1500,
) -> dict:
    """Devuelve {pbo, logits, oos_degradation, n_splits, verdict}."""
    df = daily_returns.copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how="all", axis=1)
    n, k = df.shape
    if k < 2 or n < n_groups * 5:
        return {"pbo": None, "verdict": "NODATA", "note": "muestra insuficiente"}
    blocks = np.array_split(np.arange(n), n_groups)
    half = n_groups // 2
    combos = list(itertools.combinations(range(n_groups), half))
    if len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]
    ranks, logits, degrad = [], [], []
    X = df.to_numpy(dtype=float)
    for combo in combos:
        train_idx = np.concatenate([blocks[g] for g in combo])
        test_idx = np.concatenate([blocks[g] for g in range(n_groups) if g not in combo])
        # Purga: quita bordes de cada bloque test (embargo en días)
        keep = np.ones(len(test_idx), dtype=bool)
        pos = 0
        for g in range(n_groups):
            if g not in combo:
                ln = len(blocks[g])
                keep[pos:pos + min(embargo_days, ln)] = False
                pos += ln
        test_idx = test_idx[keep]
        if len(test_idx) < 5 or len(train_idx) < 5:
            continue
        sr_tr = _sharpe_block(X[train_idx])
        if not np.isfinite(sr_tr).any():
            continue
        pick = int(np.nanargmax(sr_tr))
        sr_te = _sharpe_block(X[test_idx])
        order = np.argsort(np.nan_to_num(sr_te, nan=-np.inf))
        rank = float(np.searchsorted(np.sort(sr_te[np.isfinite(sr_te)]), sr_te[pick]) / max(np.isfinite(sr_te).sum() - 1, 1))
        ranks.append(rank)
        r = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(r / (1 - r))))
        degrad.append(float(sr_tr[pick] - sr_te[pick]))
    if not ranks:
        return {"pbo": None, "verdict": "NODATA"}
    pbo = float(np.mean(np.asarray(ranks) < 0.5))
    return {
        "pbo": pbo, "logits": logits,
        "logit_mean": float(np.mean(logits)),
        "oos_degradation": float(np.mean(degrad)),
        "n_splits": len(ranks),
        "verdict": "PASS" if pbo <= 0.30 else "FAIL",
    }
