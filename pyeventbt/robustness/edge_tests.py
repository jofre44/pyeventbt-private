"""Edge tests de data-snooping (Etapa 3.2c): White RC / Hansen SPA / Romano-Wolf.

H0: ningún trial de la familia tiene edge (benchmark 0). Stationary bootstrap
(≥2000 remuestreos) o permutación sign-flip sobre retornos diarios.
Devuelve p marginal por set (bootstrap) + p familiar (SPA max) para FDR/BH.
La familia de corrección es la del ledger (S_global), no solo el run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _studentize(mean: np.ndarray, std: np.ndarray, n: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.sqrt(n) * mean / np.where(std > 0, std, np.nan)
    return t


def edge_test(
    daily_returns: pd.DataFrame,
    method: str = "spa",
    n_resamples: int = 2000,
    bootstrap_p: float = 0.1,
    seed: int = 1500,
    use_stationary: bool = True,
) -> dict:
    """`daily_returns`: fechas × sets (shortlist top-40). Retorna
    {p_family, p_marginal: {col: p}, stat_obs, method, n_resamples}."""
    from pyeventbt.robustness.synthetic import stationary_bootstrap_matrix

    X = daily_returns.dropna(how="all").to_numpy(dtype=float)
    X = X[~np.isnan(X).all(axis=1)]
    n, k = X.shape
    if n < 30 or k == 0:
        return {"p_family": None, "p_marginal": {}, "method": method, "note": "muestra insuficiente"}
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)

    if method == "permutation":
        rng = np.random.default_rng(seed)
        t_obs = _studentize(mu, sd, n)
        max_obs = float(np.nanmax(t_obs))
        wins = 0
        marg_wins = np.zeros(k)
        for _ in range(n_resamples):
            signs = rng.choice([-1.0, 1.0], size=(n, k))
            Xb = X * signs
            tb = _studentize(np.nanmean(Xb, axis=0), np.nanstd(Xb, axis=0, ddof=1), n)
            if np.nanmax(tb) >= max_obs:
                wins += 1
            marg_wins += (tb >= t_obs).astype(float)
        p_fam = (1 + wins) / (1 + n_resamples)
        p_marg = {c: float((1 + w) / (1 + n_resamples)) for c, w in zip(daily_returns.columns, marg_wins)}
        return {"p_family": p_fam, "p_marginal": p_marg, "method": method, "n_resamples": n_resamples}

    # stationary bootstrap (spa | reality_check | romano_wolf)
    idx = stationary_bootstrap_matrix(X, bootstrap_p, seed, n_resamples)
    t_obs = _studentize(mu, sd, n) if method in ("spa", "romano_wolf") else mu
    center = mu  # imponer H0 (media cero) recentrando
    max_obs = float(np.nanmax(t_obs))
    boot_max = np.empty(n_resamples)
    marg_wins = np.zeros(k)
    for b in range(n_resamples):
        Xb = X[idx[b]] - center  # recentrado: H0 exacta
        stat = (_studentize(np.mean(Xb, axis=0), np.std(Xb, axis=0, ddof=1), n)
                if method in ("spa", "romano_wolf") else np.mean(Xb, axis=0))
        boot_max[b] = np.nanmax(stat)
        marg_wins += (stat >= t_obs).astype(float)
    p_fam = float((1 + np.sum(boot_max >= max_obs)) / (1 + n_resamples))
    p_marg = {c: float((1 + w) / (1 + n_resamples)) for c, w in zip(daily_returns.columns, marg_wins)}

    out = {"p_family": p_fam, "p_marginal": p_marg, "method": method,
           "n_resamples": n_resamples, "stat_obs_max": max_obs}
    if method == "romano_wolf":
        out["p_stepdown"] = _romano_wolf_stepdown(X, idx, t_obs)
    return out


def _romano_wolf_stepdown(X: np.ndarray, idx: np.ndarray, t_obs: np.ndarray) -> dict[int, float]:
    """Stepdown RW: elimina secuencialmente el mejor y re-evalúa el max
    bootstrap sobre los restantes. p ajustado por hipótesis (orden original)."""
    order = list(np.argsort(-np.nan_to_num(t_obs, nan=-np.inf)))
    remaining = order.copy()
    adj = {}
    for h in order:
        if not remaining:
            break
        boot_max = np.empty(idx.shape[0])
        for b in range(idx.shape[0]):
            Xb = X[idx[b]][:, remaining]
            mu_b = Xb.mean(axis=0) - X[:, remaining].mean(axis=0)  # H0
            sd_b = Xb.std(axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                tb = np.sqrt(X.shape[0]) * mu_b / np.where(sd_b > 0, sd_b, np.nan)
            boot_max[b] = np.nanmax(tb)
        p = float((1 + np.sum(boot_max >= t_obs[h])) / (1 + idx.shape[0]))
        adj[h] = max(p, max(adj.values()) if adj else 0.0)  # monotonicidad
        remaining.remove(h)
    return adj
