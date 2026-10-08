"""Asignación HRP + risk parity con Ledoit-Wolf + vol targeting + Kelly cap (Etapa 5).

Universo = todos los promoted.csv (retornos diarios netos). HRP de López de
Prado sobre correlación; shrinkage Ledoit-Wolf manual; vol targeting anual;
Kelly fraccional como TECHO (cap 0.25); peso máximo por estrategia;
rebalanceo mensual con banda (la banda se evalúa en el programador, aquí el peso).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf_cov(X: pd.DataFrame, delta: float | None = None) -> pd.DataFrame:
    """Shrinkage a identidad escalada. delta automático si no se da."""
    R = X.dropna()
    n, k = R.shape
    S = np.cov(R.to_numpy(dtype=float).T)
    mu = np.trace(S) / k
    F = mu * np.eye(k)
    if delta is None:
        # Heurística LW: más shrinkage con pocas muestras
        delta = min(0.9, max(0.1, 1.0 - n / (n + k * 10)))
    shrunk = (1 - delta) * S + delta * F
    return pd.DataFrame(shrunk, index=X.columns, columns=X.columns)


def _hrp_weights(corr: pd.DataFrame) -> pd.Series:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    dist = np.sqrt(np.clip(2 * (1 - corr.to_numpy(dtype=float)), 0, 4))
    link = linkage(squareform(dist, checks=False), method="single")
    n = len(corr)
    order = []
    _recurse(link, 2 * n - 2, n, order)  # raíz del dendrograma
    cov = corr.to_numpy(dtype=float)
    vols = np.sqrt(np.diag(cov))
    vols[vols <= 0] = 1.0
    w = pd.Series(1.0, index=[corr.columns[i] for i in order])
    clusters = [[c] for c in w.index]
    weights = pd.Series(1.0, index=w.index)
    while len(clusters) > 1:
        # fusiona pares adyacentes por varianza inversa del cluster
        new_clusters = []
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                new_clusters.append(clusters[i])
                continue
            c1, c2 = clusters[i], clusters[i + 1]
            v1 = _cluster_var(cov, corr.columns, c1)
            v2 = _cluster_var(cov, corr.columns, c2)
            a = 1 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
            weights[c1] *= a
            weights[c2] *= 1 - a
            new_clusters.append(c1 + c2)
        clusters = new_clusters
    return weights / weights.sum()


def _recurse(link, node, n, order):
    if node < n:
        order.append(int(node))
        return
    l, r = int(link[node - n, 0]), int(link[node - n, 1])
    _recurse(link, l, n, order)
    _recurse(link, r, n, order)


def _cluster_var(cov: np.ndarray, cols, members) -> float:
    idx = [list(cols).index(m) for m in members]
    sub = cov[np.ix_(idx, idx)]
    ivp = 1 / np.diag(sub)
    ivp = np.where(np.isfinite(ivp) & (ivp > 0), ivp, 0)
    if ivp.sum() <= 0:
        return float(np.mean(np.diag(sub)))
    w = ivp / ivp.sum()
    return float(w @ sub @ w)


def allocate(returns: pd.DataFrame, cfg: dict) -> dict:
    """`returns`: fechas × estrategias (netos). Devuelve pesos + diagnósticos."""
    a = cfg.get("allocation", cfg)
    rets = returns.dropna(how="all", axis=0).fillna(0.0)
    cov = ledoit_wolf_cov(rets)
    std = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov / np.outer(std, std)
    corr = pd.DataFrame(np.nan_to_num(corr.to_numpy(), nan=0.0),
                        index=rets.columns, columns=rets.columns)
    method = a.get("method", "hrp")
    w = _hrp_weights(corr) if method == "hrp" else pd.Series(1 / len(rets.columns), index=rets.columns)
    # Kelly fraccional como techo: w <= min(half-Kelly, cap) (unidades anuales)
    mu = rets.mean().to_numpy() * 252
    var_ann = np.diag(cov.to_numpy()) * 252
    with np.errstate(all="ignore"):
        kelly = np.where(var_ann > 0, mu / np.maximum(var_ann, 1e-9), 0.0) * 0.5
    kelly = np.clip(kelly, 0, None)
    cap = float(a.get("kelly_fraction_cap", 0.25))
    w = pd.Series(np.minimum(w.to_numpy(), np.minimum(kelly, cap)), index=w.index)
    if w.sum() <= 0:  # todo el universo con Kelly<=0: reparte igual (no apalanca)
        w = pd.Series(1 / len(rets.columns), index=rets.columns)
    # Peso máximo por estrategia (proyección iterativa al simplex capado)
    mw = float(a.get("max_weight_per_strategy", 0.30))
    if mw * len(w) < 1.0:
        raise ValueError(f"max_weight_per_strategy={mw} infactible con {len(w)} estrategias "
                         f"(suma máxima {mw * len(w):.2f} < 1)")
    w = w / w.sum()
    for _ in range(100):
        over = w[w > mw]
        if over.empty:
            break
        w[over.index] = mw
        rest = w[w < mw]
        if rest.empty:
            w = w / w.sum()
            break
        room = 1 - mw * len(over)
        if rest.sum() <= 0:
            # Kelly dejó al resto en 0: reparte el hueco a partes iguales
            # (el techo Kelly es guía; el simplex factible manda).
            w[rest.index] = room / len(rest)
        else:
            w[rest.index] = rest / rest.sum() * room
    # Vol targeting anual
    port_vol = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()) * np.sqrt(252))
    target = float(a.get("vol_target_annual", 0.10))
    lev = target / port_vol if port_vol > 0 else 1.0
    return {"weights": w.to_dict(), "leverage": float(lev),
            "portfolio_vol_ann": port_vol, "method": method,
            "n_strategies": len(w)}
