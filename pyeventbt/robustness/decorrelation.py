"""Decorrelación + jitter (Etapa 4.1–4.5).

Correlación sobre RETORNOS diarios (nunca niveles), pearson|spearman.
UNILATERAL: rho <= thr contra lo aceptado; rho negativo = diversificador.
Score = 0.4·sharpe + 0.3·pf_winsor_p95 + 0.2·r2 − 0.1·turnover (min-max por combo).
Greedy hasta max_sets. Reporta n_efectivo = 1/sum(w²).
Jitter ±10%: min_green vecinos con sharpe>0 y pf>=1.0 (confirmación, no filtro).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def composite_score(df: pd.DataFrame, weights: dict, pf_cap_pct: float = 95.0) -> pd.Series:
    w = weights
    pf = df["pf"].replace([np.inf, -np.inf], np.nan)
    cap = float(pf.quantile(pf_cap_pct / 100.0)) if pf.notna().any() else 1.0
    pf_w = pf.clip(upper=cap)

    def _norm(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        lo, hi = float(s.min()), float(s.max())
        if not np.isfinite(lo) or hi == lo:
            return pd.Series(0.0, index=s.index)
        return (s - lo) / (hi - lo)

    return (w.get("sharpe_net", 0.4) * _norm(df["sharpe_net"].fillna(0))
            + w.get("profit_factor", 0.3) * _norm(pf_w.fillna(0))
            + w.get("r2", 0.2) * _norm(df["r2"].fillna(0))
            - w.get("turnover_penalty", 0.1) * _norm(df["turnover"].fillna(0)))


def greedy_select(
    stats_df: pd.DataFrame,
    daily_returns: pd.DataFrame,
    cfg: dict,
) -> dict:
    """`stats_df`: set_id + n_trades + sharpe_net + pf + r2 + turnover.
    `daily_returns`: fechas × set_id."""
    thr = float(cfg.get("correlation_threshold", 0.7))
    max_sets = int(cfg.get("max_sets", 5))
    min_trades = int(cfg.get("min_trades", 30))
    method = cfg.get("corr_method", "pearson")
    df = stats_df[stats_df["n_trades"] >= min_trades].copy()
    df = df[df["daily_var"] > 0] if "daily_var" in df.columns else df
    if df.empty:
        return {"selected": [], "n_effective": 0.0, "note": "prefiltro vacío"}
    df["score"] = composite_score(df, cfg.get("weights", {}), cfg.get("pf_cap_percentile", 95.0))
    df = df.sort_values("score", ascending=False)
    rets = daily_returns.reindex(columns=df["set_id"].tolist())
    if method == "spearman":
        corr = rets.rank().corr(method="pearson")
    else:
        corr = rets.corr(method="pearson")
    accepted: list[str] = []
    for sid in df["set_id"]:
        if len(accepted) >= max_sets:
            break
        if not accepted:
            accepted.append(sid)
            continue
        rhos = [corr.loc[sid, a] for a in accepted]
        rhos = [r for r in rhos if r is not None and np.isfinite(r)]
        if rhos and max(rhos) > thr:  # unilateral: solo el positivo bloquea
            continue
        accepted.append(sid)
    w = np.ones(len(accepted)) / max(len(accepted), 1)
    n_eff = float(1 / np.sum(w**2)) if len(accepted) else 0.0
    return {"selected": accepted, "n_effective": n_eff,
            "warn_concentrated": bool(n_eff < 0.6 * len(accepted)) if accepted else False,
            "ranking": df[["set_id", "score"]].to_dict(orient="records")}


def jitter_grid(params: dict, pct: float = 10.0) -> list[dict]:
    """Vecinos ±pct por parámetro (una dimensión cada vez + esquinas)."""
    keys = [k for k in params if not k.startswith("_")]
    out = []
    for k in keys:
        v = params[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        d = abs(v) * pct / 100.0 or pct / 100.0
        for s in (-1, 1):
            nb = dict(params)
            nb[k] = type(v)(v + s * d) if isinstance(v, int) else v + s * d
            out.append(nb)
    # Esquinas: todos + y todos −
    for s in (-1, 1):
        nb = {}
        for k in keys:
            v = params[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                nb[k] = v
                continue
            d = abs(v) * pct / 100.0 or pct / 100.0
            nb[k] = type(v)(v + s * d) if isinstance(v, int) else v + s * d
        out.append(nb)
    return out


def jitter_verdict(results: list[dict], min_green: float = 0.8) -> dict:
    """`results`: [{sharpe_net, pf}]. Verde si ≥min_green con sharpe>0 y pf≥1."""
    if not results:
        return {"verdict": "FAIL", "green_share": 0.0}
    green = sum(1 for r in results
                if (r.get("sharpe_net") or -np.inf) > 0 and (r.get("pf") or -np.inf) >= 1.0)
    share = green / len(results)
    return {"verdict": "PASS" if share >= min_green else "FAIL",
            "green_share": share, "n": len(results)}
