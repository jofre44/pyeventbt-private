"""Capacity ligero (Etapa 5): curva capital → decaimiento de Sharpe.

Modelo: el coste extra crece con sqrt(capital) (impacto aproximado para
retail/fx sin order-book). Decay ≤ max_decay_pct en el grid, si no la
estrategia no escala al capital objetivo. Regla de admisión: solo si mejora
Sharpe/DD de cartera neto de costes (beneficio marginal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sharpe(s: pd.Series) -> float:
    s = s.dropna()
    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return float(s.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


def _parse_cap(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip().upper().replace("K", "e3").replace("M", "e6")
    try:
        return float(x)
    except ValueError:
        return float(x.replace("e3", "")) * 1e3


def capacity_curve(daily: pd.Series, turnover_per_year: float,
                   capital_grid=("10k", "50k", "250k", "1M"),
                   impact_bps_per_sqrt_m: float = 5.0,
                   max_decay_pct: float = 25.0) -> dict:
    base = _sharpe(daily)
    rows = []
    for cap in capital_grid:
        c = _parse_cap(cap)
        extra_bps = impact_bps_per_sqrt_m * np.sqrt(c / 1e6)
        extra_daily = extra_bps / 1e4 * turnover_per_year / 252
        adj = daily - extra_daily
        sr = _sharpe(adj)
        decay = (base - sr) / abs(base) if base and abs(base) > 1e-9 else 0.0
        rows.append({"capital": cap, "extra_bps_day": extra_bps,
                     "sharpe_adj": sr, "decay_pct": decay * 100})
    ok = all(r["decay_pct"] <= max_decay_pct for r in rows
             if r["sharpe_adj"] == r["sharpe_adj"])
    return {"base_sharpe": base, "curve": rows,
            "verdict": "PASS" if ok else "FAIL"}


def marginal_benefit(port_before: dict, port_after: dict) -> dict:
    """¿Mejora Sharpe/DD neto de costes al añadir la estrategia?"""
    d_sharpe = port_after.get("sharpe", 0) - port_before.get("sharpe", 0)
    d_dd = port_before.get("maxdd", 0) - port_after.get("maxdd", 0)
    ok = d_sharpe > 0 and d_dd >= 0
    return {"d_sharpe": d_sharpe, "d_dd": d_dd,
            "verdict": "ADMIT" if ok else "REJECT"}
