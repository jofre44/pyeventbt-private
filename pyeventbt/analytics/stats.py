"""Métricas resumen a partir de (pnl, trades). Fuente única para todo el pipeline.

`summary()` canónico por trial/combo:
  sharpe_net (anualizado, Rf=0, retornos diarios netos), pf, n_trades,
  maxdd_pct, exposure_pct, r2 (R² log-equity vs tiempo), calmar, turnover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def to_daily_returns(pnl: pd.DataFrame) -> pd.Series:
    if pnl is None or pnl.empty or "EQUITY" not in pnl.columns:
        return pd.Series(dtype=float)
    eq = pnl["EQUITY"].astype(float).sort_index()
    if not isinstance(eq.index, pd.DatetimeIndex):
        try:
            eq.index = pd.to_datetime(eq.index)
        except Exception:
            return pd.Series(dtype=float)
    d = eq.resample("1D").last().dropna()
    if len(d) < 2:
        return pd.Series(dtype=float)
    prev = d.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((prev > 0) & np.isfinite(prev), d / prev - 1.0, np.nan)
    return pd.Series(r, index=d.index).replace([np.inf, -np.inf], np.nan).dropna()


def sharpe_net(daily: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    if daily is None or len(daily) < 2:
        return float("nan")
    sd = float(daily.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(daily.mean() / sd * np.sqrt(periods_per_year))


def _closed_net(trades: pd.DataFrame) -> pd.Series:
    """Columna NET por trade cerrado. Acepta el formato crudo de PyEventBT
    (DEAL IN/OUT con GROSS_PROFIT/COMMISSION/SWAP/FEE) o un frame con NET."""
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    if "NET" in trades.columns:
        return pd.to_numeric(trades["NET"], errors="coerce").dropna().astype(float)
    if "DEAL" not in trades.columns:
        return pd.Series(dtype=float)
    outs = trades[trades["DEAL"].astype(str).str.upper() == "OUT"].copy()
    if outs.empty:
        return pd.Series(dtype=float)
    ins = trades[trades["DEAL"].astype(str).str.upper() == "IN"].copy()
    in_cost = {}
    if not ins.empty and "POSITION_ID" in ins.columns:
        for pid, g in ins.groupby("POSITION_ID"):
            in_cost[pid] = sum(
                float(pd.to_numeric(g[c], errors="coerce").fillna(0.0).sum())
                for c in ("COMMISSION", "SWAP", "FEE") if c in g.columns
            )
    nets = []
    for _, o in outs.iterrows():
        pid = o.get("POSITION_ID")
        gross = float(o.get("GROSS_PROFIT", 0.0) or 0.0)
        o_comm = float(o.get("COMMISSION", 0.0) or 0.0)
        o_swap = float(o.get("SWAP", 0.0) or 0.0)
        o_fee = float(o.get("FEE", 0.0) or 0.0)
        nets.append(gross - o_comm + o_swap - o_fee - in_cost.get(pid, 0.0))
    return pd.Series(nets, dtype=float)


def profit_factor(trades: pd.DataFrame) -> float:
    net = _closed_net(trades)
    if net.empty:
        return float("nan")
    won = float(net[net > 0].sum())
    lost = float(-net[net < 0].sum())
    if lost <= 0:
        return float("inf") if won > 0 else float("nan")
    return float(won / lost)


def maxdd_pct(pnl: pd.DataFrame) -> float:
    if pnl is None or pnl.empty or "EQUITY" not in pnl.columns:
        return float("nan")
    eq = pnl["EQUITY"].astype(float).sort_index()
    roll = eq.cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(roll > 0, eq / roll - 1.0, np.nan)
    return float(np.nanmin(dd) * 100.0)


def exposure_pct(pnl: pd.DataFrame) -> float:
    try:
        eq = pnl["EQUITY"].astype(float)
        bal = pnl["BALANCE"].astype(float) if "BALANCE" in pnl.columns else eq
        return float((((eq - bal).abs() > 1e-9).mean() * 100.0))
    except Exception:
        return float("nan")


def r2_trend(pnl: pd.DataFrame) -> float:
    """R² de log(equity) vs tiempo. Curva plana/ruido => ~0; tendencia limpia => ~1."""
    try:
        eq = pnl["EQUITY"].astype(float).sort_index()
        eq = eq[eq > 0]
        if len(eq) < 10:
            return float("nan")
        y = np.log(eq.values)
        x = np.arange(len(y))
        if np.std(y) == 0:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return float(corr**2) if np.isfinite(corr) else float("nan")
    except Exception:
        return float("nan")


def summarize(pnl: pd.DataFrame, trades: pd.DataFrame, cash: float = 10000.0) -> dict:
    daily = to_daily_returns(pnl)
    net = _closed_net(trades)
    n = int(len(net))
    years = max(len(daily) / 365.25, 1 / 365.25) if len(daily) else float("nan")
    pf = profit_factor(trades)
    return {
        "sharpe_net": sharpe_net(daily),
        "pf": float(pf) if pf is not None else float("nan"),
        "n_trades": n,
        "maxdd_pct": abs(maxdd_pct(pnl)),
        "exposure_pct": exposure_pct(pnl),
        "r2": r2_trend(pnl),
        "calmar": float("nan"),
        "turnover": float(n / years) if n and years and np.isfinite(years) else 0.0,
        "n_days": int(len(daily)),
    }
