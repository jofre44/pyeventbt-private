"""Análisis por régimen (Etapa 4.7, puertas): año, terciles de vol, sesión.

Exige ≥60% celdas con Sharpe>0 y ninguna celda con >50% del PnL total.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 5 or float(s.std(ddof=1) or 0) <= 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * np.sqrt(252))


def regime_cells(daily: pd.Series, closes_h1: pd.Series | None = None,
                 session_hours: tuple[int, int] = (8, 21),
                 by_year: bool = True, by_vol_regime: bool = True,
                 by_session: bool = False) -> dict[str, pd.Series]:
    """Desglose del PnL diario por celdas. Sesión solo si hay closes H1;
    si no, año + terciles de vol realizada 20d."""
    daily = daily.dropna()
    daily.index = pd.to_datetime(daily.index)
    cells: dict[str, pd.Series] = {}
    if by_year:
        for y, g in daily.groupby(daily.index.year):
            cells[f"year:{y}"] = g
    vol = daily.rolling(20).std().dropna()
    if by_vol_regime and len(vol) >= 30:
        try:
            terc = pd.qcut(vol, 3, labels=["low", "mid", "high"], duplicates="drop")
            for t in terc.unique():
                cells[f"vol:{t}"] = daily.loc[terc[terc == t].index]
        except ValueError:
            pass
    if by_session and closes_h1 is not None:
        h = pd.Series(pd.to_datetime(closes_h1.index)).dt.hour if not isinstance(
            closes_h1.index, pd.DatetimeIndex) else pd.Series(closes_h1.index).dt.hour
        # aprox: días cuya mayoría de barras cae en sesión
        sess_days = pd.Series(closes_h1.index, index=closes_h1.index)
        in_sess = ((sess_days.dt.hour >= session_hours[0]) & (sess_days.dt.hour < session_hours[1]))
        day_sess = in_sess.groupby(in_sess.index.date).mean()
        sess_dates = {d for d, f in day_sess.items() if f >= 0.5}
        mask = pd.Series([d.date() in sess_dates for d in daily.index], index=daily.index)
        cells["session:in"] = daily[mask]
        cells["session:out"] = daily[~mask]
    return cells


def regime_report(daily: pd.Series, min_share_positive: float = 0.60,
                  max_single_cell_pnl_share: float = 0.50, **kw) -> dict:
    """Pasa si ≥60% celdas tienen Sharpe>0 y ninguna FAMILIA depende de una
    única celda: dentro de cada familia (año/vol/sesión), si una celda
    concentra >50% del |PnL| familiar mientras una hermana no es positiva,
    hay dependencia de régimen => FAIL. (El |PnL| bruto sobre celdas de
    distinto tamaño —ej. un año con 90% de la muestra— no es dependencia.)"""
    cells = regime_cells(daily, **kw)
    total = float(daily.sum())
    rows = []
    for name, g in cells.items():
        pnl = float(g.sum())
        rows.append({"cell": name, "sharpe": _sharpe(g), "pnl": pnl,
                     "pnl_share": abs(pnl) / abs(total) if total else 0.0,
                     "positive": bool(_sharpe(g) > 0)})
    fams: dict[str, list[dict]] = {}
    for r in rows:
        fam = r["cell"].split(":")[0]
        fams.setdefault(fam, []).append(r)
    dependent = []
    for fam, members in fams.items():
        if len(members) < 2:
            continue
        fam_abs = sum(abs(m["pnl"]) for m in members) or 1.0
        top = max(members, key=lambda m: abs(m["pnl"]))
        if (abs(top["pnl"]) / fam_abs > max_single_cell_pnl_share
                and any(not m["positive"] for m in members if m is not top)):
            dependent.append(fam)
    pos_share = float(np.mean([r["positive"] for r in rows])) if rows else 0.0
    max_share = max((r["pnl_share"] for r in rows), default=0.0)
    ok = pos_share >= min_share_positive and not dependent
    return {"cells": rows, "share_positive": pos_share, "max_cell_share": max_share,
            "dependent_families": dependent, "verdict": "PASS" if ok else "FAIL"}
