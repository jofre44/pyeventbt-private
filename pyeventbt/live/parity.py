"""Paridad backtest ↔ live (Etapa 6.1): replay diario señal a señal.

Métricas: % señales discordantes (≤2%), slippage real vs modelado p95
(≤2 ticks), tracking error diario de equity (≤0.5%). Divergencias típicas:
tz servidor, requotes, sufijos, tick size/lot step/swaps, cierres.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def signal_mismatch(live_signals: pd.DataFrame, bt_signals: pd.DataFrame,
                    tol_minutes: int = 65) -> dict:
    """Compara (timestamp, símbolo, lado). Discordante = sin pareja en ±tol."""
    live = live_signals.copy()
    bt = bt_signals.copy()
    live["ts"] = pd.to_datetime(live["timestamp"])
    bt["ts"] = pd.to_datetime(bt["timestamp"])
    matched = 0
    for _, s in live.iterrows():
        cand = bt[(bt["symbol"] == s["symbol"]) & (bt["side"] == s["side"])
                  & ((bt["ts"] - s["ts"]).abs() <= pd.Timedelta(minutes=tol_minutes))]
        if len(cand):
            matched += 1
            bt = bt.drop(cand.index[:1])
    total = max(len(live), len(bt_signals))
    mismatch = 1 - matched / total if total else 0.0
    return {"mismatch_pct": mismatch * 100, "matched": matched, "total": total}


def slippage_report(live_fills: pd.DataFrame, modeled_slippage_points: float) -> dict:
    """live_fills: [{expected, fill}] en puntos. p95 real vs modelado."""
    slip = (live_fills["fill"] - live_fills["expected"]).abs()
    p95 = float(slip.quantile(0.95)) if len(slip) else 0.0
    return {"slippage_p95": p95, "modeled": modeled_slippage_points,
            "ok": bool(p95 <= max(modeled_slippage_points, 2.0))}


def tracking_error(live_equity: pd.Series, bt_equity: pd.Series) -> dict:
    a, b = live_equity.align(bt_equity, join="inner")
    if len(a) < 2:
        return {"tracking_error_pct": float("nan")}
    te = float((a.pct_change() - b.pct_change()).std() * 100)
    return {"tracking_error_pct": te}


def parity_verdict(mismatch: dict, slip: dict, te: dict, cfg: dict) -> dict:
    import math

    te_v = te.get("tracking_error_pct")
    te_ok = te_v is not None and math.isfinite(te_v) and te_v <= cfg.get("max_daily_tracking_error_pct", 0.5)
    ok = (mismatch["mismatch_pct"] <= cfg.get("max_signal_mismatch_pct", 2.0)
          and slip.get("ok", False) and te_ok)
    return {"verdict": "PASS" if ok else "FAIL",
            "mismatch": mismatch, "slippage": slip, "tracking_error": te}
