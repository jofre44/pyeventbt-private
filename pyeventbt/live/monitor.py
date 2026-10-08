"""Monitorización y decay (Etapa 6.3): Sharpe rodante vs p5, DD vs envolvente MC
(alerta p95, halt p99), CUSUM/SPRT, deriva de frecuencia/duración/hit/slippage/
spread, KS de retornos. Health score GREEN/YELLOW/ORANGE/RED. Retirada
pre-registrada: DD > p99 | 2 CUSUM | Sharpe < p5 30d → paper/shadow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_sharpe(daily: pd.Series, window_days: int = 60) -> pd.Series:
    r = daily.dropna()
    return r.rolling(window_days).mean() / r.rolling(window_days).std(ddof=1) * np.sqrt(252)


def cusum_alerts(trade_returns: pd.Series, k: float = 0.5, h: float = 5.0) -> dict:
    """CUSUM tabular bilateral sobre retornos por trade estandarizados."""
    x = trade_returns.dropna().to_numpy(dtype=float)
    if len(x) < 10 or float(np.std(x)) <= 0:
        return {"alarms": 0, "verdict": "OK"}
    z = (x - np.mean(x)) / np.std(x)
    gp = gn = 0.0
    alarms = 0
    for v in z:
        gp = max(0, gp + v - k)
        gn = min(0, gn + v + k)
        if gp > h or gn < -h:
            alarms += 1
            gp = gn = 0.0
    return {"alarms": alarms, "verdict": "ALARM" if alarms else "OK"}


def ks_drift(live_rets: pd.Series, ref_rets: pd.Series) -> dict:
    from scipy.stats import ks_2samp

    try:
        stat, p = ks_2samp(live_rets.dropna(), ref_rets.dropna())
        return {"ks_stat": float(stat), "ks_p": float(p),
                "verdict": "OK" if p > 0.05 else "DRIFT"}
    except Exception:
        return {"ks_stat": float("nan"), "ks_p": float("nan"), "verdict": "NODATA"}


def health_score(checks: dict) -> dict:
    """checks: {performance, execution, statistical, regime} en 0..1.
    GREEN ≥0.8, YELLOW ≥0.6, ORANGE ≥0.4, si no RED."""
    s = float(np.mean(list(checks.values()))) if checks else 0.0
    level = "GREEN" if s >= 0.8 else ("YELLOW" if s >= 0.6 else ("ORANGE" if s >= 0.4 else "RED"))
    return {"score": s, "level": level}


def retirement_check(state: dict, cfg: dict) -> dict:
    """state: {dd_vs_p99(bool), cusum_alarms(int), sharpe_below_p5_days(int)}."""
    trig = []
    if state.get("dd_vs_p99"):
        trig.append("dd>p99")
    if state.get("cusum_alarms", 0) >= 2:
        trig.append("2x CUSUM")
    if state.get("sharpe_below_p5_days", 0) >= cfg.get("sharpe_below_p5_days", 30):
        trig.append("sharpe<p5 30d")
    return {"retire": bool(trig), "triggers": trig,
            "action": "paper/shadow, run nuevo completo" if trig else "continue"}
