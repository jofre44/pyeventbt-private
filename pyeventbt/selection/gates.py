"""Puertas ANTES del lockbox (Etapa 4.7): PF≥1.3, R2≥0.5, DD<365d, ≤10/12 rojos,
régimen ≥60% celdas positivas y ninguna >50% del PnL.

PF 1.1–1.3: banda de retest (3 salidas alternativas MA/BB/RSI ≥1.3).
Todo fallo aquí descarta SIN gastar el lockbox.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _max_dd_days(pnl: pd.DataFrame) -> float:
    try:
        eq = pnl["EQUITY"].astype(float).sort_index()
        roll = eq.cummax()
        dd = eq / roll - 1.0
        trough = dd.idxmin()
        peak_val = float(eq.loc[:trough].max())
        peak = eq.loc[:trough][eq.loc[:trough] == peak_val].index[-1]
        after = eq.loc[trough:]
        rec = after[after >= peak_val]
        end = rec.index[0] if len(rec) else eq.index[-1]
        return (pd.Timestamp(end) - pd.Timestamp(peak)).total_seconds() / 86400.0
    except Exception:
        return float("nan")


def _red_months(pnl: pd.DataFrame) -> int:
    try:
        eq = pnl["EQUITY"].astype(float).sort_index()
        m = eq.resample("ME").last().pct_change().dropna()
        return int((m.tail(12) < 0).sum())
    except Exception:
        return 99


def apply_gates(summary: dict, pnl: pd.DataFrame, daily: pd.Series,
                gates_cfg: dict, retest_pf: float | None = None) -> dict:
    checks: dict[str, dict] = {}
    pf = summary.get("pf") or 0.0
    if pf >= gates_cfg.get("pf_min", 1.3):
        checks["pf"] = {"verdict": "PASS", "pf": pf}
    elif (gates_cfg.get("pf_retest_band", [1.1, 1.3])[0] <= pf
          and retest_pf is not None and retest_pf >= gates_cfg.get("pf_min", 1.3)):
        checks["pf"] = {"verdict": "PASS_RETEST", "pf": pf, "retest_pf": retest_pf}
    else:
        checks["pf"] = {"verdict": "FAIL", "pf": pf}
    r2 = summary.get("r2")
    checks["r2"] = {"verdict": "PASS" if (r2 is not None and r2 >= gates_cfg.get("r2_min", 0.5)) else "FAIL",
                    "r2": r2}
    dd_days = _max_dd_days(pnl)
    checks["dd_duration"] = {"verdict": "PASS" if dd_days < gates_cfg.get("max_dd_days", 365) else "FAIL",
                             "dd_days": dd_days}
    reds = _red_months(pnl)
    checks["red_months"] = {"verdict": "PASS" if reds <= gates_cfg.get("max_red_months", 10) else "FAIL",
                            "red_months_12": reds}
    from pyeventbt.analytics.regimes import regime_report

    reg = regime_report(daily, **gates_cfg.get("regimes", {}))
    checks["regime"] = {"verdict": reg["verdict"], **{k: v for k, v in reg.items() if k != "cells"}}
    passed = all(c["verdict"] in ("PASS", "PASS_RETEST") for c in checks.values())
    return {"verdict": "PASS" if passed else "FAIL", "checks": checks}
