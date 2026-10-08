"""Potencia estadística / MinTRL (Etapa 0.4).

SE(Sharpe anual) ≈ sqrt((1 + 0.5*SR²) / años).
MinTRL: años para que target_sharpe sea significativo a `alpha`
(ajuste simple por skew/kurtosis vía factor de vara si se proveen).
Si años_IS < MinTRL => INCONCLUSIVE (nunca aprobar por defecto).
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import pandas as pd


def se_sharpe(annual_sr: float, years: float) -> float:
    if years <= 0:
        return float("nan")
    return math.sqrt((1.0 + 0.5 * annual_sr**2) / years)


def min_trl_years(target_sharpe: float, alpha: float = 0.05, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Años mínimos para detectar target_sharpe a nivel alpha (una cola)."""
    if target_sharpe <= 0:
        return float("inf")
    z = NormalDist().inv_cdf(1.0 - alpha)
    base = (z**2 * (1.0 + 0.5 * target_sharpe**2)) / target_sharpe**2
    # Penalización por colas pesadas/asimetría (Bailey-Lopez de Prado aprox simple)
    adj = 1.0 + 0.25 * abs(skew) + 0.10 * max(0.0, kurt - 3.0)
    return float(base * adj)


def power_report(
    daily_returns: pd.Series | None,
    n_trades: int,
    years: float,
    target_sharpe: float = 0.8,
    alpha: float = 0.05,
) -> dict[str, Any]:
    skew = float(daily_returns.skew()) if daily_returns is not None and len(daily_returns) > 2 else 0.0
    kurt = float(daily_returns.kurt() + 3.0) if daily_returns is not None and len(daily_returns) > 3 else 3.0
    mtrl = min_trl_years(target_sharpe, alpha, skew, kurt)
    verdict = "READY" if years >= mtrl else "INCONCLUSIVE"
    return {
        "years_effective": round(float(years), 2),
        "n_trades": int(n_trades),
        "se_sharpe_at_target": round(se_sharpe(target_sharpe, years), 3) if years > 0 else None,
        "mintrl_years": round(mtrl, 2),
        "target_sharpe": target_sharpe,
        "alpha": alpha,
        "verdict": verdict,
    }
