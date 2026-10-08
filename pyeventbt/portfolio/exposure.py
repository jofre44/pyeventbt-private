"""Exposición neta por divisa (Etapa 5): EURUSD largo = +EUR −USD.

Evita que cinco pares "distintos" sean la misma apuesta corto USD.
"""
from __future__ import annotations

import pandas as pd


def currency_exposure(positions: pd.DataFrame, limit_lots: float) -> dict:
    """`positions`: [{symbol, direction(+1/-1), lots}]. Devuelve neto por
    divisa + breach del límite."""
    net: dict[str, float] = {}
    for _, p in positions.iterrows():
        sym = str(p["symbol"])
        if len(sym) != 6:
            continue
        base, quote = sym[:3], sym[3:]
        v = float(p["direction"]) * float(p["lots"])
        net[base] = net.get(base, 0.0) + v
        net[quote] = net.get(quote, 0.0) - v
    breach = {c: v for c, v in net.items() if abs(v) > limit_lots}
    return {"net_by_currency": net, "breach": breach,
            "verdict": "PASS" if not breach else "FAIL"}
