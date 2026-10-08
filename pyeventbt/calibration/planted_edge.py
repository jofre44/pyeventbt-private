"""Edge plantado (Meta-etapa M): inyecta drift condicionado para Sharpe objetivo.

Mide la POTENCIA del pipeline: ¿sobrevive un edge conocido de SR {0.3,0.6,1.0}?
El drift se distribuye uniforme en los retornos M1 de los días con señal.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def drift_for_sharpe(sharpe_target: float, daily_vol: float) -> float:
    """Drift diario necesario para Sharpe anualizado objetivo."""
    return float(sharpe_target * daily_vol / np.sqrt(252))


def plant_drift_m1(m1: pl.DataFrame, signal_dates: set, sharpe_target: float,
                   daily_vol: float, seed: int = 0) -> pl.DataFrame:
    delta = drift_for_sharpe(sharpe_target, daily_vol)
    df = m1.sort("datetime").with_columns(pl.col("datetime").dt.date().alias("_d"))
    out_frames = []
    anchor = None  # close plantado del día anterior: el drift COMPONE entre días
    for day, g in df.group_by("_d", maintain_order=True):
        g = g.sort("datetime")
        closes = g["close"].to_numpy().astype(float)
        if anchor is not None:
            scale = anchor / closes[0]
            g = g.with_columns([
                (pl.col("open") * scale).alias("open"),
                (pl.col("high") * scale).alias("high"),
                (pl.col("low") * scale).alias("low"),
                (pl.col("close") * scale).alias("close"),
            ])
            closes = closes * scale
        if day[0] in signal_dates:
            n = len(g)
            extra = delta / n
            r = np.diff(np.log(closes), prepend=np.log(closes[0])) + extra
            new_c = closes[0] * np.exp(np.cumsum(r))
            factor = new_c / closes
            g = g.with_columns([
                (pl.col("open") * factor).alias("open"),
                (pl.col("high") * factor).alias("high"),
                (pl.col("low") * factor).alias("low"),
                pl.Series("close", new_c),
            ])
            anchor = float(new_c[-1])
        else:
            anchor = float(closes[-1])
        out_frames.append(g)
    return pl.concat(out_frames).drop("_d").sort("datetime")
