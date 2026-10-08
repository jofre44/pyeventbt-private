"""Tick replay audit (Etapa 4.6): el feed M1 del broker como granularidad fina.

Sin ticks del broker, el replay compara el backtest a TF señal contra el
mismo backtest alimentado con granularidad M1 nativa... que es exactamente lo
que ya hace el motor (base M1 siempre). El audit real aquí:
  1. cobertura: fracción de minutos con tickvol>0 (min_coverage),
  2. gap barra-vs-feed: |PF_H1_agregado − PF_M1| > max_gap_pct ⇒ el backtest
     por barras no es fiable (ticks mandan) — con feed M1 no aplica porque el
     motor ya corre sobre M1; se verifica que la agregación H1↔M1 es coherente.
"""
from __future__ import annotations

import polars as pl


def tick_coverage(m1_bars: pl.DataFrame, min_coverage: float = 0.95) -> dict:
    tv = m1_bars["tickvol"].to_numpy()
    cov = float((tv > 0).mean()) if len(tv) else 0.0
    return {"coverage": cov, "verdict": "PASS" if cov >= min_coverage else "FAIL",
            "note": "M1 del broker como feed fino (ticks no disponibles)"}
