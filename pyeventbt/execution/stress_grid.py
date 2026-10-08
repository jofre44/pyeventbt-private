"""Grid de estrés de ejecución (Etapa 4.6): spread×slippage sobre IS.

Re-ejecuta el set en cada celda (mismo código, costes distintos).
Pasa si PF peor celda ≥ min_pf_worst_cell y decay de Sharpe ≤ 50%.
Sin ticks reales del broker: el feed M1 del broker ES el replay disponible
(cobertura documentada en el manifiesto); la celda peor manda.
"""
from __future__ import annotations

import numpy as np


def stress_verdict(cell_results: list[dict], pass_cfg: dict) -> dict:
    """`cell_results`: [{cell, sharpe_net, pf}]. Base = primera celda."""
    if not cell_results:
        return {"verdict": "NODATA"}
    base = cell_results[0]
    pfs = [c.get("pf") for c in cell_results if c.get("pf") is not None and np.isfinite(c["pf"])]
    worst_pf = min(pfs) if pfs else float("nan")
    s0 = base.get("sharpe_net")
    decay = 0.0
    if s0 is not None and np.isfinite(s0) and abs(s0) > 1e-9:
        worst_sr = min(c["sharpe_net"] for c in cell_results
                       if c.get("sharpe_net") is not None and np.isfinite(c["sharpe_net"]))
        decay = max(0.0, (s0 - worst_sr) / abs(s0))
    ok = (worst_pf >= pass_cfg.get("min_pf_worst_cell", 1.05)
          and decay <= pass_cfg.get("max_sharpe_decay_pct", 50) / 100.0)
    return {"verdict": "PASS" if ok else "FAIL", "worst_pf": worst_pf,
            "sharpe_decay": decay, "n_cells": len(cell_results)}
