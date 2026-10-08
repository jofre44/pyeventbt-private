"""Control negativo (Etapa 3.2b): ¿gana donde no hay nada que explotar?

Pool: Sharpe medio nulo ≈ 0 (|media| ≤ max_abs) y fracción de sets nulos
"significativos" (sharpe > sig_threshold) ≤ max_pos_rate. Si no: STOP del run,
hay sesgo (look-ahead, costes, intrabarra optimista). Nivel set: sharpe nulo
positivo significativo => FAIL del set.
"""
from __future__ import annotations

import numpy as np


def null_pool_verdict(null_sharpes: list[float], max_abs_mean: float = 0.15,
                      max_pos_rate: float = 0.10, sig_threshold: float = 0.5) -> dict:
    s = np.asarray([v for v in null_sharpes if v is not None and np.isfinite(v)], dtype=float)
    if len(s) == 0:
        return {"verdict": "NODATA", "n": 0}
    mean = float(np.mean(s))
    pos_rate = float(np.mean(s > sig_threshold))
    # Unilateral hacia arriba: lo que invalida el run es GANAR en nulo (sesgo).
    # Perder en nulo es cost-drag esperado (spread/comisión sobre ruido).
    if mean > max_abs_mean or pos_rate > max_pos_rate:
        verdict, stop = "STOP", True
    else:
        verdict, stop = "PASS", False
    return {
        "verdict": verdict, "n": int(len(s)),
        "null_sharpe_mean": mean, "null_pos_rate": pos_rate,
        "cost_drag": bool(mean < -max_abs_mean),
    }


def null_set_verdict(null_sharpes: list[float], sig_threshold: float = 0.5) -> dict:
    s = np.asarray([v for v in null_sharpes if v is not None and np.isfinite(v)], dtype=float)
    if len(s) == 0:
        return {"verdict": "NODATA"}
    mean = float(np.mean(s))
    # Significativo y positivo => vive de un artefacto
    if mean > sig_threshold:
        return {"verdict": "FAIL", "null_sharpe_mean": mean}
    return {"verdict": "PASS", "null_sharpe_mean": mean}
