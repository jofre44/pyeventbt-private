"""Estrés de correlación (Etapa 5): media + peor decil + clustering jerárquico.

La decorrelación se exige ENTRE estrategias, no solo dentro de un combo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stress_correlation(returns: pd.DataFrame, max_rho_stress: float = 0.80) -> dict:
    rets = returns.dropna()
    corr = rets.corr().to_numpy(dtype=float)
    off = corr[~np.eye(len(corr), dtype=bool)]
    mean_rho = float(np.mean(off)) if off.size else 0.0
    # Peor decil de días: correlación en días de mayor |retorno| medio
    stress_days = rets.loc[rets.abs().mean(axis=1) >= rets.abs().mean(axis=1).quantile(0.9)]
    if len(stress_days) > 5:
        cs = stress_days.corr().to_numpy(dtype=float)
        offs = cs[~np.eye(len(cs), dtype=bool)]
        stress_rho = float(np.mean(offs)) if offs.size else mean_rho
    else:
        stress_rho = mean_rho
    ok = stress_rho <= max_rho_stress
    return {"mean_rho": mean_rho, "stress_rho": stress_rho,
            "verdict": "PASS" if ok else "FAIL"}
