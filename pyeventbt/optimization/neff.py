"""N efectivo de tests (Etapa 2.7): con TPE los trials se concentran en zonas
buenas; S=400 no son 400 hipótesis independientes.

Agrupa trials por correlación de retornos diarios (jerárquico sobre 1−rho) →
`N_eff` = nº de clusters conductuales. El DSR se reporta con S (conservador)
y N_eff (optimista); el veto usa N_eff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_pandas(daily_returns) -> "pd.DataFrame":
    import pandas as pd

    if isinstance(daily_returns, pd.DataFrame):
        df = daily_returns.copy()
    else:  # polars (evita pyarrow): dict -> pandas
        d = daily_returns.to_dict(as_series=False)
        idx = d.pop("date", None)
        df = pd.DataFrame(d, index=pd.to_datetime(idx) if idx is not None else None)
    df.index = pd.to_datetime(df.index)
    return df


def neff_from_returns(
    daily_returns,
    corr_threshold: float = 0.7,
    method: str = "onc_returns",
) -> dict:
    """`daily_returns`: pandas o polars, índice/fechas × columnas trial_*.
    Devuelve {S, N_eff, method, n_clusters, mean_abs_corr}."""
    df = _as_pandas(daily_returns)
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    S = int(df.shape[1])
    if S == 0:
        return {"S": 0, "N_eff": 0, "method": method}
    if S == 1:
        return {"S": 1, "N_eff": 1, "method": method}
    corr = df.corr(method="pearson").to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - corr_threshold, criterion="distance")
    n_eff = int(len(set(labels.tolist())))
    off = corr[~np.eye(S, dtype=bool)]
    return {
        "S": S, "N_eff": n_eff, "method": method,
        "n_clusters": n_eff, "corr_threshold": corr_threshold,
        "mean_abs_corr": float(np.mean(np.abs(off))) if off.size else 0.0,
    }
