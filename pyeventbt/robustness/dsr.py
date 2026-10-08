"""PSR/DSR + haircut Harvey-Liu (Etapa 3.2d).

PSR con skew/kurtosis/T reales (Bailey–López de Prado). DSR corrige por el nº
de trials con la cota de Sharpe esperado bajo H0. Veto: DSR(N_eff) < 0.20 → FAIL.
Se reporta con S bruto (conservador) y N_eff (optimista).
"""
from __future__ import annotations

import math
from statistics import NormalDist

_N = NormalDist()


def psr(observed_sr: float, benchmark_sr: float, T: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    if T <= 1 or not math.isfinite(observed_sr):
        return float("nan")
    sr0 = benchmark_sr
    var = (1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr**2) / max(T - 1, 1)
    if var <= 0:
        return float("nan")
    return float(_N.cdf((observed_sr - sr0) / math.sqrt(var)))


def expected_sr_null(sharpes: list[float], S: int) -> float:
    """Sharpe esperado del mejor bajo H0 (sin edge), dada la dispersión
    observada entre trials."""
    import numpy as np

    s = np.asarray([v for v in sharpes if v is not None and np.isfinite(v)], dtype=float)
    if len(s) < 2:
        return 0.0
    var = float(np.var(s, ddof=1))
    if var <= 0:
        return 0.0
    gamma = 0.5772156649
    return float(math.sqrt(var) * ((1 - gamma) * _N.inv_cdf(1 - 1 / S) + gamma * _N.inv_cdf(1 - 1 / (S * math.e))))


def dsr(observed_sr: float, sharpes_all: list[float], S: int, T: int,
        skew: float = 0.0, kurt: float = 3.0) -> dict:
    sr0 = expected_sr_null(sharpes_all, S)
    return {"dsr": psr(observed_sr, sr0, T, skew, kurt), "sr_null_expected": sr0, "S": S}


def harvey_liu_haircut(observed_sr: float, sharpes_all: list[float], S: int, T: int) -> float:
    """Haircut multiplicativo aprox: SR ajustado por nº de tests. Ordena la
    cola, no veta solo."""
    import numpy as np

    s = np.asarray([v for v in sharpes_all if v is not None and np.isfinite(v)], dtype=float)
    if len(s) < 2 or T <= 0 or not math.isfinite(observed_sr):
        return float("nan")
    rho = float(np.corrcoef(np.arange(len(s)), np.sort(s))[0, 1]) if len(s) > 2 else 0.0
    rho = 0.0 if not math.isfinite(rho) else abs(rho)
    # t-ratio observado vs esperado bajo selección
    t_obs = observed_sr * math.sqrt(T)
    t_hurdle = math.sqrt(2 * math.log(S))  # Bonferroni-flavoured
    adj_t = max(t_obs - t_hurdle * (1 - rho * 0.5), 0.0)
    return float(adj_t / math.sqrt(T))
