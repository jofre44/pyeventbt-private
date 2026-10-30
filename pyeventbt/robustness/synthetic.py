"""Generador sintético GJR-GARCH(1,1)-t + nulo iid + stationary bootstrap (Etapa 3).

- Ajuste en H1 real: MLE GJR-GARCH-t (scipy, @njit) sobre muestra ≤30k,
  nu por curtosis, wicks reales bootstrap, spread honesto.
- Simulación a nivel del TF señal (el conector M1→H1 hace round-trip exacto
  con barras H1; documentado en ASSUMPTIONS F1).
- Seeds (master.seed, cluster_id, sim_idx); pickle-safe para joblib.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger("pyeventbt.pipeline")


@np.errstate(all="ignore")
def _gjr_loglik_py(theta, r):
    from math import lgamma, log, pi, sqrt

    omega, alpha, gamma, beta, nu = theta
    n = len(r)
    var = float(np.mean(r * r))
    ll = 0.0
    for i in range(n):
        prev = r[i - 1] if i else 0.0
        var = omega + (alpha + gamma * (1.0 if prev < 0 else 0.0)) * prev * prev + beta * var if i else var
        if var <= 1e-12:
            var = 1e-12
        x = r[i] / sqrt(var)
        s2 = (nu - 2.0) / nu
        ll += (-0.5 * log(var) - 0.5 * log(s2)
               + lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0)
               - 0.5 * log(pi * nu)
               - (nu + 1.0) / 2.0 * log(1.0 + x * x / (nu * s2)))
    return ll


try:
    from numba import njit

    @njit
    def _gjr_sim(innov: np.ndarray, omega: float, alpha: float, gamma: float,
                 beta: float, init_var: float) -> np.ndarray:
        n = innov.shape[0]
        out = np.empty(n)
        var = init_var
        prev = 0.0
        for i in range(n):
            var = omega + (alpha + gamma * (1.0 if prev < 0.0 else 0.0)) * prev * prev + beta * var
            if var <= 1e-12:
                var = 1e-12
            shock = innov[i] * np.sqrt(var)
            out[i] = shock
            prev = shock
        return out

    _NJIT_OK = True
except Exception:  # pragma: no cover
    _NJIT_OK = False

    def _gjr_sim(innov, omega, alpha, gamma, beta, init_var):
        out = np.empty_like(innov)
        var, prev = init_var, 0.0
        for i in range(len(innov)):
            var = omega + (alpha + gamma * (1.0 if prev < 0 else 0.0)) * prev * prev + beta * var
            out[i] = innov[i] * np.sqrt(max(var, 1e-12))
            prev = out[i]
        return out


@dataclass
class GJRParams:
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float
    resid_std: float  # std de r (retornos ×100) para re-escalar
    anchor_log_close: float = 0.0


@dataclass
class SyntheticMarketGenerator:
    seed: int = 1500
    params: GJRParams | None = None
    wick_up: np.ndarray = field(default_factory=lambda: np.zeros(1))
    wick_down: np.ndarray = field(default_factory=lambda: np.zeros(1))
    spread_mean_real: float = 20.0
    spreads_real: np.ndarray = field(default_factory=lambda: np.zeros(1))
    stats_real: dict = field(default_factory=dict)
    stats_range_h1: dict = field(default_factory=dict)

    # ---------------- ajuste ----------------
    def fit(self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
            spread_mean: float = 20.0, spreads: np.ndarray | None = None) -> "SyntheticMarketGenerator":
        from scipy.optimize import minimize

        close = np.asarray(close, dtype=float)
        r = np.diff(np.log(close)) * 100.0
        r = r[np.isfinite(r)]
        if len(r) > 30000:
            rng = np.random.default_rng(self.seed)
            r = rng.choice(r, 30000, replace=False)
        std = float(np.std(r)) or 1.0
        z = r / std
        self.spread_mean_real = float(spread_mean)
        self.spreads_real = np.asarray(spreads if spreads is not None else [spread_mean], dtype=float)
        self.stats_real = _describe_returns(r, close, high, low)

        def nll(theta):
            omega, alpha, gamma, beta, nu = theta
            if alpha + gamma / 2 + beta >= 0.999:
                return 1e12
            try:
                return -_gjr_loglik_py(theta, z)
            except Exception:
                return 1e12

        x0 = np.array([0.05, 0.07, 0.05, 0.88, 7.0])
        bounds = [(1e-4, 1.0), (0.005, 0.30), (0.0, 0.30), (0.50, 0.97), (4.5, 12.0)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200})
        th = res.x if res.success else x0
        self.params = GJRParams(omega=float(th[0]) * std * std, alpha=float(th[1]),
                                gamma=float(th[2]), beta=float(th[3]), nu=float(th[4]),
                                resid_std=std, anchor_log_close=float(np.log(close[-1])))
        # Wicks como fracción del precio (no del move): preserva la distribución
        # de rangos al simular. Antes: ratio vs |C-O| explotaba en dojis.
        o, c = close[:-1], close[1:]
        px = np.maximum(close[1:], 1e-9)
        self.wick_up = np.clip((high[1:] - np.maximum(o, c)) / px, 0, 0.05)
        self.wick_down = np.clip((np.minimum(o, c) - low[1:]) / px, 0, 0.05)
        self.wick_up = self.wick_up[np.isfinite(self.wick_up)]
        self.wick_down = self.wick_down[np.isfinite(self.wick_down)]
        logger.warning(f"GJR ajustado: a={th[1]:.3f} g={th[2]:.3f} b={th[3]:.3f} nu={th[4]:.2f}")
        return self

    # ---------------- simulación ----------------
    def simulate_close(self, n_bars: int, seed: int) -> np.ndarray:
        assert self.params is not None, "fit() primero"
        p = self.params
        rng = np.random.default_rng(seed)
        innov = rng.standard_t(p.nu, size=n_bars + 50) * np.sqrt((p.nu - 2.0) / p.nu)
        sim_r = _gjr_sim(innov, p.omega, p.alpha, p.gamma, p.beta, p.omega / max(1e-6, 1 - p.alpha - p.gamma / 2 - p.beta))
        sim_r = sim_r[50:] / 100.0  # quemar transitorio, volver a escala log
        return sim_r

    def build_ohlc(self, sim_r: np.ndarray, start_close: float, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = len(sim_r)
        close = np.empty(n)
        close[0] = start_close * float(np.exp(sim_r[0]))
        for i in range(1, n):
            close[i] = close[i - 1] * float(np.exp(sim_r[i]))
        open_ = np.empty(n)
        open_[0] = start_close
        open_[1:] = close[:-1]
        iu = rng.integers(0, len(self.wick_up), n)
        idn = rng.integers(0, len(self.wick_down), n)
        ref = np.maximum(close, 1e-9)
        high = np.maximum(open_, close) + self.wick_up[iu] * ref
        low = np.minimum(open_, close) - self.wick_down[idn] * ref
        return {"open": open_, "high": high, "low": low, "close": close}

    def generate_bars(self, n_bars: int, start_dt, freq_min: int, seed: int,
                      start_close: float, spread_mult: float = 1.0) -> pl.DataFrame:
        import pandas as pd

        rng = np.random.default_rng(seed + 7)
        sim_r = self.simulate_close(n_bars, seed)
        ohlc = self.build_ohlc(sim_r, start_close, seed + 999)
        idx = pd.date_range(start_dt, periods=n_bars, freq=f"{freq_min}min")
        n_sp = len(self.spreads_real)
        spread = self.spreads_real[rng.integers(0, n_sp, n_bars)] * (1.0 + (spread_mult - 1.0))
        # datetime64 array directly (no `.to_pydatetime().tolist()`: that
        # built a ~1M-object Python list per sim → OOM next to the backtest).
        return pl.DataFrame({
            "datetime": idx.to_numpy(),
            "open": ohlc["open"], "high": ohlc["high"],
            "low": ohlc["low"], "close": ohlc["close"],
            "tickvol": np.ones(n_bars, dtype=int),
            "volume": np.ones(n_bars, dtype=int),
            "spread": spread,
        })

    def cache_key(self, symbol: str, tf: str) -> str:
        raw = f"{symbol}|{tf}|{self.params}".encode()
        return hashlib.sha256(raw).hexdigest()[:12]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


def _describe_returns(r_pct: np.ndarray, close, high, low) -> dict:
    a = np.abs(r_pct)
    ac1 = float(np.corrcoef(a[:-1], a[1:])[0, 1]) if len(a) > 10 else 0.0
    kurt = float(((r_pct - r_pct.mean()) ** 4).mean() / (r_pct.std() ** 4)) if r_pct.std() > 0 else 3.0
    import pandas as pd

    dr = pd.Series((np.asarray(high) - np.asarray(low)) / np.maximum(np.asarray(close), 1e-9))
    return {"vol": float(np.std(r_pct)), "ac1_abs": float(ac1) if np.isfinite(ac1) else 0.0,
            "kurtosis": float(kurt), "mean_range": float(dr.mean())}


def validate_synthetic(stats_real: dict, stats_synth: dict, tol: float = 0.20) -> dict:
    # Suelo absoluto para métricas cercanas a cero (ej. ac1_abs): la
    # desviación relativa explota sobre bases ~0.05 sin significado económico.
    floors = {"ac1_abs": 0.05, "vol": 0.0, "kurtosis": 0.0, "mean_range": 0.0}
    checks = {}
    for k in ("vol", "ac1_abs", "kurtosis", "mean_range"):
        ref, got = stats_real.get(k, 0.0), stats_synth.get(k, 0.0)
        denom = max(abs(ref), floors.get(k, 0.0), 1e-9)
        dev = abs(got - ref) / denom
        checks[k] = {"real": ref, "synth": got, "rel_dev": dev, "pass": bool(dev <= tol)}
    checks["overall"] = all(v["pass"] for v in checks.values() if isinstance(v, dict))
    return checks


def iid_null_closes(real_close: np.ndarray, seed: int) -> np.ndarray:
    """Generador nulo: retornos remuestreados iid (sin clustering de vol ni
    estructura serial). Punto de partida = primer close real."""
    rng = np.random.default_rng(seed)
    r = np.diff(np.log(np.asarray(real_close, dtype=float)))
    r = r[np.isfinite(r)]
    sim = rng.choice(r, size=len(r), replace=True)
    out = np.empty(len(real_close))
    out[0] = real_close[0]
    out[1:] = real_close[0] * np.exp(np.cumsum(sim))
    return out


def block_permuted_closes(real_close: np.ndarray, block: int, seed: int) -> np.ndarray:
    """Señal destruida por permutación de bloques (calibración, Etapa M)."""
    rng = np.random.default_rng(seed)
    r = np.diff(np.log(np.asarray(real_close, dtype=float)))
    r = r[np.isfinite(r)]
    idx = np.arange(len(r))
    blocks = [idx[i:i + block] for i in range(0, len(idx), block)]
    rng.shuffle(blocks)
    sim = np.concatenate(blocks)[:len(r)]
    out = np.empty(len(real_close))
    out[0] = real_close[0]
    out[1:] = real_close[0] * np.exp(np.cumsum(r[sim]))
    return out


def stationary_bootstrap_matrix(X: np.ndarray, p: float, seed: int, n_resamples: int) -> np.ndarray:
    """Índices stationary bootstrap (Politis-Romano): longitudes ~Geom(p).
    Devuelve array (n_resamples, n) de índices de remuestreo por filas."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    out = np.empty((n_resamples, n), dtype=np.int64)
    for b in range(n_resamples):
        idx = np.empty(n, dtype=np.int64)
        pos = int(rng.integers(0, n))
        for i in range(n):
            idx[i] = pos
            pos = int(rng.integers(0, n)) if rng.random() < p else (pos + 1) % n
        out[b] = idx
    return out
