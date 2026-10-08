"""Tests Etapa 5: asignación, exposición, estrés, envolvente MC, capacity."""
import numpy as np
import pandas as pd

from pyeventbt.portfolio.allocation import allocate, ledoit_wolf_cov
from pyeventbt.portfolio.capacity import capacity_curve, marginal_benefit
from pyeventbt.portfolio.exposure import currency_exposure
from pyeventbt.portfolio.mc_envelope import mc_dd_envelope
from pyeventbt.portfolio.stress_corr import stress_correlation


def _rets(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=500, freq="D")
    return pd.DataFrame({
        "bb_AUDCAD_H1": rng.normal(0.0008, 0.006, 500),
        "bb_AUDCHF_H1": rng.normal(0.0006, 0.007, 500),
        "m_AUDCAD_H4": rng.normal(0.0007, 0.008, 500),
        "null_X": rng.normal(0.0, 0.006, 500),
    }, index=idx)


def test_allocation_hrp_weights():
    res = allocate(_rets(), {"method": "hrp", "covariance": "ledoit_wolf",
                             "vol_target_annual": 0.10, "kelly_fraction_cap": 0.25,
                             "max_weight_per_strategy": 0.30, "rebalance": "monthly"})
    w = res["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v <= 0.30 + 1e-9 for v in w.values())
    assert res["leverage"] > 0


def test_ledoit_wolf_shrinks():
    cov = ledoit_wolf_cov(_rets())
    assert cov.shape == (4, 4)
    assert abs(cov.iloc[0, 0]) > 0


def test_exposure_netting():
    pos = pd.DataFrame([
        {"symbol": "EURUSD", "direction": 1, "lots": 2.0},
        {"symbol": "GBPUSD", "direction": 1, "lots": 1.0},
    ])
    r = currency_exposure(pos, 5.0)
    assert r["net_by_currency"]["USD"] == -3.0  # misma apuesta corto USD
    assert r["verdict"] == "PASS"
    r2 = currency_exposure(pos, 2.0)
    assert r2["verdict"] == "FAIL" and "USD" in r2["breach"]


def test_stress_corr_flags_crash_correlation():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2021-01-01", periods=500, freq="D")
    a = rng.normal(0, 0.01, 500)
    df = pd.DataFrame({"A": a, "B": a + rng.normal(0, 1e-4, 500)}, index=idx)
    r = stress_correlation(df, max_rho_stress=0.80)
    assert r["stress_rho"] > 0.9 and r["verdict"] == "FAIL"


def test_mc_envelope_monotonic():
    rng = np.random.default_rng(2)
    daily = pd.Series(rng.normal(0.0005, 0.01, 500))
    env = mc_dd_envelope(daily, n_paths=300, seed=3)
    assert env["dd_envelope"]["p99"] >= env["dd_envelope"]["p95"] >= 0
    assert 0 <= env["p_annual_loss"] <= 1


def test_capacity_decay_monotonic_and_admission():
    rng = np.random.default_rng(3)
    daily = pd.Series(rng.normal(0.001, 0.01, 500))
    cur = capacity_curve(daily, 200.0)
    decays = [r["decay_pct"] for r in cur["curve"]]
    assert decays == sorted(decays)  # más capital => más decay
    assert marginal_benefit({"sharpe": 1.0, "maxdd": 10}, {"sharpe": 1.2, "maxdd": 8})["verdict"] == "ADMIT"
    assert marginal_benefit({"sharpe": 1.0, "maxdd": 10}, {"sharpe": 1.2, "maxdd": 12})["verdict"] == "REJECT"
