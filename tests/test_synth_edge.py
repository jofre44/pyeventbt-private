"""Tests Etapa 3: scoring, determinismo sintético, control nulo, FDR, DSR."""
import numpy as np
import pandas as pd

from pyeventbt.robustness.dsr import dsr, harvey_liu_haircut, psr
from pyeventbt.robustness.edge_tests import edge_test
from pyeventbt.robustness.fdr import assert_resolution, bh_fdr, resolution_check
from pyeventbt.robustness.null_control import null_pool_verdict, null_set_verdict
from pyeventbt.robustness.scoring import consistency_verdict, rank_verdict, score_percentile_rank
from pyeventbt.robustness.synthetic import (
    SyntheticMarketGenerator,
    block_permuted_closes,
    iid_null_closes,
    validate_synthetic,
)


def test_scoring_ranks_and_borders():
    synth = [0.5, 0.6, 0.7, 0.8, 0.9]
    assert score_percentile_rank(0.2, synth) == 0.0      # bajo => FAIL
    assert score_percentile_rank(2.0, synth) == 100.0    # alto => AMBAR
    assert score_percentile_rank(0.7, synth) == 60.0
    assert score_percentile_rank(float("nan"), synth) is None   # borde
    assert score_percentile_rank(0.5, [0.5] * 5) == 50.0        # varianza cero
    assert rank_verdict(5.0) == "FAIL"
    assert rank_verdict(95.0) == "AMBER"
    assert rank_verdict(50.0) == "GREEN"
    v = consistency_verdict({"sharpe_net": 5.0, "pf": 50.0, "n_trades": 60.0})
    assert v["verdict"] == "RED"  # un FAIL contamina el set
    v2 = consistency_verdict({"sharpe_net": 95.0, "pf": 50.0, "n_trades": 60.0})
    assert v2["verdict"] == "GREEN"  # ámbar-alto + quorum: lo resuelve edge+3B


def test_synthetic_determinism():
    rng = np.random.default_rng(0)
    close = 1.0 + np.cumsum(rng.normal(0, 0.001, 2000))
    high, low = close * 1.0005, close * 0.9995
    g1 = SyntheticMarketGenerator(seed=9).fit(close, high, low)
    g2 = SyntheticMarketGenerator(seed=9).fit(close, high, low)
    a = g1.simulate_close(500, seed=1001)
    b = g2.simulate_close(500, seed=1001)
    assert np.array_equal(a, b)  # misma seed => mismo universo
    c = g1.simulate_close(500, seed=1002)
    assert not np.array_equal(a, c)


def test_null_control():
    rng = np.random.default_rng(4)
    pool_ok = list(rng.normal(0, 0.2, 200))
    v = null_pool_verdict(pool_ok)
    assert v["verdict"] == "PASS" and not v["cost_drag"]
    pool_biased = list(rng.normal(0.5, 0.2, 200))  # backtester que gana en nulo
    assert null_pool_verdict(pool_biased)["verdict"] == "STOP"
    pool_drag = list(rng.normal(-0.5, 0.2, 200))  # pierde en nulo: costes, no sesgo
    v2 = null_pool_verdict(pool_drag)
    assert v2["verdict"] == "PASS" and v2["cost_drag"] is True
    assert null_set_verdict([0.8, 0.9, 0.7])["verdict"] == "FAIL"
    assert null_set_verdict([0.0, -0.1, 0.1])["verdict"] == "PASS"


def test_fdr_resolution():
    assert resolution_check(2000, 0.10, 40)["verdict"] == "PASS"   # 1/2001 < .0025
    assert resolution_check(50, 0.10, 250)["verdict"] == "ABORT"   # caso v1 decorativo
    try:
        assert_resolution(50, 0.10, 250)
        raise AssertionError("debió abortar")
    except RuntimeError:
        pass
    p = {"s1": 0.001, "s2": 0.02, "s3": 0.5}
    r = bh_fdr(p, q=0.10)
    assert r["rejected"] == ["s1", "s2"]  # .02 <= 2/3*.10
    assert r["adj_p"]["s1"] <= 0.10
    assert bh_fdr({"a": 0.5, "b": 0.6}, q=0.10)["rejected"] == []


def test_edge_spa_detects_planted_edge():
    rng = np.random.default_rng(21)
    idx = pd.date_range("2021-01-01", periods=300, freq="D")
    noise = {f"s{i}": rng.normal(0, 0.01, 300) for i in range(5)}
    noise["edge"] = rng.normal(0.004, 0.01, 300)  # Sharpe ~0.004/0.01*sqrt(252)≈6
    df = pd.DataFrame(noise, index=idx)
    res = edge_test(df, method="spa", n_resamples=300, seed=5)
    assert res["p_family"] is not None and res["p_family"] < 0.05
    assert res["p_marginal"]["edge"] < 0.05


def test_edge_spa_no_false_alarm_on_noise():
    rng = np.random.default_rng(22)
    idx = pd.date_range("2021-01-01", periods=300, freq="D")
    df = pd.DataFrame({f"s{i}": rng.normal(0, 0.01, 300) for i in range(5)}, index=idx)
    res = edge_test(df, method="spa", n_resamples=300, seed=6)
    assert res["p_family"] > 0.05


def test_dsr_veto_and_psr():
    rng = np.random.default_rng(33)
    sharpes = list(rng.normal(0, 0.5, 400))
    r = dsr(0.1, sharpes, S=400, T=756)  # SR 0.1 entre 400 intentos
    assert r["dsr"] < 0.20  # veto: FAIL
    assert 0.0 <= r["dsr"] <= 1.0
    assert psr(2.0, 0.0, T=756) > 0.95
    h = harvey_liu_haircut(1.5, sharpes, S=400, T=756)
    assert 0.0 <= h <= 1.5


def test_iid_and_block_generators():
    rng = np.random.default_rng(0)
    close = 1.0 + np.cumsum(rng.normal(0, 0.001, 1000))
    n1 = iid_null_closes(close, seed=1)
    n2 = iid_null_closes(close, seed=1)
    assert np.array_equal(n1, n2) and len(n1) == len(close)
    b = block_permuted_closes(close, block=10, seed=1)
    assert len(b) == len(close) and not np.array_equal(b, close)


def test_validate_synthetic_tol():
    real = {"vol": 1.0, "ac1_abs": 0.2, "kurtosis": 6.0, "mean_range": 0.001}
    good = {"vol": 1.1, "ac1_abs": 0.21, "kurtosis": 5.5, "mean_range": 0.0011}
    bad = dict(good, vol=2.0)
    assert validate_synthetic(real, good)["overall"] is True
    rep = validate_synthetic(real, bad)
    assert rep["overall"] is False and rep["vol"]["pass"] is False
