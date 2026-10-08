"""Tests Etapa 4: decorrelación unilateral, orden puertas→lockbox, integridad lockbox."""
import numpy as np
import pandas as pd
import pytest

from pyeventbt.robustness.decorrelation import composite_score, greedy_select, jitter_grid, jitter_verdict
from pyeventbt.selection.gates import apply_gates
from pyeventbt.selection.lockbox import guard_gates_before_lockbox, holm, lockbox_evaluate


def _rets(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=200, freq="D")
    base = rng.normal(0.001, 0.01, 200)
    return pd.DataFrame({
        "A": base,                                          # referencia
        "B": base + rng.normal(0, 1e-4, 200),                # rho≈1 con A
        "C": -base + rng.normal(0, 1e-4, 200),               # rho≈-1 (diversifica)
        "D": rng.normal(0.001, 0.01, 200),                   # independiente
    }, index=idx)


def _stats():
    return pd.DataFrame([
        {"set_id": "A", "n_trades": 200, "sharpe_net": 1.0, "pf": 1.5, "r2": 0.6, "turnover": 100.0},
        {"set_id": "B", "n_trades": 200, "sharpe_net": 0.9, "pf": 1.4, "r2": 0.5, "turnover": 110.0},
        {"set_id": "C", "n_trades": 200, "sharpe_net": 0.8, "pf": 1.3, "r2": 0.4, "turnover": 90.0},
        {"set_id": "D", "n_trades": 200, "sharpe_net": 0.7, "pf": 1.2, "r2": 0.3, "turnover": 120.0},
    ])


def test_decorr_unilateral():
    cfg = {"correlation_threshold": 0.7, "max_sets": 5, "min_trades": 30,
           "corr_method": "pearson", "weights": {}, "pf_cap_percentile": 95.0}
    res = greedy_select(_stats(), _rets(), cfg)
    assert "A" in res["selected"]
    assert "B" not in res["selected"]   # rho≈+1 bloqueado
    assert "C" in res["selected"]       # rho≈−1 se queda (diversificador)
    assert "D" in res["selected"]
    assert res["n_effective"] > 0


def test_decorr_prefilter_and_score():
    df = _stats()
    df.loc[3, "n_trades"] = 5  # D fuera por prefiltro
    cfg = {"correlation_threshold": 0.7, "max_sets": 5, "min_trades": 30,
           "corr_method": "spearman", "weights": {}, "pf_cap_percentile": 95.0}
    res = greedy_select(df, _rets(), cfg)
    assert "D" not in res["selected"]
    s = composite_score(_stats(), {})
    assert s.idxmax() == 0  # A domina


def test_jitter_grid_and_verdict():
    nbs = jitter_grid({"bb_period": 20, "bb_std_dev": 2.4}, pct=10.0)
    assert len(nbs) == 6  # 2 por param + 2 esquinas
    assert any(n["bb_period"] == 22 for n in nbs) and any(n["bb_period"] == 18 for n in nbs)
    ok = jitter_verdict([{"sharpe_net": 0.5, "pf": 1.2}] * 5)
    assert ok["verdict"] == "PASS" and ok["green_share"] == 1.0
    bad = jitter_verdict([{"sharpe_net": 0.5, "pf": 1.2}] * 3 + [{"sharpe_net": -1.0, "pf": 0.5}] * 2)
    assert bad["verdict"] == "FAIL"


def _pnl(trend=0.0005, n=400):
    rng = np.random.default_rng(0)
    eq = 10000 * np.cumprod(1 + rng.normal(trend, 0.002, n))
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame({"EQUITY": eq, "BALANCE": 10000}, index=idx)


def test_gates_pass_and_fail():
    pnl = _pnl(trend=0.001)
    daily = pnl["EQUITY"].pct_change().dropna()
    summary = {"pf": 1.8, "r2": 0.7, "sharpe_net": 1.5}
    cfg = {"pf_min": 1.3, "pf_retest_band": [1.1, 1.3], "r2_min": 0.5,
           "max_dd_days": 365, "max_red_months": 10, "regimes": {}}
    assert apply_gates(summary, pnl, daily, cfg)["verdict"] == "PASS"
    bad = apply_gates({"pf": 0.9, "r2": 0.1, "sharpe_net": -1.0}, _pnl(trend=-0.001),
                      _pnl(trend=-0.001)["EQUITY"].pct_change().dropna(), cfg)["verdict"] == "FAIL"
    assert bad


def test_regime_dependence_detected():
    rng = np.random.default_rng(9)
    # Año 1 gana, año 2 pierde: dependencia de régimen aunque el total sea +
    r1 = rng.normal(0.003, 0.01, 252)
    r2 = rng.normal(-0.001, 0.01, 252)
    idx = pd.date_range("2021-01-01", periods=504, freq="D")
    from pyeventbt.analytics.regimes import regime_report

    rep = regime_report(pd.Series(np.concatenate([r1, r2]), index=idx))
    assert rep["verdict"] == "FAIL" and "year" in rep["dependent_families"]


def test_gate_order_lockbox_requires_gates_pass():
    guard_gates_before_lockbox("PASS")  # no revienta
    with pytest.raises(RuntimeError):
        guard_gates_before_lockbox("FAIL")  # puertas después del lockbox = inválido


def test_lots_sl_tp_promote():
    from pyeventbt.selection.lots import compute_lots
    from pyeventbt.selection.sl_tp import compute_sl_tp
    from pyeventbt.selection.promote import promote_row, write_promoted

    net = pd.Series([10.0, -20.0, 15.0, -5.0, 8.0])
    lots = compute_lots(net, {"max_dd_allow": 1200, "max_trade_lost_allow": 200,
                              "mean_trade_lost_allow": 100})
    assert 0.01 <= lots["lots"] <= 1.0
    idx = pd.date_range("2021-01-01", periods=48, freq="h")
    h1 = pd.DataFrame({"high": 1.0, "low": 0.99, "close": 0.995}, index=idx)
    closed = pd.DataFrame([{
        "OPEN_TIME": idx[0].tz_localize("UTC"), "CLOSE_TIME": idx[10].tz_localize("UTC"),
        "ENTRY_PRICE": 0.995, "DIRECTION": "BUY"}])
    st = compute_sl_tp(closed, h1, {"p_sl": 10, "p_tp": 90})
    assert st["method"].startswith("percentil") and st["sl_pct"] >= 0 and st["tp_pct"] >= 0
    row = promote_row({"set_id": "s1", "params": {"bb_period": 20}, "lots": lots["lots"]})
    assert row["set_id"] == "s1" and row["lots"] == lots["lots"]
    import tempfile, os

    p = write_promoted([{"set_id": "s1", "params": {"a": 1}}], os.path.join(tempfile.mkdtemp(), "prom.csv"))
    assert pd.read_csv(p).iloc[0]["set_id"] == "s1"


def test_lockbox_integrity_and_holm():
    assert holm([0.001, 0.02, 0.5], alpha=0.05) == [True, True, False]
    ok = lockbox_evaluate({"pf": 1.2, "sharpe_net": 0.3}, 0.001, [0.4, 0.5],
                           {"min_pf": 1.1})
    assert ok["verdict"] == "PASS" and ok["holm_reject"] is True
    no = lockbox_evaluate({"pf": 1.0, "sharpe_net": 0.3}, 0.001, [0.4, 0.5],
                           {"min_pf": 1.1})
    assert no["verdict"] == "FAIL"  # filtro operativo pf≥1.1
    no2 = lockbox_evaluate({"pf": 1.5, "sharpe_net": 0.3}, 0.6, [0.01, 0.02],
                            {"min_pf": 1.1})
    assert no2["verdict"] == "FAIL"  # Holm no rechaza
