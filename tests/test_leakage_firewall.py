"""Tests Etapa 0: leakage firewall (FAIL bloqueante, no warnings)."""
from pyeventbt.runner.data_manifest import data_check


def _manifest(**kw):
    m = {"symbol": "AUDCAD", "rows": 1000, "n_duplicates": 0, "n_null_ohlc": 0,
         "n_gaps_gt_1min": 50, "spread_mean": 20.0}
    m.update(kw)
    return m


def test_leakage_duplicates_fail():
    assert data_check(_manifest(n_duplicates=3))["verdict"] == "FAIL"


def test_leakage_null_ohlc_fail():
    assert data_check(_manifest(n_null_ohlc=7))["verdict"] == "FAIL"


def test_leakage_empty_fail():
    assert data_check(_manifest(rows=0))["verdict"] == "FAIL"


def test_clean_manifest_pass_with_weekend_gap_warn():
    res = data_check(_manifest())
    assert res["verdict"] == "PASS"
    assert res["warns"]  # gaps de fin de semana documentados, no bloquean
