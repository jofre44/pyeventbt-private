"""Tests Etapa 0: embargo en tiempo + firewall de ventanas + lockbox intacto."""
from datetime import datetime

from pyeventbt.runner.data_manifest import (
    apply_embargo,
    check_lockbox_untouched,
    check_window_purity,
    compute_embargo,
)


def test_embargo_time_scales_with_tf():
    # bbands H1: max_holding 24 barras (1 día time-stop), warmup 30 (bb_period max), +10
    e_h1 = compute_embargo("H1", max_holding_bars=24, warmup_bars=30, extra_bars=10)
    assert e_h1["bars"] == 40
    assert e_h1["delta"].total_seconds() == 40 * 3600
    # Mismo holding en H4 debe dar 4x tiempo (embargo EN TIEMPO, no en barras)
    e_h4 = compute_embargo("H4", max_holding_bars=24, warmup_bars=30, extra_bars=10)
    assert e_h4["bars"] == 40
    assert e_h4["delta"].total_seconds() == 40 * 4 * 3600


def test_embargo_is_max_holding_warmup_plus_extra():
    e = compute_embargo("H1", max_holding_bars=96, warmup_bars=20, extra_bars=10)
    assert e["bars"] == 106  # manda max_holding


def test_window_purity_fail_without_embargo():
    is_end = datetime(2023, 12, 31)
    res = check_window_purity(is_end, datetime(2024, 1, 1), "H1",
                              max_holding_bars=24, warmup_bars=30, extra_bars=10)
    assert res["verdict"] == "FAIL"  # 1 día < 40h de embargo


def test_window_purity_pass_with_embargo():
    is_end = datetime(2023, 12, 31)
    fwd = apply_embargo(is_end, "H1", max_holding_bars=24, warmup_bars=30, extra_bars=10)
    res = check_window_purity(is_end, fwd, "H1",
                              max_holding_bars=24, warmup_bars=30, extra_bars=10)
    assert res["verdict"] == "PASS"


def test_lockbox_untouched_when_no_registry(tmp_path):
    res = check_lockbox_untouched(tmp_path / "lockbox_uses.yaml", "bbands_breakout", "AUDCAD", "H1")
    assert res["verdict"] == "PASS"
