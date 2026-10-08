"""Tests Etapa 6: risk limits, reconcile, idempotencia, paridad, monitor, rollout."""
import pandas as pd

from pyeventbt.live.monitor import cusum_alerts, health_score, ks_drift, retirement_check
from pyeventbt.live.parity import parity_verdict, signal_mismatch, slippage_report, tracking_error
from pyeventbt.live.reconcile import OrderBook, reconcile
from pyeventbt.live.risk_limits import KillSwitch, evaluate_limits
from pyeventbt.live.rollout import rollout_plan, stage_gate

LIMITS = {"per_trade_risk_pct": 1.0, "daily_loss_pct": 3.0, "strategy_dd_pct": 10.0,
          "portfolio_dd_pct": 15.0, "max_spread_points": 30, "heartbeat_seconds": 30,
          "stale_data_seconds": 120}


def test_live_risk_limits():
    kill = KillSwitch()
    ok = {"trade_risk_pct": 0.5, "daily_loss_pct": 1.0, "strategy_dd_pct": 3.0,
          "portfolio_dd_pct": 5.0, "spread_points": 12, "data_stale_s": 5}
    assert evaluate_limits(ok, LIMITS, kill)["action"] == "ok"
    assert evaluate_limits({**ok, "daily_loss_pct": 5.0}, LIMITS, kill)["action"] == "halt"
    assert evaluate_limits({**ok, "portfolio_dd_pct": 20.0}, LIMITS, kill)["action"] == "flatten"
    assert evaluate_limits({**ok, "spread_points": 50}, LIMITS, kill)["action"] == "halt"
    assert evaluate_limits({**ok, "data_stale_s": 500}, LIMITS, kill)["action"] == "halt"
    kill.engage("manual")
    assert evaluate_limits(ok, LIMITS, kill)["action"] == "flatten"
    kill.reset()
    assert evaluate_limits(ok, LIMITS, kill)["action"] == "ok"


def test_live_reconcile():
    good = reconcile({"positions": 1, "orders": 0, "equity": 10000.0},
                     {"positions": 1, "orders": 0, "equity": 10000.0})
    assert good["action"] == "ok"
    bad = reconcile({"positions": 1, "orders": 0, "equity": 10000.0},
                    {"positions": 2, "orders": 0, "equity": 9990.0})
    assert bad["action"] == "halt_and_alert" and "positions" in bad["mismatches"]


def test_order_idempotency(tmp_path):
    book = OrderBook(tmp_path / "orders.json")
    calls = []
    r1 = book.send_once(11313, "bbands-0001", lambda: (calls.append(1), "T1")[1])
    r2 = book.send_once(11313, "bbands-0001", lambda: (calls.append(1), "T2")[1])
    assert r1["status"] == "sent" and r2["status"] == "duplicate_suppressed"
    assert r2["ticket"] == "T1" and len(calls) == 1  # un solo envío real
    # Persistencia ante reinicio
    book2 = OrderBook(tmp_path / "orders.json")
    assert book2.already_sent(11313, "bbands-0001")


def test_parity_metrics():
    live = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
                         "symbol": "AUDCAD", "side": "BUY"})
    bt = live.copy()
    m = signal_mismatch(live, bt)
    assert m["mismatch_pct"] == 0.0
    bt2 = bt.iloc[:8]
    assert signal_mismatch(live, bt2)["mismatch_pct"] > 0
    fills = pd.DataFrame({"expected": [1.0] * 20, "fill": [1.0 + i * 0.00001 for i in range(20)]})
    assert slippage_report(fills, 2.0)["ok"] is True
    te = tracking_error(pd.Series([10000.0, 10010.0, 10005.0]),
                        pd.Series([10000.0, 10010.0, 10005.0]))
    assert te["tracking_error_pct"] == 0.0
    v = parity_verdict(m, {"ok": True}, te,
                       {"max_signal_mismatch_pct": 2.0, "max_daily_tracking_error_pct": 0.5})
    assert v["verdict"] == "PASS"


def test_monitor_and_rollout():
    import numpy as np

    rng = np.random.default_rng(0)
    assert cusum_alerts(pd.Series([0.0] * 100))["verdict"] == "OK"
    shift = pd.Series([0.0] * 50 + [3.0] * 50)  # salto de nivel: debe alarmar
    assert cusum_alerts(shift)["alarms"] >= 1
    assert health_score({"performance": 1, "execution": 1, "statistical": 1, "regime": 1})["level"] == "GREEN"
    assert health_score({"performance": 0, "execution": 0, "statistical": 0, "regime": 0})["level"] == "RED"
    assert ks_drift(pd.Series(rng.normal(0, 1, 200)), pd.Series(rng.normal(0, 1, 200)))["verdict"] == "OK"
    ret = retirement_check({"dd_vs_p99": True, "cusum_alarms": 0, "sharpe_below_p5_days": 0}, {})
    assert ret["retire"] is True
    stages = [{"name": "paper", "lot_mult": 0.0, "min_days": 60, "min_trades": 30},
              {"name": "micro", "lot_mult": 0.1, "min_days": 60}]
    perf = {"days": 61, "trades": 35, "parity_ok": True, "sharpe_in_envelope": True,
            "dd_in_envelope": True, "incidents": 0}
    assert stage_gate(stages[0], perf)["advance"] is True
    assert stage_gate(stages[0], {**perf, "incidents": 1})["advance"] is False
    assert rollout_plan(stages, "paper")["next"] == "micro"
