"""Test Etapa 1: regresión determinista punta a punta con el motor real.

Ventana de 1 mes + params nominales: dos backtests idénticos deben dar el
mismo Sharpe y nº de trades (deterministic_mode). Lento (~10s): es el precio
de testear contra el motor de verdad y no contra un mock.
"""
from datetime import datetime

from pyeventbt.optimization.adapter import run_pyeventbt_once


def _kw():
    return {
        "costs": {"spread_mult": 1.0},
        "window": (datetime(2021, 1, 3), datetime(2021, 2, 3)),
        "cash": 10000,
        "factory_path": "strategies.bbands_breakout.signal:make_signal_fn",
        "params_class_path": "strategies.bbands_breakout.params:BBandsParams",
        "data_dir": "E:/Data_csv/MT5",
        "strategy_id": "11313",
        "base_spread_points": 28.0,
        "aux_fallback_rates": {"AUDUSD": 0.66, "USDCAD": 1.36},
    }


def test_regression_run():
    combo = {"strategy_family": "bbands_breakout", "symbol": "AUDCAD", "timeframe": "H1"}
    params = {"bb_period": 20, "bb_std_dev": 2.5}
    a = run_pyeventbt_once(params, combo, seed=1500, **_kw())
    b = run_pyeventbt_once(params, combo, seed=1500, **_kw())
    assert a["summary"]["n_trades"] > 0  # el nominal opera en este símbolo/TF
    assert a["summary"]["sharpe_net"] == b["summary"]["sharpe_net"]
    assert a["summary"]["n_trades"] == b["summary"]["n_trades"]
    assert set(a["summary"]) >= {"sharpe_net", "pf", "n_trades", "maxdd_pct", "exposure_pct", "r2"}
