"""Tests multi-símbolo/multi-TF: mapa de TFs, spread por símbolo, rama D1."""
import pytest

from pyeventbt.optimization.adapter import resolve_base_spread, tf_to_strategy_timeframe
from pyeventbt.strategy.core.strategy_timeframes import StrategyTimeframes


def test_tf_map_covers_pipeline_timeframes():
    assert tf_to_strategy_timeframe("M30") == StrategyTimeframes.THIRTY_MIN
    assert tf_to_strategy_timeframe("H1") == StrategyTimeframes.ONE_HOUR
    assert tf_to_strategy_timeframe("H4") == StrategyTimeframes.FOUR_HOUR
    assert tf_to_strategy_timeframe("D1") == StrategyTimeframes.ONE_DAY
    assert tf_to_strategy_timeframe("M1") == StrategyTimeframes.ONE_MIN


def test_tf_map_unknown_fails_fast():
    with pytest.raises(ValueError):
        tf_to_strategy_timeframe("W1")  # no soportado: falla aquí, no en H1 silencioso


def test_resolve_base_spread_per_symbol():
    master = {"universe": {"base_spread_points": {"AUDCAD": 28.0, "AUDCHF": 24.0}},
              "costs": {"base_spread_points": 20.0}}
    assert resolve_base_spread(master, "AUDCAD") == 28.0
    assert resolve_base_spread(master, "AUDCHF") == 24.0
    assert resolve_base_spread({"universe": {}, "costs": {}}, "XXX") == 20.0


def test_d1_factory_uses_daily_branch():
    from strategies.bbands_breakout.signal import make_signal_fn

    fn = make_signal_fn({"bb_period": 20, "bb_std_dev": 2.5},
                        signal_timeframe=StrategyTimeframes.ONE_DAY)
    assert callable(fn) and fn.strategy_family == "bbands_breakout"
    fn_h1 = make_signal_fn({"bb_period": 20, "bb_std_dev": 2.5},
                           signal_timeframe=StrategyTimeframes.ONE_HOUR)
    assert callable(fn_h1)


def test_master_universe_multipair_multitf():
    from pyeventbt.runner.master_config import load_master

    cfg = load_master("pipeline_master.yaml")
    assert len(cfg["universe"]["symbols"]) == 26
    assert set(cfg["universe"]["timeframes"]) == {"M30", "H1", "H4", "D1"}
    assert set(cfg["universe"]["symbols"]) >= {"AUDCAD", "AUDCHF", "EURUSD", "GBPUSD", "USDJPY"}
    spreads = cfg["universe"]["base_spread_points"]
    assert set(spreads) == set(cfg["universe"]["symbols"])  # cada par con su spread
    assert spreads["AUDCHF"] == 24.0 and spreads["EURUSD"] == 6.0
    for s in ("AUDUSD", "USDCAD", "USDCHF"):
        assert s in cfg["universe"]["aux_fallback_rates"]
