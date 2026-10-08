"""Tests Etapa 1: params_filter estricto + constraints del sampler."""
import math

import optuna
import pytest

from strategies.bbands_breakout.params import BBandsParams, filter_params
from pyeventbt.optimization.objective import make_objective, robust_score, sample_search_space


def test_params_filter_valid():
    p = filter_params({"bb_period": 20, "bb_std_dev": 2.5})
    assert isinstance(p, BBandsParams) and p.bb_period == 20


def test_params_filter_unknown_key_fails():
    with pytest.raises(ValueError):
        filter_params({"bb_period": 20, "spread": 15})  # microestructura nunca en espacio


def test_params_filter_nan_fails():
    with pytest.raises(ValueError):
        filter_params({"bb_period": 20, "bb_std_dev": math.nan})


def test_params_filter_out_of_range_fails():
    with pytest.raises(ValueError):
        filter_params({"bb_period": 500, "bb_std_dev": 2.5})


def _master():
    return {
        "seed": 7,
        "strategy": {
            "name": "bbands_breakout", "cash": 10000,
            "factory": "x", "params_class": "y", "strategy_id": "1",
            "search_space": {"bb_period": {"type": "int", "low": 10, "high": 12, "step": 2}},
        },
        "universe": {"data_dir": "nodata"},
        "costs": {"spread_mult": 1.0},
        "optimization": {
            "constraints": {"min_trades": 100, "max_maxdd_pct": 25.0, "min_pf": 1.0, "max_exposure_pct": 80.0},
            "robust": {"enabled": True, "n_subwindows": 4, "lambda_std": 0.5},
            "early_gate": {"enabled": False},
        },
        "cluster": {"max_clusters_kept": 5},
    }


def _fake_run(summary, daily):
    def _fn(params, combo, **kw):
        return {"summary": summary, "daily": daily, "params_hash": "h"}
    return _fn


def test_constraints_func_feasible_and_infeasible_kept():
    import pandas as pd

    master = _master()
    combo = {"symbol": "AUDCAD", "timeframe": "H1"}
    good = {"sharpe_net": 1.2, "pf": 1.5, "n_trades": 150, "maxdd_pct": 10.0, "exposure_pct": 40.0, "r2": 0.6}
    bad = {"sharpe_net": 2.0, "pf": 0.8, "n_trades": 150, "maxdd_pct": 10.0, "exposure_pct": 40.0, "r2": 0.6}
    idx = pd.date_range("2021-01-01", periods=200, freq="D")
    daily = pd.Series(0.001, index=idx)

    for summary, feasible in ((good, True), (bad, False)):
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(
            seed=1, constraints_func=lambda tr: tr.user_attrs.get("constraints", [])))
        study.optimize(make_objective(master, combo, window=(idx[0], idx[-1]),
                                       run_fn=_fake_run(summary, daily)), n_trials=2)
        for t in study.trials:
            # infactibles conservan evidencia: NO se podan tras backtest completo
            assert t.state.name == "COMPLETE"
            assert t.user_attrs["feasible"] is feasible
            assert len(t.user_attrs["constraints"]) == 4


def test_robust_score_prefers_plateau():
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2021-01-01", periods=252, freq="D")
    rng = np.random.default_rng(7)
    plateau = pd.Series(0.001 + rng.normal(0, 0.002, len(idx)), index=idx)
    spike = pd.Series(rng.normal(0, 0.002, len(idx)), index=idx)
    spike.iloc[100:110] += 0.03  # pico concentrado: media mayor pero inestable
    assert robust_score(plateau) > robust_score(spike)


def test_sample_search_space_respects_step():
    study = optuna.create_study()
    trial = study.ask()
    p = sample_search_space(trial, {"bb_period": {"type": "int", "low": 10, "high": 30, "step": 2}})
    assert p["bb_period"] % 2 == 0
