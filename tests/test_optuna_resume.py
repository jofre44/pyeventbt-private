"""Tests Etapa 1: resume Optuna (20+20==40) + determinismo con seed derivada."""
from datetime import datetime

import optuna

from pyeventbt.optimization.adapter import derive_seed
from pyeventbt.optimization.runner_optuna import optimize


def _fake_master(tmp_path, n_trials=20):
    return {
        "seed": 1500,
        "strategy": {"name": "bbands_breakout", "cash": 10000, "factory": "f",
                     "params_class": "c", "strategy_id": "11313",
                     "search_space": {"x": {"type": "int", "low": 1, "high": 5}}},
        "universe": {"data_dir": str(tmp_path)},
        "costs": {"spread_mult": 1.0},
        "optimization": {
            "direction": "maximize", "n_trials": n_trials, "n_startup_random": 2,
            "sampler": "TPE", "pruner": "none", "n_jobs": 1, "deterministic_mode": True,
            "storage": f"sqlite:///{tmp_path}/optuna.db",
            "constraints": {"min_trades": 1, "max_maxdd_pct": 100.0, "min_pf": 0.0, "max_exposure_pct": 100.0},
            "robust": {"enabled": False}, "early_gate": {"enabled": False},
        },
        "cluster": {"max_clusters_kept": 5},
        "research_ledger": {"path": str(tmp_path / "ledger.parquet")},
    }


def _fake_objective(values):
    state = {"i": 0}

    def _obj(trial):
        i = state["i"] % len(values)
        state["i"] += 1
        trial.set_user_attr("sharpe_net_full", values[i])
        trial.set_user_attr("n_trades", 120)
        trial.set_user_attr("constraints", [0, 0, 0, 0])
        trial.set_user_attr("feasible", True)
        return values[i]

    return _obj


def test_optuna_resume_20_plus_20_eq_40(tmp_path):
    master = _fake_master(tmp_path)
    combo = {"strategy_family": "bbands_breakout", "symbol": "AUDCAD", "timeframe": "H1"}
    w = (datetime(2021, 1, 3), datetime(2021, 2, 3))
    optimize(master, combo, tmp_path / "o1", "run1", window=w,
             objective_fn=_fake_objective([0.5]), n_trials=20)
    res = optimize(master, combo, tmp_path / "o1", "run1", window=w,
                   objective_fn=_fake_objective([0.5]), n_trials=40)
    assert res["S"] == 40


def test_derive_seed():
    assert derive_seed(1500, 0) == 1500
    assert derive_seed(1500, 5) == 1505
    assert derive_seed(1500, 5) == derive_seed(1500, 5)  # determinista
    assert derive_seed(1500, 5) != derive_seed(1500, 6)  # distinta por trial


def test_sampler_constraints_func_supported():
    import inspect

    sig = inspect.signature(optuna.samplers.TPESampler.__init__)
    assert "constraints_func" in sig.parameters  # verificado en Optuna instalada
