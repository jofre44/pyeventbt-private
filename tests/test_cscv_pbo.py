"""Tests Etapa 3B: CSCV/PBO, CPCV y walk-forward con casos conocidos."""
import numpy as np
import pandas as pd

from pyeventbt.robustness.cpcv import cpcv_distribution
from pyeventbt.robustness.cscv_pbo import cscv_pbo
from pyeventbt.robustness.walkforward import anchored_folds, walkforward


def _matrix(seed=0, n=500, k=20, edge_col=None, edge_mu=0.0):
    rng = np.random.default_rng(seed)
    data = {f"t{i}": rng.normal(0, 0.01, n) for i in range(k)}
    if edge_col is not None:
        data[edge_col] = rng.normal(edge_mu, 0.01, n)
    return pd.DataFrame(data, index=pd.date_range("2021-01-01", periods=n, freq="D"))


def test_cscv_pbo_noise_is_high():
    df = _matrix(seed=1)
    r = cscv_pbo(df, n_groups=10, max_splits=100)
    assert r["verdict"] in ("PASS", "FAIL")
    # En ruido puro el mejor de train no persiste: PBO ≈ 0.5
    assert 0.3 <= r["pbo"] <= 0.7


def test_cscv_pbo_planted_edge_is_low():
    # Edge fuerte y estable en una columna => el mejor de train persiste en test
    df = _matrix(seed=2, edge_col="t0", edge_mu=0.006)
    r = cscv_pbo(df, n_groups=10, max_splits=100)
    assert r["pbo"] <= 0.30
    assert r["verdict"] == "PASS"


def test_cpcv_pass_and_fail():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2021-01-01", periods=500, freq="D")
    good = pd.Series(rng.normal(0.002, 0.01, 500), index=idx)
    bad = pd.Series(rng.normal(-0.002, 0.01, 500), index=idx)
    assert cpcv_distribution(good)["verdict"] == "PASS"
    assert cpcv_distribution(bad)["verdict"] == "FAIL"


def test_anchored_folds_cover_is():
    from datetime import datetime

    folds = anchored_folds(datetime(2021, 1, 3), datetime(2023, 12, 31), n_folds=5, test_months=12)
    assert len(folds) >= 2
    # Anclado: todos los trains empiezan igual, los tests son consecutivos
    assert all(f[0] == folds[0][0] for f in folds)
    assert folds[-1][3] == datetime(2023, 12, 31)


def _wf_master(tmp_path):
    return {
        "seed": 1500,
        "strategy": {"name": "s", "cash": 10000, "factory": "f", "params_class": "c",
                     "strategy_id": "1",
                     "search_space": {"x": {"type": "int", "low": 1, "high": 3}}},
        "universe": {"data_dir": str(tmp_path),
                     "windows": {"IS": ["2021-01-01", "2022-12-31"]}},
        "costs": {"spread_mult": 1.0},
        "optimization": {
            "direction": "maximize", "n_trials": 4, "n_startup_random": 1,
            "sampler": "TPE", "pruner": "none", "n_jobs": 1, "deterministic_mode": True,
            "storage": None,
            "constraints": {"min_trades": 1, "max_maxdd_pct": 100.0, "min_pf": 0.0, "max_exposure_pct": 100.0},
            "robust": {"enabled": False}, "early_gate": {"enabled": False},
        },
        "cluster": {"max_clusters_kept": 5},
        "walkforward_cpcv": {"wf": {"n_folds": 2, "test_months": 6, "wfe_min": 0.5,
                                    "min_folds_positive": 0.5, "n_trials_per_fold": 4}},
        "research_ledger": {"path": str(tmp_path / "ledger.parquet")},
    }


def test_walkforward_with_fake_objective(tmp_path):
    import optuna

    master = _wf_master(tmp_path)
    combo = {"strategy_family": "s", "symbol": "X", "timeframe": "H1"}

    def factory(train_w):
        def _obj(trial):
            x = trial.suggest_int("x", 1, 3)
            trial.set_user_attr("sharpe_net_full", 0.5 + 0.1 * x)
            trial.set_user_attr("n_trades", 200)
            trial.set_user_attr("constraints", [0, 0, 0, 0])
            trial.set_user_attr("feasible", True)
            return 0.5 + 0.1 * x
        return _obj

    def eval_fn(params, test_w):
        return {"summary": {"sharpe_net": 0.4}}

    res = walkforward(master, combo, tmp_path / "wf", "r1",
                      objective_fn_factory=factory, eval_fn=eval_fn)
    assert len(res["folds"]) == 2
    assert res["frac_positive"] == 1.0
    assert res["verdict"] == "PASS"  # WFE 0.4/0.8=0.5 >= 0.5
    # Todos los trials de folds cuentan en el ledger
    from pyeventbt.research.ledger import read_ledger

    assert len(read_ledger(master["research_ledger"]["path"])) == 8
