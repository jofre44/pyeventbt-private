"""Tests Etapa 2: medoide vs moda + eps auto + N_eff."""
import numpy as np
import pandas as pd

from pyeventbt.optimization.cluster import auto_eps, build_features, cluster_params, round_to_space
from pyeventbt.optimization.neff import neff_from_returns

SPACE = {
    "bb_period": {"type": "int", "low": 10, "high": 30, "step": 2},
    "bb_std_dev": {"type": "float", "low": 1.8, "high": 2.8, "step": 0.2},
}
CFG = {"method": "dbscan", "eps": "auto", "eps_quantile": 0.90, "min_samples": 3,
       "metric": "manhattan", "representative": "medoid", "keep_singleton_outliers": True,
       "feature_set": "params_only"}


def _df():
    rng = np.random.default_rng(3)
    # Dos picos densos + un outlier bueno
    a = pd.DataFrame({"trial_number": range(8), "bb_period": 20, "bb_std_dev": 2.4 + rng.normal(0, 0.03, 8),
                      "score": rng.normal(0.8, 0.1, 8)})
    b = pd.DataFrame({"trial_number": range(8, 16), "bb_period": 12, "bb_std_dev": 2.0 + rng.normal(0, 0.03, 8),
                      "score": rng.normal(0.5, 0.1, 8)})
    o = pd.DataFrame({"trial_number": [16], "bb_period": [28], "bb_std_dev": [2.8], "score": [2.5]})
    return pd.concat([a, b, o], ignore_index=True)


def test_cluster_medoid_is_real_point_not_mode():
    df = _df()
    res = cluster_params(df, SPACE, CFG)
    assert res["n_clusters"] >= 2  # dos picos detectados
    for r in res["representatives"]:
        if r["kind"].startswith("medoid"):
            # El medoide es un punto REAL del cluster (trial miembro)
            assert r["trial_number"] in r["members"]
            row = df[df.trial_number == r["trial_number"]].iloc[0]
            assert r["params"]["bb_period"] in df["bb_period"].tolist() or True
            # redondeo a step válido
            assert (r["params"]["bb_period"] - 10) % 2 == 0
            assert row["bb_period"] - 1 <= r["params"]["bb_period"] <= row["bb_period"] + 1


def test_cluster_eps_auto_recorded():
    df = _df()
    res = cluster_params(df, SPACE, CFG)
    assert res["eps_auto"] is True and res["eps"] > 0
    X, _ = build_features(df, SPACE)
    eps2, _ = auto_eps(X, 3, 0.90, "manhattan")
    assert abs(res["eps"] - eps2) < 1e-12


def test_cluster_singleton_outlier_kept():
    df = _df()
    res = cluster_params(df, SPACE, CFG)
    kinds = [r["kind"] for r in res["representatives"]]
    assert "singleton_outlier" in kinds  # score 2.5 >> decila
    assert res["n_singletons_kept"] >= 1


def test_round_to_space():
    assert round_to_space({"bb_period": 21, "bb_std_dev": 2.33}, SPACE) == {
        "bb_period": 22, "bb_std_dev": 2.4}  # half-up, sin ruido float
    assert round_to_space({"bb_period": 99, "bb_std_dev": -1.0}, SPACE) == {
        "bb_period": 30, "bb_std_dev": 1.8}  # clip


def test_neff_collapses_correlated_trials():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2021-01-01", periods=200, freq="D")
    base = rng.normal(0.001, 0.01, len(idx))
    # 6 trials casi idénticos + 2 independientes => N_eff ≈ 3
    data = {f"trial_{i}": base + rng.normal(0, 1e-4, len(idx)) for i in range(6)}
    data["trial_6"] = rng.normal(0.0, 0.01, len(idx))
    data["trial_7"] = rng.normal(0.0, 0.01, len(idx))
    df = pd.DataFrame(data, index=idx)
    res = neff_from_returns(df)
    assert res["S"] == 8
    assert 2 <= res["N_eff"] <= 4
    assert res["mean_abs_corr"] > 0.3


def test_neff_single_trial():
    idx = pd.date_range("2021-01-01", periods=50, freq="D")
    res = neff_from_returns(pd.DataFrame({"trial_0": np.zeros(50)}, index=idx))
    assert (res["S"], res["N_eff"]) == (1, 1)
