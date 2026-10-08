"""Objetivo Optuna §1.1: screening barato, backtest idéntico, constraints del
sampler (NO TrialPruned tras backtest completo), objetivo robusto meseta.

`make_objective(master, combo, ...)` devuelve `fn(trial) -> float`.
Guarda daily_returns por trial vía callback `daily_sink(trial_number, daily)`.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

import numpy as np
import optuna

logger = logging.getLogger("pyeventbt.pipeline")


def sample_search_space(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif t == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"tipo de espacio desconocido: {t} ({name})")
    return params


def subwindow_sharpes(daily, n: int = 4) -> np.ndarray:
    import pandas as pd

    s = pd.Series(daily).dropna()
    if len(s) < 4 * n:
        return np.array([float("nan")])
    parts = np.array_split(s.values, n)
    out = []
    for p in parts:
        sd = float(np.std(p, ddof=1))
        out.append(float(np.mean(p) / sd * np.sqrt(252)) if sd > 0 else float("nan"))
    return np.array(out)


def robust_score(daily, lambda_std: float = 0.5, n_subwindows: int = 4) -> float:
    sw = subwindow_sharpes(daily, n_subwindows)
    sw = sw[np.isfinite(sw)]
    if len(sw) == 0:
        return float("-inf")
    return float(np.mean(sw) - lambda_std * np.std(sw))


def make_objective(
    master: dict[str, Any],
    combo: dict[str, Any],
    *,
    window: tuple[datetime, datetime],
    daily_sink: Callable[[int, Any], None] | None = None,
    run_fn: Callable | None = None,
) -> Callable[[optuna.Trial], float]:
    """`run_fn` inyectable para tests (firma como run_pyeventbt_once)."""
    from pyeventbt.optimization.adapter import derive_seed, run_pyeventbt_once

    cfg = master["optimization"]
    run_fn = run_fn or run_pyeventbt_once
    _run_kw: dict[str, Any] = {
        "factory_path": master["strategy"]["factory"],
        "params_class_path": master["strategy"]["params_class"],
        "data_dir": master["universe"]["data_dir"],
        "strategy_id": master["strategy"].get("strategy_id", "11313"),
        "account_currency": master["strategy"].get("account_currency", "USD"),
        "aux_fallback_rates": master["universe"].get("aux_fallback_rates", {}),
        "base_spread_points": master["costs"].get("base_spread_points", 20.0),
    }

    def objective(trial: optuna.Trial) -> float:
        from pyeventbt.optimization.cluster import round_to_space

        params = round_to_space(
            sample_search_space(trial, master["strategy"]["search_space"]),
            master["strategy"]["search_space"],
        )
        seed = derive_seed(master["seed"], trial.number)

        if cfg.get("early_gate", {}).get("enabled"):
            from pyeventbt.analytics.stats import summarize  # noqa

            w0, w1 = window
            mid = w0 + (w1 - w0) * float(cfg["early_gate"].get("at_fraction", 0.5))
            fast = run_fn(
                params, combo, costs=master["costs"], window=(w0, mid),
                cash=master["strategy"]["cash"], seed=seed, **_run_kw,
            )
            trial.set_user_attr("n_trades_fast", fast["summary"]["n_trades"])
            trial.report(fast["summary"]["sharpe_net"], step=0)
            if (
                fast["summary"]["n_trades"] < cfg["early_gate"].get("min_trades_fast", 30)
                or trial.should_prune()
            ):
                raise optuna.TrialPruned()

        res = run_fn(
            params, combo, costs=master["costs"], window=window,
            cash=master["strategy"]["cash"], seed=seed, **_run_kw,
        )
        s = res["summary"]
        if daily_sink is not None:
            daily_sink(trial.number, res["daily"])

        for k in ("pf", "n_trades", "maxdd_pct", "exposure_pct", "r2"):
            v = s.get(k)
            trial.set_user_attr(k, float(v) if v is not None and np.isfinite(v) else None)
        trial.set_user_attr("sharpe_net_full", s["sharpe_net"])
        trial.set_user_attr("params_hash", res.get("params_hash"))
        trial.set_user_attr("params_rounded", params)

        c = cfg["constraints"]
        trial.set_user_attr("constraints", [
            float(c["min_trades"]) - float(s["n_trades"]),
            float(s["maxdd_pct"]) - float(c["max_maxdd_pct"]),
            float(c["min_pf"]) - float(s["pf"]),
            float(s["exposure_pct"]) - float(c["max_exposure_pct"]),
        ])
        trial.set_user_attr("feasible", bool(
            s["n_trades"] >= c["min_trades"]
            and s["maxdd_pct"] <= c["max_maxdd_pct"]
            and s["pf"] >= c["min_pf"]
            and s["exposure_pct"] <= c["max_exposure_pct"]
        ))

        r = cfg.get("robust", {})
        if r.get("enabled"):
            return robust_score(res["daily"], r.get("lambda_std", 0.5), r.get("n_subwindows", 4))
        return float(s["sharpe_net"])

    return objective
