"""Walk-forward anclado con re-optimización (Etapa 3B): mide el PROCEDIMIENTO.

Train creciente, test 12m, ~5 folds. Por fold: mini-Optuna (n_trials_per_fold)
+ mejor factible → evaluación en test con embargo. WFE = Sharpe_test /
Sharpe_train. Gate: WFE ≥ 0.5 y ≥70% folds positivos. Todos los trials de los
folds cuentan en el ledger (vía optimize).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd

logger = logging.getLogger("pyeventbt.pipeline")


def anchored_folds(is_start: datetime, is_end: datetime, n_folds: int = 5,
                   test_months: int = 12) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Devuelve [(train_start, train_end, test_start, test_end)] anclados."""
    import pandas as pd

    start = pd.Timestamp(is_start)
    end = pd.Timestamp(is_end)
    total_m = (end.year - start.year) * 12 + (end.month - start.month)
    test_m = test_months
    # El último test termina en is_end; los test son consecutivos hacia atrás
    folds = []
    for i in range(n_folds):
        te = end - pd.DateOffset(months=(n_folds - 1 - i) * test_m)
        ts = te - pd.DateOffset(months=test_m)
        tr_s = start
        tr_e = ts
        if tr_e <= tr_s + pd.DateOffset(months=6):
            continue  # train mínimo 6 meses
        folds.append((tr_s.to_pydatetime(), tr_e.to_pydatetime(),
                      ts.to_pydatetime(), te.to_pydatetime()))
    return folds


def walkforward(
    master: dict[str, Any],
    combo: dict[str, Any],
    out_dir: str | Path,
    run_id: str,
    *,
    objective_fn_factory: Callable[[tuple[datetime, datetime]], Callable[[optuna.Trial], float]] | None = None,
    eval_fn: Callable[[dict[str, Any], tuple[datetime, datetime]], dict] | None = None,
) -> dict[str, Any]:
    """`objective_fn_factory(train_window)` y `eval_fn(params, test_window)` son
    inyectables (tests). Por defecto usan el pipeline real (lento)."""
    from pyeventbt.optimization.runner_optuna import optimize

    cfg = master["walkforward_cpcv"]["wf"]
    w = master["universe"]["windows"]["IS"]
    is_start, is_end = datetime.fromisoformat(str(w[0])), datetime.fromisoformat(str(w[1]))
    folds = anchored_folds(is_start, is_end, cfg.get("n_folds", 5), cfg.get("test_months", 12))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if objective_fn_factory is None or eval_fn is None:
        from pyeventbt.optimization.objective import make_objective
        from pyeventbt.optimization.adapter import run_pyeventbt_once

        def objective_fn_factory(train_w):  # type: ignore[misc]
            return make_objective(master, combo, window=train_w)

        def eval_fn(params, test_w):  # type: ignore[misc]
            from pyeventbt.optimization.adapter import derive_seed, resolve_base_spread

            return run_pyeventbt_once(
                params, combo, costs=master["costs"], window=test_w,
                cash=master["strategy"]["cash"], seed=derive_seed(master["seed"], 0),
                factory_path=master["strategy"]["factory"],
                params_class_path=master["strategy"]["params_class"],
                data_dir=master["universe"]["data_dir"],
                strategy_id=master["strategy"].get("strategy_id", "11313"),
                account_currency=master["strategy"].get("account_currency", "USD"),
                aux_fallback_rates=master["universe"].get("aux_fallback_rates", {}),
                base_spread_points=resolve_base_spread(master, combo["symbol"]),
            )

    rows = []
    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        res = optimize(master, combo, out / f"fold_{fi}", f"{run_id}_wf{fi}",
                       window=(tr_s, tr_e),
                       objective_fn=objective_fn_factory((tr_s, tr_e)),
                       n_trials=int(cfg.get("n_trials_per_fold", 20)))
        import polars as pl

        trials = pl.read_parquet(res["trials_path"])
        d = trials.to_dict(as_series=False)
        feas = [i for i, f in enumerate(d.get("feasible", [])) if f]
        pool = feas or list(range(len(d["trial_number"])))
        vals = [d["value"][i] for i in pool]
        best_i = pool[int(np.nanargmax([v if v is not None else -np.inf for v in vals]))]
        params = {k.replace("param_", ""): d[k][best_i]
                  for k in d if k.startswith("param_")}
        ev = eval_fn(params, (te_s, te_e))
        s_test = ev["summary"]["sharpe_net"]
        # Sharpe train del elegido: attr real, no el score robusto
        train_sr = (d.get("attr_sharpe_net_full") or [None] * len(d["trial_number"]))[best_i]
        if train_sr is None:
            train_sr = d["value"][best_i]
        train_sr = float(train_sr) if train_sr is not None else float("nan")
        s_test = float(s_test) if s_test is not None else float("nan")
        rows.append({"fold": fi, "sharpe_train": train_sr,
                     "sharpe_test": s_test, "params": params,
                     "n_trials": len(d["trial_number"])})
    df = pd.DataFrame(rows)
    pos = float((df["sharpe_test"] > 0).mean()) if len(df) else 0.0
    with np.errstate(all="ignore"):
        ratio = (df["sharpe_test"] / df["sharpe_train"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    wfe = float(ratio.median()) if len(ratio) and ratio.notna().any() else float("nan")
    ok = (wfe >= cfg.get("wfe_min", 0.5)) and (pos >= cfg.get("min_folds_positive", 0.70))
    df.to_csv(out / "walkforward.csv", index=False)
    return {"folds": rows, "wfe": wfe, "frac_positive": pos,
            "verdict": "PASS" if ok else "FAIL"}
