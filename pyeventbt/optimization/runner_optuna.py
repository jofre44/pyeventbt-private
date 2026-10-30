"""Runner Optuna (Etapa 1): sampler con constraints_func, persistencia de TODOS
los trials + daily_returns, filtro duro → Pareto/top-k → candidatos.csv,
volcado S/K al ledger global.

`optimize(master, combo, out_dir, run_id, ...)` -> dict resumen.
`objective_fn` inyectable para tests (evita backtests reales).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
import polars as pl

logger = logging.getLogger("pyeventbt.pipeline")


def build_sampler(master: dict[str, Any]) -> optuna.samplers.BaseSampler:
    cfg = master["optimization"]
    kind = cfg.get("sampler", "TPE")
    if kind == "NSGAII":
        return optuna.samplers.NSGAIISampler(
            seed=master["seed"], constraints_func=lambda trial: trial.user_attrs.get("constraints", [])
        )
    return optuna.samplers.TPESampler(
        seed=master["seed"],
        n_startup_trials=cfg.get("n_startup_random", 40),
        constraints_func=lambda trial: trial.user_attrs.get("constraints", []),
    )


def build_pruner(master: dict[str, Any]) -> optuna.pruners.BasePruner | None:
    name = master["optimization"].get("pruner", "median")
    if not name or name == "none":
        return None
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(f"pruner desconocido: {name}")


def build_storage(master: dict[str, Any], out_dir: str | Path):
    """Storage persistente del estudio (resume ante crash mid-run).

    Orden de prioridad:
      1. ``optimization.storage`` explícito (sqlite://, journal:// por fichero de
         Optuna NO es válido — ``journal://...`` como string lo parsea SQLAlchemy
         y lanza ``NoSuchModuleError`` en Optuna 5).
      2. JournalStorage sobre fichero ``<out_dir>/optuna_journal.log``, que
         tolera workers concurrentes y permite reanudar tras OOM/kill.
         NOTA versión: construir el OBJETO ``JournalStorage`` directamente;
         la pseudo-URL ``journal://`` no está soportada desde Optuna 3.x.
    """
    raw = master["optimization"].get("storage")
    if raw:
        return str(raw)
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(out / "optuna_journal.log")))


def optimize(
    master: dict[str, Any],
    combo: dict[str, Any],
    out_dir: str | Path,
    run_id: str,
    *,
    window: tuple[datetime, datetime] | None = None,
    objective_fn: Callable[[optuna.Trial], float] | None = None,
    n_trials: int | None = None,
) -> dict[str, Any]:
    from pyeventbt.optimization.objective import make_objective
    from pyeventbt.research.ledger import append_trials

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = window or (
        datetime.fromisoformat(str(master["universe"]["windows"]["IS"][0])),
        datetime.fromisoformat(str(master["universe"]["windows"]["IS"][1])),
    )
    n = n_trials or int(master["optimization"]["n_trials"])
    daily_store: dict[int, Any] = {}

    def _sink(trial_number: int, daily: Any) -> None:
        daily_store[trial_number] = daily

    obj = objective_fn or make_objective(master, combo, window=w, daily_sink=_sink)
    storage = build_storage(master, out)
    study_name = f"{combo.get('strategy_family', master['strategy']['name'])}_{combo['symbol']}_{combo['timeframe']}"
    study = optuna.create_study(
        study_name=study_name,
        direction=master["optimization"].get("direction", "maximize"),
        sampler=build_sampler(master),
        pruner=build_pruner(master),
        storage=storage,
        load_if_exists=True,
    )
    studies_previos = len(study.trials)
    todo = max(0, n - studies_previos)
    if todo:
        study.optimize(obj, n_trials=todo, n_jobs=int(master["optimization"].get("n_jobs", 1)))

    trials_path, daily_path = persist_study(study, daily_store, out, study_name)
    candidatos = filter_candidates(study, master, out, study_name)
    s_total = len(study.trials)
    k = len(candidatos)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    feas = [t for t in done if t.user_attrs.get("feasible")]
    pool = feas or done
    best = (max(t.value for t in pool) if study.direction == "maximize" else min(t.value for t in pool)) if pool else None

    # Volcado al ledger global (S y K del run). Idempotente ante REANUDACIÓN:
    # si un trial ya está en el ledger (misma run_id+combo+trial+ventana) no se
    # duplica; el ledger sigue siendo append-only (nunca encoge) y S_global no
    # se infla al reanudar un estudio con JournalStorage.
    try:
        from strategies.bbands_breakout.params import params_hash  # noqa
        from pyeventbt.research.ledger import read_ledger

        window_tag = f"IS[{w[0].date()},{w[1].date()}]"
        combo_tag = f"{combo['symbol']}_{combo['timeframe']}"
        ledger_path = master["research_ledger"]["path"]
        ya = set()
        try:
            ldf = read_ledger(ledger_path)
            if not ldf.is_empty():
                for r in ldf.iter_rows(named=True):
                    ya.add((r.get("run_id"), r.get("combo"), r.get("trial_number"), r.get("window")))
        except Exception:
            ya = set()
        rows = []
        for t in study.trials:
            key = (run_id, combo_tag, t.number, window_tag)
            if key in ya:
                continue
            rows.append({
                "run_id": run_id,
                "strategy_family": combo.get("strategy_family", master["strategy"]["name"]),
                "combo": combo_tag,
                "trial_number": t.number,
                "params_hash": str(t.user_attrs.get("params_hash", "")),
                "window": window_tag,
                "sharpe_net": t.user_attrs.get("sharpe_net_full"),
                "n_trades": t.user_attrs.get("n_trades"),
                "daily_returns_ref": f"{study_name}/daily_returns.parquet#trial_{t.number}",
            })
        if rows:
            append_trials(ledger_path, rows)
    except Exception as exc:
        logger.warning(f"ledger no actualizado (no bloquea): {exc}")

    return {
        "study_name": study_name, "S": s_total, "K": k,
        "trials_path": str(trials_path), "daily_path": str(daily_path),
        "candidatos_path": str(out / f"{study_name}_candidatos.csv"),
        "best_value": best, "best_feasible": bool(feas),
    }


def persist_study(
    study: optuna.Study, daily_store: dict[int, Any], out_dir: str | Path, name: str
) -> tuple[Path, Path]:
    out = Path(out_dir)
    recs = []
    for t in study.trials:
        rec = {
            "trial_number": t.number,
            "value": t.value,
            "state": t.state.name,
            "feasible": t.user_attrs.get("feasible"),
            "pruned_at": "prefix" if t.state == optuna.trial.TrialState.PRUNED else None,
            # Params REDONDEADOS al step (los realmente backtesteados); Optuna
            # guarda el float crudo de suggest_* con ruido binario.
            **{f"param_{k}": v for k, v in (t.user_attrs.get("params_rounded") or t.params or {}).items()},
            **{f"attr_{k}": (str(v) if isinstance(v, (list, dict)) else v)
               for k, v in (t.user_attrs or {}).items() if k != "constraints"},
        }
        recs.append(rec)
    trials_path = out / f"{name}_trials.parquet"
    pl.DataFrame(recs).write_parquet(trials_path)

    daily_path = out / f"{name}_daily_returns.parquet"
    # Al REANUDAR tras un crash, `daily_store` solo contiene los trials de ESTA
    # sesión; los trials previos ya tienen su columna trial_N en el parquet.
    # Se mezcla por nombre de columna (contenido determinista para el mismo
    # trial: mismo seed derivado + mismos costs) en vez de sobrescribir.
    cols: dict[str, pd.Series] = {}
    if daily_path.exists():
        try:
            prev = pl.read_parquet(daily_path).to_pandas()
            if "date" in prev.columns:
                prev = prev.set_index("date")
            for c in prev.columns:
                cols[str(c)] = prev[c]
        except Exception as exc:  # fichero ilegible: se reconstruye con lo nuevo
            logger.warning(f"daily_returns previo ilegible ({exc}); se reescribe")
    for k, v in daily_store.items():
        s = pd.Series(v)
        s.index.name = "date"
        cols[f"trial_{k}"] = s
    if cols:
        frame = pd.DataFrame(cols)
        pl.from_pandas(frame.reset_index(names=["date"]).astype({"date": str})).write_parquet(daily_path)
    else:
        pl.DataFrame({"date": []}).write_parquet(daily_path)
    return trials_path, daily_path


def filter_candidates(
    study: optuna.Study, master: dict[str, Any], out_dir: str | Path, name: str
) -> pd.DataFrame:
    c = master["optimization"]["constraints"]
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        a = t.user_attrs
        if (
            (a.get("n_trades") or 0) >= c["min_trades"]
            and (a.get("maxdd_pct") if a.get("maxdd_pct") is not None else 1e9) <= c["max_maxdd_pct"]
            and (a.get("exposure_pct") if a.get("exposure_pct") is not None else 1e9) <= c["max_exposure_pct"]
            and (a.get("pf") if a.get("pf") is not None else -1e9) >= c["min_pf"]
        ):
            rows.append({"trial_number": t.number, "score": t.value, **t.params})
    df = pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame(
        columns=["trial_number", "score"])
    k = 3 * int(master["cluster"].get("max_clusters_kept", 250))
    top = df.head(k) if not df.empty else df
    top.to_csv(Path(out_dir) / f"{name}_candidatos.csv", index=False)
    return top
