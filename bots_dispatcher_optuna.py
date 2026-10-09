"""Dispatcher del pipeline (Transversal): `python bots_dispatcher_optuna.py MASTER
[--only prep,optimize,cluster,synth,wfcpcv,select,calibrate] [--pair X] [--timeframe Y]
[--run-dir DIR] [--max-reps N] [--auto-lockbox] [--report RUN_DIR]`.

Orquesta por combo (símbolo × TF) con ProcessLog reanudable (resume por
config_hash + por representante dentro de synth/select). Sin `print` en
pipeline: logging. Dashboard Rich diferido (nota en ASSUMPTIONS, sec. T).

Orden Etapa 4 garantizado por construcción: puertas → lockbox → sizing.
El lockbox (consumo irreversible del FORWARD) exige --auto-lockbox; sin la
bandera los sets que pasan puertas quedan en AWAITING_LOCKBOX y el FORWARD
intacto.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pyeventbt.pipeline")

STAGES = ["prep", "optimize", "cluster", "synth", "wfcpcv", "select", "calibrate"]


def _run_once_for(master: dict, combo: dict, params: dict,
                  window: tuple, seed: int, costs: dict | None = None) -> dict:
    """Atajo que arma los kwargs del adaptador desde el maestro + combo."""
    from pyeventbt.optimization.adapter import resolve_base_spread, run_pyeventbt_once

    return run_pyeventbt_once(
        params, combo, costs=costs or master["costs"], window=window,
        cash=master["strategy"]["cash"], seed=seed,
        factory_path=master["strategy"]["factory"],
        params_class_path=master["strategy"]["params_class"],
        data_dir=master["universe"]["data_dir"],
        strategy_id=master["strategy"].get("strategy_id", "11313"),
        account_currency=master["strategy"].get("account_currency", "USD"),
        aux_fallback_rates=master["universe"].get("aux_fallback_rates", {}),
        base_spread_points=resolve_base_spread(master, combo["symbol"]))


def new_run_dir(master: dict, base: str = "runs") -> Path:
    try:
        gh = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     text=True).strip()
    except Exception:
        gh = "nogit"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path(base) / master["strategy"]["name"] / f"{ts}_{gh}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def combos(master: dict, pair: str | None, timeframe: str | None) -> list[dict]:
    syms = [pair] if pair else master["universe"]["symbols"]
    tfs = [timeframe] if timeframe else master["universe"]["timeframes"]
    return [{"strategy_family": master["strategy"]["name"], "symbol": s, "timeframe": t}
            for s in syms for t in tfs]


def run_prep(master: dict, combo: list[dict], run_dir: Path, run_id: str, log) -> None:
    from pyeventbt.runner import data_manifest as dm
    from pyeventbt.runner.master_config import config_hash, immutable_hash
    from pyeventbt.robustness.power import power_report
    from pyeventbt.optimization.adapter import resolve_base_spread, run_pyeventbt_once
    import yaml

    (run_dir / "master_used.yaml").write_text(
        yaml.safe_dump(master, sort_keys=True), encoding="utf-8")
    (run_dir / "master_hash.txt").write_text(
        config_hash(master) + "\n" + immutable_hash(master) + "\n", encoding="utf-8")
    (run_dir / "immutable_hash.txt").write_text(immutable_hash(master), encoding="utf-8")
    w = master["universe"]["windows"]
    is_w = (datetime.fromisoformat(str(w["IS"][0])), datetime.fromisoformat(str(w["IS"][1])))
    fwd = (datetime.fromisoformat(str(w["FORWARD"][0])), datetime.fromisoformat(str(w["FORWARD"][1])))
    # warmup = max lookback del espacio; holding = barras de 1 día de time-stop
    # en el TF señal (M30:48, H1:24, H4:6, D1:1).
    warmup = max(v.get("high", 20) for v in master["strategy"]["search_space"].values()
                 if isinstance(v, dict) and v.get("type") in ("int", "float"))
    for c in combo:
        sym, tf = c["symbol"], c["timeframe"]
        if log.is_done(sym, tf, "prep", config_hash(master)):
            continue
        log.init_stage(sym, tf, "prep", config_hash(master))
        prep = run_dir / "00_prep"
        (prep / "manifests").mkdir(parents=True, exist_ok=True)
        man = dm.build_manifest(sym, Path(master["universe"]["data_dir"]) / f"{sym}.csv",
                                master["universe"].get("broker_server_tz", "UTC"))
        (prep / "manifests" / f"{sym}.json").write_text(
            __import__("json").dumps(man, indent=1, default=str), encoding="utf-8")
        chk = dm.data_check(man)
        holding_bars = max(1, 1440 // dm.TF_MINUTES[tf])
        pur = dm.check_window_purity(is_w[1], fwd[0], tf, holding_bars, int(warmup),
                                     master["universe"]["embargo"].get("extra_bars", 10))
        if chk["verdict"] == "FAIL" or pur["verdict"] == "FAIL":
            log.mark_error(sym, tf, "prep", f"data_check={chk} purity={pur}")
            raise RuntimeError(f"PREP FAIL {sym} {tf}: {chk['fails']} {pur}")
        # Nominal: ¿el combo da ~min_trades_IS? + años para potencia
        from strategies.bbands_breakout.params import BBandsParams

        nominal = {k: v.model_dump()[k] for k, v in [("x", BBandsParams())] for k in ("bb_period", "bb_std_dev")}
        r = run_pyeventbt_once(
            nominal, c, costs=master["costs"], window=is_w, cash=master["strategy"]["cash"],
            seed=master["seed"], factory_path=master["strategy"]["factory"],
            params_class_path=master["strategy"]["params_class"],
            data_dir=master["universe"]["data_dir"],
            strategy_id=master["strategy"].get("strategy_id", "11313"),
            account_currency=master["strategy"].get("account_currency", "USD"),
            aux_fallback_rates=master["universe"].get("aux_fallback_rates", {}),
            base_spread_points=resolve_base_spread(master, sym))
        years = (is_w[1] - is_w[0]).days / 365.25
        pw = power_report(r["daily"], r["summary"]["n_trades"], years,
                          master["universe"]["power"].get("target_sharpe", 0.8),
                          master["universe"]["power"].get("alpha", 0.05))
        # budget.json: n_trials fijado antes (50-100 por dimensión efectiva)
        n_dim = len(master["strategy"]["search_space"])
        (prep / "budget.json").write_text(__import__("json").dumps({
            "n_trials": master["optimization"]["n_trials"],
            "rule": "50-100 por dimensión", "n_dim": n_dim,
            "range_ok": 50 * n_dim <= master["optimization"]["n_trials"] <= 100 * n_dim,
            "seed_derivation": "seed+trial_number"}, indent=1), encoding="utf-8")
        (prep / "power.json").write_text(__import__("json").dumps({
            **pw, "nominal_n_trades_IS": r["summary"]["n_trades"],
            "nominal_sharpe_IS": r["summary"]["sharpe_net"]}, indent=1, default=str),
            encoding="utf-8")
        shutil.copy(master["strategy"]["spec"], prep / "spec.yaml")
        log.mark_done(sym, tf, "prep", counts={"nominal_trades": r["summary"]["n_trades"]})
        logger.warning(f"[prep] {sym} {tf}: nominal_trades={r['summary']['n_trades']} power={pw['verdict']}")


def run_optimize(master: dict, combo: list[dict], run_dir: Path, run_id: str, log) -> None:
    from pyeventbt.optimization.runner_optuna import optimize
    from pyeventbt.runner.master_config import assert_immutable_unchanged, config_hash

    imm = (run_dir / "immutable_hash.txt").read_text().strip()
    w = master["universe"]["windows"]
    is_w = (datetime.fromisoformat(str(w["IS"][0])), datetime.fromisoformat(str(w["IS"][1])))
    for c in combo:
        if log.is_done(c["symbol"], c["timeframe"], "optimize", config_hash(master)):
            continue
        assert_immutable_unchanged(master, imm)
        log.init_stage(c["symbol"], c["timeframe"], "optimize", config_hash(master))
        out = run_dir / "optimization" / f"{c['symbol']}_{c['timeframe']}"
        res = optimize(master, c, out, run_id, window=is_w)
        log.mark_done(c["symbol"], c["timeframe"], "optimize", counts={"S": res["S"], "K": res["K"]})
        logger.warning(f"[optimize] {c['symbol']} {c['timeframe']}: S={res['S']} K={res['K']}")


def run_cluster(master: dict, combo: list[dict], run_dir: Path, run_id: str, log) -> None:
    import pandas as pd
    import polars as pl
    from pyeventbt.optimization.cluster import build_features, cluster_params, save_diagnostics
    from pyeventbt.optimization.neff import neff_from_returns
    from pyeventbt.runner.master_config import config_hash

    name = f"{master['strategy']['name']}"
    for c in combo:
        if log.is_done(c["symbol"], c["timeframe"], "cluster", config_hash(master)):
            continue
        log.init_stage(c["symbol"], c["timeframe"], "cluster", config_hash(master))
        base = run_dir / "optimization" / f"{c['symbol']}_{c['timeframe']}"
        tag = f"{name}_{c['symbol']}_{c['timeframe']}"
        trials = pl.read_parquet(base / f"{tag}_trials.parquet")
        d = trials.to_dict(as_series=False)
        space = master["strategy"]["search_space"]
        df = pd.DataFrame({"trial_number": d["trial_number"],
                           **{k: d[f"param_{k}"] for k in space},
                           "score": pd.Series(d["value"]).fillna(-99)})
        daily = pl.read_parquet(base / f"{tag}_daily_returns.parquet")
        cr = cluster_params(df, space, master["cluster"])
        X, _ = build_features(df, space)
        cdir = run_dir / "cluster" / f"{c['symbol']}_{c['timeframe']}"
        cdir.mkdir(parents=True, exist_ok=True)
        save_diagnostics(cr, X, cdir / "diagnostics")
        pd.DataFrame(cr["representatives"]).to_csv(cdir / f"{tag}.csv", index=False)
        neff = neff_from_returns(daily)
        (cdir / "neff.json").write_text(__import__("json").dumps(neff, indent=1), encoding="utf-8")
        log.mark_done(c["symbol"], c["timeframe"], "cluster",
                      counts={"clusters": cr["n_clusters"], "reps": len(cr["representatives"]),
                              "N_eff": neff["N_eff"]})
        logger.warning(f"[cluster] {c['symbol']} {c['timeframe']}: "
                       f"{cr['n_clusters']} clusters, {len(cr['representatives'])} reps, N_eff={neff['N_eff']}")


def _combo_tag(master: dict, c: dict) -> str:
    return f"{master['strategy']['name']}_{c['symbol']}_{c['timeframe']}"


def _load_reps(run_dir: Path, master: dict, c: dict) -> list[dict]:
    """Representantes del cluster con params parseados. [] si no hay."""
    import pandas as pd

    tag = _combo_tag(master, c)
    p = run_dir / "cluster" / f"{c['symbol']}_{c['timeframe']}" / f"{tag}.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    reps = []
    for i, row in df.iterrows():
        try:
            params = ast.literal_eval(str(row["params"]))
        except Exception:
            continue
        reps.append({"set_id": f"{tag}#t{int(row['trial_number'])}",
                     "trial_number": int(row["trial_number"]),
                     "params": params, "score": float(row.get("score", 0.0)),
                     "kind": str(row.get("kind", "")), "idx": int(i)})
    return reps


def _rep_daily(run_dir: Path, master: dict, c: dict, trial_number: int):
    """Serie diaria del trial (columna trial_N del daily_returns)."""
    import polars as pl

    tag = _combo_tag(master, c)
    base = run_dir / "optimization" / f"{c['symbol']}_{c['timeframe']}"
    df = pl.read_parquet(base / f"{tag}_daily_returns.parquet")
    col = f"trial_{trial_number}"
    if col not in df.columns:
        return None
    import pandas as pd

    d = df.to_dict(as_series=False)
    return pd.Series(d[col], index=pd.to_datetime(d["date"])).dropna()


def run_synth(master: dict, combo: list[dict], run_dir: Path, run_id: str, log,
              max_reps: int | None = None) -> None:
    from pyeventbt.robustness.validate_synthetic import run_consistency
    from pyeventbt.runner.master_config import config_hash
    import pandas as pd

    w = master["universe"]["windows"]
    is_w = (datetime.fromisoformat(str(w["IS"][0])), datetime.fromisoformat(str(w["IS"][1])))
    n_sims = int(master["validation"].get("n_sim_screening", 20))
    for c in combo:
        if log.is_done(c["symbol"], c["timeframe"], "synth", config_hash(master)):
            continue
        log.init_stage(c["symbol"], c["timeframe"], "synth", config_hash(master))
        reps = _load_reps(run_dir, master, c)[:max_reps] if max_reps else _load_reps(run_dir, master, c)
        sdir = run_dir / "synthetic_validation"
        sdir.mkdir(parents=True, exist_ok=True)
        outp = sdir / f"{_combo_tag(master, c)}.csv"
        done = set()
        if outp.exists():
            done = set(pd.read_csv(outp)["set_id"].tolist())
        rows = outp.exists() and pd.read_csv(outp).to_dict(orient="records") or []
        for r in reps:
            if r["set_id"] in done:
                continue
            try:
                res = run_consistency(r["params"], c, master, is_w, n_sims=n_sims,
                                      seed_base=int(master["seed"]), cluster_id=r["idx"])
                rows.append({
                    "set_id": r["set_id"], "trial_number": r["trial_number"],
                    "rank_sharpe": res["ranks"].get("sharpe_net"),
                    "rank_pf": res["ranks"].get("pf"),
                    "rank_ntrades": res["ranks"].get("n_trades"),
                    "consistency": res["consistency"]["verdict"],
                    "validation_overall": res["validation"].get("overall"),
                    "stress_pf": res.get("stress_pf"), "n_sims": n_sims})
            except Exception as exc:
                rows.append({"set_id": r["set_id"], "trial_number": r["trial_number"],
                             "consistency": "ERROR", "error": str(exc)[:200]})
            pd.DataFrame(rows).to_csv(outp, index=False)  # checkpoint por rep (resume)
        log.mark_done(c["symbol"], c["timeframe"], "synth", counts={"reps": len(reps)})
        logger.warning(f"[synth] {c['symbol']} {c['timeframe']}: {len(reps)} reps")


def run_wfcpcv(master: dict, combo: list[dict], run_dir: Path, run_id: str, log) -> None:
    from pyeventbt.optimization.neff import _as_pandas
    from pyeventbt.robustness.cpcv import cpcv_distribution
    from pyeventbt.robustness.cscv_pbo import cscv_pbo
    from pyeventbt.robustness.walkforward import walkforward
    from pyeventbt.runner.data_manifest import compute_embargo, TF_MINUTES
    from pyeventbt.runner.master_config import config_hash
    import polars as pl

    cfg = master.get("walkforward_cpcv", {})
    for c in combo:
        if log.is_done(c["symbol"], c["timeframe"], "wfcpcv", config_hash(master)):
            continue
        log.init_stage(c["symbol"], c["timeframe"], "wfcpcv", config_hash(master))
        tag = _combo_tag(master, c)
        base = run_dir / "optimization" / f"{c['symbol']}_{c['timeframe']}"
        daily = _as_pandas(pl.read_parquet(base / f"{tag}_daily_returns.parquet"))
        wdir = run_dir / "wf_cpcv" / f"{c['symbol']}_{c['timeframe']}"
        wdir.mkdir(parents=True, exist_ok=True)
        hold = max(1, 1440 // TF_MINUTES[c["timeframe"]])
        emb_days = max(1, round(compute_embargo(
            c["timeframe"], hold, 30,
            master["universe"]["embargo"].get("extra_bars", 10))["delta"].total_seconds() / 86400))
        pbo = cscv_pbo(daily, n_groups=int(cfg.get("pbo", {}).get("n_groups", 10)),
                       embargo_days=emb_days)
        (wdir / "pbo.json").write_text(json.dumps(
            {k: v for k, v in pbo.items() if k != "logits"}, indent=1, default=str), encoding="utf-8")
        reps = _load_reps(run_dir, master, c)
        cpcv_rows = []
        for r in reps:
            s = _rep_daily(run_dir, master, c, r["trial_number"])
            if s is None or len(s) < 50:
                cpcv_rows.append({"set_id": r["set_id"], "verdict": "NODATA"})
                continue
            d = cpcv_distribution(s, n_groups=int(cfg.get("cpcv", {}).get("n_groups", 10)),
                                  k_test_groups=int(cfg.get("cpcv", {}).get("k_test_groups", 2)),
                                  purge_days=emb_days)
            cpcv_rows.append({"set_id": r["set_id"], "median": d.get("median"),
                              "frac_positive": d.get("frac_positive"),
                              "verdict": d.get("verdict")})
        import pandas as pd

        pd.DataFrame(cpcv_rows).to_csv(wdir / "cpcv.csv", index=False)
        wf = walkforward(master, c, wdir / "wf", run_id)
        (wdir / "wfcpcv_summary.json").write_text(json.dumps(
            {"pbo": pbo.get("pbo"), "pbo_verdict": pbo.get("verdict"),
             "wfe": wf.get("wfe"), "wf_verdict": wf.get("verdict")},
            indent=1, default=str), encoding="utf-8")
        log.mark_done(c["symbol"], c["timeframe"], "wfcpcv",
                      counts={"pbo": pbo.get("pbo"), "wfe": wf.get("wfe")})
        logger.warning(f"[wfcpcv] {c['symbol']} {c['timeframe']}: PBO={pbo.get('pbo')} WFE={wf.get('wfe')}")


def _program_pvals(master: dict) -> list[float]:
    """p-valores PSR de usos previos del lockbox (misma ventana programática)."""
    import yaml

    p = Path(master["research_ledger"]["lockbox_registry"])
    if not p.exists():
        return []
    uses = yaml.safe_load(open(p, encoding="utf-8")) or []
    return [float(u["psr_p"]) for u in uses if u.get("psr_p") is not None]


def _pair_closed_trades(raw_trades):
    """Empareja IN→OUT por POSITION_ID para MFE/MAE (aprox H1)."""
    import pandas as pd

    if raw_trades is None or len(raw_trades) == 0:
        return pd.DataFrame()
    df = raw_trades.copy()
    if "DEAL" not in df.columns:
        return pd.DataFrame()
    ins = df[df["DEAL"].astype(str).str.upper() == "IN"]
    outs = df[df["DEAL"].astype(str).str.upper() == "OUT"]
    in_by_pos = {pid: g.sort_values("TIME_GENERATED").iloc[0]
                 for pid, g in ins.groupby("POSITION_ID")} if len(ins) else {}
    rows = []
    for _, o in outs.iterrows():
        pid = o.get("POSITION_ID")
        first = in_by_pos.get(pid)
        rows.append({
            "OPEN_TIME": first["TIME_GENERATED"] if first is not None else o.get("TIME_GENERATED"),
            "CLOSE_TIME": o.get("TIME_GENERATED"),
            "ENTRY_PRICE": first["PRICE"] if first is not None else o.get("PRICE"),
            "DIRECTION": o.get("SIGNAL_TYPE", first.get("SIGNAL_TYPE") if first is not None else ""),
        })
    return pd.DataFrame(rows)


def run_select(master: dict, combo: list[dict], run_dir: Path, run_id: str, log,
               auto_lockbox: bool = False) -> None:
    from pyeventbt.analytics.stats import _closed_net
    from pyeventbt.execution.cost_models import cell_cost_cfg, stress_cells
    from pyeventbt.execution.stress_grid import stress_verdict
    from pyeventbt.research.ledger import register_lockbox_use
    from pyeventbt.robustness.decorrelation import greedy_select, jitter_grid, jitter_verdict
    from pyeventbt.robustness.dsr import psr
    from pyeventbt.robustness.validate_synthetic import load_real_h1
    from pyeventbt.runner.master_config import config_hash
    from pyeventbt.selection.gates import apply_gates
    from pyeventbt.selection.lockbox import guard_gates_before_lockbox, lockbox_evaluate
    from pyeventbt.selection.lots import compute_lots
    from pyeventbt.selection.promote import write_promoted
    from pyeventbt.selection.sl_tp import compute_sl_tp
    import numpy as np
    import pandas as pd
    import polars as pl

    w = master["universe"]["windows"]
    is_w = (datetime.fromisoformat(str(w["IS"][0])), datetime.fromisoformat(str(w["IS"][1])))
    fwd_w = (datetime.fromisoformat(str(w["FORWARD"][0])), datetime.fromisoformat(str(w["FORWARD"][1])))
    years_is = max((is_w[1] - is_w[0]).days / 365.25, 0.1)
    tag_of = lambda c: _combo_tag(master, c)  # noqa: E731
    for c in combo:
        if log.is_done(c["symbol"], c["timeframe"], "select", config_hash(master)):
            continue
        log.init_stage(c["symbol"], c["timeframe"], "select", config_hash(master))
        tag = tag_of(c)
        base = run_dir / "optimization" / f"{c['symbol']}_{c['timeframe']}"
        trials = pl.read_parquet(base / f"{tag}_trials.parquet").to_dict(as_series=False)
        reps = _load_reps(run_dir, master, c)
        param_cols = [k for k in trials if k.startswith("param_")]
        n = len(trials["trial_number"])
        params_of = {int(trials["trial_number"][i]):
                     {k.replace("param_", ""): trials[k][i] for k in param_cols}
                     for i in range(n)}
        stats = pd.DataFrame({
            "set_id": [f"{tag}#t{trials['trial_number'][i]}" for i in range(n)],
            "trial_number": trials["trial_number"],
            "n_trades": pd.Series(trials.get("attr_n_trades", [0] * n)).fillna(0).astype(int),
            "sharpe_net": pd.Series(trials.get("attr_sharpe_net_full", [np.nan] * n), dtype=float),
            "pf": pd.Series(trials.get("attr_pf", [np.nan] * n), dtype=float),
            "r2": pd.Series(trials.get("attr_r2", [np.nan] * n), dtype=float),
            "turnover": pd.Series(trials.get("attr_n_trades", [0] * n)).fillna(0) / years_is,
        })
        daily_all = _as_pandas_daily(base / f"{tag}_daily_returns.parquet")
        for _, row in stats.iterrows():
            col = f"trial_{int(row['trial_number'])}"
            stats.loc[stats["trial_number"] == row["trial_number"], "daily_var"] = (
                float(daily_all[col].var()) if col in daily_all.columns else 0.0)
        rep_ids = {r["set_id"] for r in reps} or set(stats["set_id"].tolist())
        stats = stats[stats["set_id"].isin(rep_ids)].reset_index(drop=True)
        cols = {f"trial_{int(t)}": f"{tag}#t{int(t)}"
                for t in stats["trial_number"] if f"trial_{int(t)}" in daily_all.columns}
        daily = daily_all[[k for k in cols]].rename(columns=cols)
        sel = greedy_select(stats, daily, master["decorrelation"])
        sdir = run_dir / "decorrelation" / f"{c['symbol']}_{c['timeframe']}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "selection.json").write_text(json.dumps(sel, indent=1, default=str), encoding="utf-8")

        scfg = master["decorrelation"]["stability_check"]
        ecfg, pcfg = master["execution_realism"], master["execution_realism"]["pass"]
        gpath, ppath = sdir / "gates.csv", run_dir / "promoted.csv"
        grows = gpath.exists() and pd.read_csv(gpath).to_dict(orient="records") or []
        gdone = {g["set_id"] for g in grows}
        for sid in sel.get("selected", []):
            if sid in gdone:
                continue
            t = int(sid.split("#t")[1])
            params = params_of[t]
            base_res = _run_once_for(master, c, params, is_w, int(master["seed"]))
            summ, pnl, dly = base_res["summary"], base_res["pnl"], base_res["daily"]
            jn = jitter_grid(params, float(scfg.get("jitter_pct", 10)))
            jr = [_run_once_for(master, c, p, is_w, int(master["seed"]) + 1 + i)["summary"]
                  for i, p in enumerate(jn)]
            jit = jitter_verdict(jr, float(scfg.get("min_green", 0.8)))
            cells, sr = [], []
            for cell in stress_cells(ecfg["stress_grid"]):
                if cell.get("delay_bars", 1) != 1:
                    continue  # delay>1 no soportado en señal v1 (documentado)
                cc = cell_cost_cfg(master["costs"], cell["spread_mult"], cell["slippage_ticks_add"])
                rr = _run_once_for(master, c, params, is_w, int(master["seed"]), costs=cc)
                sr.append({"cell": cell, "sharpe_net": rr["summary"]["sharpe_net"],
                           "pf": rr["summary"]["pf"]})
            stv = stress_verdict(sr, pcfg)
            gates = apply_gates(summ, pnl, dly, master["gates"])
            row = {"set_id": sid, "trial_number": t, "params": params,
                   "sharpe_net": summ["sharpe_net"], "pf": summ["pf"],
                   "n_trades": summ["n_trades"], "jitter": jit["verdict"],
                   "worst_cell_pf": stv.get("worst_pf"), "stress": stv["verdict"],
                   "gates": gates["verdict"], "forward": "SKIPPED"}
            if gates["verdict"] == "PASS" and auto_lockbox:
                guard_gates_before_lockbox(gates["verdict"])
                fwd = _run_once_for(master, c, params, fwd_w, int(master["seed"]) + 999)
                fs = fwd["summary"]
                fd = fwd["daily"].dropna()
                T = max(len(fd), 2)
                p_val = 1.0 - psr(float(fs["sharpe_net"]), 0.0, T,
                                  float(fd.skew()), float(fd.kurt() + 3.0))
                lb = lockbox_evaluate(fs, p_val, _program_pvals(master), master["forward"])
                row["forward"] = lb["verdict"]
                row["forward_pf"] = fs["pf"]
                if lb["verdict"] == "PASS":
                    register_lockbox_use(master["research_ledger"]["lockbox_registry"], {
                        "family": master["strategy"]["name"], "symbol": c["symbol"],
                        "timeframe": c["timeframe"], "set_id": sid, "run_id": run_id,
                        "psr_p": p_val, "forward_pf": fs["pf"]})
                    lots = compute_lots(_closed_net(base_res["trades"]), master["promotion"]["lots"])
                    h1pl = load_real_h1(c["symbol"], Path(master["universe"]["data_dir"]) / f"{c['symbol']}.csv", is_w)
                    h1d = h1pl.to_dict(as_series=False)  # sin pyarrow: dict -> pandas
                    h1 = pd.DataFrame(
                        {"high": h1d["high"], "low": h1d["low"], "close": h1d["close"]},
                        index=pd.to_datetime(h1d["datetime"]))
                    st = compute_sl_tp(_pair_closed_trades(base_res["trades"]), h1,
                                       master["promotion"]["sl_tp"])
                    from pyeventbt.selection.promote import promote_row

                    magic = int(master["promotion"].get("magic_start", 11313))
                    prow = promote_row({
                        "set_id": sid, "strategy_family": master["strategy"]["name"],
                        "combo": f"{c['symbol']}_{c['timeframe']}", "params": params,
                        "gates": gates["verdict"], "lots": lots["lots"], "magic": magic,
                        "sl_pct": st.get("sl_pct"), "tp_pct": st.get("tp_pct"),
                        "forward_pf": fs["pf"], "forward_holm": lb["holm_reject"],
                        "master_hash": (run_dir / "master_hash.txt").read_text().split()[0]
                        if (run_dir / "master_hash.txt").exists() else ""})
                    existing = (ppath.exists() and pd.read_csv(ppath).to_dict(orient="records")) or []
                    if not any(e["set_id"] == sid for e in existing):
                        existing.append(prow)
                    write_promoted(existing, ppath)
            elif gates["verdict"] == "PASS":
                row["forward"] = "AWAITING_LOCKBOX"
            grows.append(row)
            pd.DataFrame(grows).to_csv(gpath, index=False)  # checkpoint por set
        log.mark_done(c["symbol"], c["timeframe"], "select",
                      counts={"selected": len(sel.get("selected", []))})
        logger.warning(f"[select] {c['symbol']} {c['timeframe']}: sel={sel.get('selected', [])}")


def _as_pandas_daily(path: Path):
    from pyeventbt.optimization.neff import _as_pandas
    import polars as pl

    return _as_pandas(pl.read_parquet(path))


def run_calibrate(master: dict, combo: list[dict], run_dir: Path, run_id: str, log) -> None:
    from pyeventbt.calibration.run import run_calibration
    from pyeventbt.runner.master_config import config_hash

    cal = master.get("calibration", {})
    w = master["universe"]["windows"]
    is_w = (datetime.fromisoformat(str(w["IS"][0])), datetime.fromisoformat(str(w["IS"][1])))
    for c in combo:
        if log.is_done(c["symbol"], c["timeframe"], "calibrate", config_hash(master)):
            continue
        log.init_stage(c["symbol"], c["timeframe"], "calibrate", config_hash(master))
        out = run_dir / "calibration" / f"{c['symbol']}_{c['timeframe']}"
        res = run_calibration(
            master, c, is_w, out,
            n_nulls=int(cal.get("null_strategies", {}).get("n", 4)),
            seeds=range(int(cal.get("planted_edge", {}).get("seeds", 2))),
            sharpe_levels=cal.get("planted_edge", {}).get("sharpe_levels", [0.3, 1.0]))
        log.mark_done(c["symbol"], c["timeframe"], "calibrate", counts={"fpr": res["fpr"]})
        logger.warning(f"[calibrate] {c['symbol']} {c['timeframe']}: {res['report']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", nargs="?", default=None)
    ap.add_argument("--only", default=",".join(STAGES))
    ap.add_argument("--pair", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--max-reps", type=int, default=None,
                    help="cap de representantes en synth (coste)")
    ap.add_argument("--auto-lockbox", action="store_true",
                    help="permite consumir el FORWARD (una vez, con Holm). Sin esto: AWAITING_LOCKBOX")
    ap.add_argument("--report", default=None, help="emite gate_report.json de RUN_DIR y sale")
    args = ap.parse_args()
    if args.report:
        from pyeventbt.robustness.report import build_report

        rep = build_report(args.report, full=True)
        out = Path(args.report) / "gate_report.json"
        out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(str(out))
        return
    if not args.master:
        ap.error("se requiere MASTER (o usa solo --report RUN_DIR)")
    from pyeventbt.runner.master_config import load_master
    from pyeventbt.runner.process_log import ProcessLog

    master = load_master(args.master)
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir(master)
    run_id = run_dir.name
    log = ProcessLog(run_dir / "process_logs.yaml")
    cb = combos(master, args.pair, args.timeframe)
    only = [s.strip() for s in args.only.split(",")]
    if "prep" in only:
        run_prep(master, cb, run_dir, run_id, log)
    if "optimize" in only:
        run_optimize(master, cb, run_dir, run_id, log)
    if "cluster" in only:
        run_cluster(master, cb, run_dir, run_id, log)
    if "synth" in only:
        run_synth(master, cb, run_dir, run_id, log, args.max_reps)
    if "wfcpcv" in only:
        run_wfcpcv(master, cb, run_dir, run_id, log)
    if "select" in only:
        run_select(master, cb, run_dir, run_id, log, args.auto_lockbox)
    if "calibrate" in only:
        run_calibrate(master, cb, run_dir, run_id, log)
    print(str(run_dir))


if __name__ == "__main__":
    main()
