"""Dispatcher del pipeline (Transversal): `python bots_dispatcher_optuna.py MASTER
[--only prep,optimize,cluster,synth,wfcpcv,select] [--pair X] [--timeframe Y]
[--calibrate] [--report RUN_DIR]`.

Orquesta por combo (símbolo × TF) con ProcessLog reanudable (resume por
config_hash). Sin `print` en pipeline: logging. Dashboard Rich diferido
(nota en ASSUMPTIONS, sec. Transversal).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pyeventbt.pipeline")

STAGES = ["prep", "optimize", "cluster", "synth", "wfcpcv", "select"]


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
    from pyeventbt.optimization.adapter import run_pyeventbt_once
    import yaml

    (run_dir / "master_used.yaml").write_text(
        yaml.safe_dump(master, sort_keys=True), encoding="utf-8")
    (run_dir / "master_hash.txt").write_text(
        config_hash(master) + "\n" + immutable_hash(master) + "\n", encoding="utf-8")
    (run_dir / "immutable_hash.txt").write_text(immutable_hash(master), encoding="utf-8")
    w = master["universe"]["windows"]
    is_w = (datetime.fromisoformat(str(w["IS"][0])), datetime.fromisoformat(str(w["IS"][1])))
    fwd = (datetime.fromisoformat(str(w["FORWARD"][0])), datetime.fromisoformat(str(w["FORWARD"][1])))
    # warmup = max bb lookback del espacio; holding 24 barras H1 (time-stop diario)
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
        pur = dm.check_window_purity(is_w[1], fwd[0], tf, 24, int(warmup),
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
            base_spread_points=master["costs"].get("base_spread_points", 20.0))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("master")
    ap.add_argument("--only", default=",".join(STAGES))
    ap.add_argument("--pair", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
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
        logger.warning("[synth] usa run_consistency por representante (ver robustness.validate_synthetic)")
    if "wfcpcv" in only:
        logger.warning("[wfcpcv] usa cscv_pbo/cpcv/walkforward (ver robustness.*)")
    if "select" in only:
        logger.warning("[select] usa decorrelation/gates/lockbox/promote en orden (ver selection.*)")
    print(str(run_dir))


if __name__ == "__main__":
    main()
