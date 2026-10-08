"""Runner de calibración (Meta-etapa M): `python -m pyeventbt.calibration.run MASTER`.

Modo reducido (CI): --smoke con pocos nulos/semillas y ventana corta.
Modo completo: nulos del maestro × data_modes + edge plantado × niveles.
Cada nulo: backtest → gates (puertas) como proxy de promoción; el FPR real a
nivel `promoted.csv` exige el pipeline completo y se anota como extensión.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("pyeventbt.pipeline")


def run_calibration(master: dict[str, Any], combo: dict[str, Any],
                    window: tuple[datetime, datetime], out_dir: str | Path,
                    *, n_nulls: int = 4, seeds=(0, 1),
                    sharpe_levels=(0.3, 1.0)) -> dict[str, Any]:
    from pyeventbt.calibration.null_strategies import NULL_FACTORIES, NULL_PARAMS_CLASS
    from pyeventbt.calibration.planted_edge import plant_drift_m1
    from pyeventbt.calibration.report import fpr_by_stage, power_curve, write_report
    from pyeventbt.optimization.adapter import run_pyeventbt_once
    from pyeventbt.robustness.synthetic import block_permuted_closes, iid_null_closes
    from pyeventbt.robustness.validate_synthetic import load_real_m1
    from pyeventbt.runner.master_config import config_hash
    from pyeventbt.selection.gates import apply_gates
    import polars as pl

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    symbol = combo["symbol"]

    def _kw(factory, params_cls):
        return dict(costs=master["costs"], cash=master["strategy"]["cash"],
                    factory_path=factory, params_class_path=params_cls,
                    data_dir=master["universe"]["data_dir"],
                    strategy_id=master["strategy"].get("strategy_id", "11313"),
                    account_currency=master["strategy"].get("account_currency", "USD"),
                    aux_fallback_rates=master["universe"].get("aux_fallback_rates", {}),
                    base_spread_points=master["costs"].get("base_spread_points", 20.0))

    m1 = load_real_m1(symbol, Path(master["universe"]["data_dir"]) / f"{symbol}.csv", window)
    vol_d = float(m1.select((pl.col("close").log().diff() * 100).std()).item() or 0.5) / 100 * (1440 ** 0.5)
    gate_results: list[dict] = []
    kinds = list(NULL_FACTORIES)[:n_nulls] if n_nulls <= 3 else list(NULL_FACTORIES)
    for kind in kinds:
        for mode in ("iid_synthetic", "real_block_permuted"):
            for s in seeds:
                closes = m1["close"].to_numpy()
                sim_c = (iid_null_closes(closes, seed=100 + s) if mode == "iid_synthetic"
                         else block_permuted_closes(closes, block=10, seed=100 + s))
                bars = m1.with_columns([
                    pl.Series("close", sim_c),
                    (pl.Series("close", sim_c) * 1.0002).alias("high"),
                    (pl.Series("close", sim_c) * 0.9998).alias("low"),
                    pl.Series("open", sim_c),
                ])
                try:
                    r = run_pyeventbt_once(
                        {"seed": s, "kind": kind}, combo, window=window, seed=s,
                        mode="null", bars_df=bars,
                        **_kw(NULL_FACTORIES[kind], NULL_PARAMS_CLASS))
                    g = apply_gates(r["summary"], r["pnl"], r["daily"], master["gates"])
                    gate_results.append({"null_id": f"{kind}/{mode}/{s}", "kind": kind,
                                         "data_mode": mode, "stage": "gates",
                                         "passed": g["verdict"] == "PASS"})
                except Exception as exc:
                    gate_results.append({"null_id": f"{kind}/{mode}/{s}", "kind": kind,
                                         "data_mode": mode, "stage": "gates",
                                         "passed": False, "error": str(exc)[:100]})

    power_results: list[dict] = []
    for lv in sharpe_levels:
        for s in seeds:
            # Drift en TODOS los días (baseline de tendencia): la estrategia
            # nula opera a diario; plantarlo en 1/5 diluiría 5× y mediría otra cosa.
            dates = sorted({d.date() for d in m1["datetime"].to_list()})
            bars = plant_drift_m1(m1, set(dates), lv, vol_d, seed=s)
            try:
                r = run_pyeventbt_once(
                    {"seed": s, "kind": "drift_rider"}, combo, window=window, seed=s,
                    mode="synthetic", bars_df=bars,
                    **_kw(NULL_FACTORIES["drift_rider"], NULL_PARAMS_CLASS))
                g = apply_gates(r["summary"], r["pnl"], r["daily"], master["gates"])
                power_results.append({"sharpe_target": lv, "seed": s,
                                      "promoted": g["verdict"] == "PASS"})
            except Exception:
                power_results.append({"sharpe_target": lv, "seed": s, "promoted": False})

    fpr = fpr_by_stage(gate_results)
    power = power_curve(power_results)
    rep = write_report(
        out / "calibration_report.json", fpr=fpr, power=power,
        thresholds={"gates": master["gates"], "validation": master["validation"]},
        master_hash=config_hash(master), generator_version="gjr-garch-t-m1/v1",
        targets=master["calibration"]["targets"])
    return {"fpr": fpr, "power": power, "report": str(rep)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("master", help="pipeline_master.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pair", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from pyeventbt.runner.master_config import load_master

    master = load_master(args.master)
    sym = args.pair or master["universe"]["symbols"][0]
    tf = args.timeframe or master["universe"]["timeframes"][0]
    combo = {"strategy_family": master["strategy"]["name"], "symbol": sym, "timeframe": tf}
    if args.smoke:
        w = master["universe"]["windows"]["IS"]
        import datetime as _dt

        window = (_dt.datetime.fromisoformat(str(w[0])),
                  min(_dt.datetime.fromisoformat(str(w[1])),
                      _dt.datetime.fromisoformat(str(w[0])) + _dt.timedelta(days=60)))
        res = run_calibration(master, combo, window, args.out or "runs/calibration_smoke",
                              n_nulls=2, seeds=(0,), sharpe_levels=(1.0,))
    else:
        w = master["universe"]["windows"]["IS"]
        import datetime as _dt

        window = (_dt.datetime.fromisoformat(str(w[0])), _dt.datetime.fromisoformat(str(w[1])))
        cal = master["calibration"]
        res = run_calibration(
            master, combo, window, args.out or "runs/calibration",
            n_nulls=cal["null_strategies"]["n"], seeds=range(3),
            sharpe_levels=cal["planted_edge"]["sharpe_levels"])
    print(res["report"])


if __name__ == "__main__":
    main()
