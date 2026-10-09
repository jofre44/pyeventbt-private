"""E2E dispatcher completo sobre datos sinteticos minimos (10d M1).

Recorre prep -> optimize(2) -> cluster -> synth(2 sims) -> wfcpcv ->
select (sin --auto-lockbox: FORWARD intacto) -> report. ~1-2 min.
"""


def _tiny_setup(tmp_path):
    import numpy as np
    import polars as pl
    from datetime import datetime, timedelta

    dd = tmp_path / "data"
    dd.mkdir()
    n = 10 * 1440
    dts = [datetime(2021, 1, 3) + timedelta(minutes=i) for i in range(n)]
    cl = 0.98 + np.cumsum(np.random.default_rng(0).normal(0, 0.00005, n))
    pl.DataFrame({
        "date": [x.strftime("%Y.%m.%d") for x in dts],
        "time": [x.strftime("%H:%M") for x in dts],
        "open": cl, "high": cl * 1.0002, "low": cl * 0.9998, "close": cl,
        "tickvol": np.ones(n), "volume": np.ones(n), "spread": np.full(n, 20),
    }).write_csv(str(dd / "AUDCAD.csv"), include_header=False)

    from pyeventbt.runner.master_config import load_master

    cfg = load_master("pipeline_master.yaml")
    cfg["universe"]["data_dir"] = str(dd)
    cfg["universe"]["windows"] = {"IS": ["2021-01-03", "2021-01-11"],
                                  "FORWARD": ["2021-01-13", "2021-01-20"]}
    cfg["optimization"]["n_trials"] = 2
    cfg["validation"]["n_sim_screening"] = 2
    cfg["decorrelation"]["min_trades"] = 1  # 10d dan pocos trades: evaluar igual
    cfg["research_ledger"]["path"] = str(tmp_path / "ledger.parquet")
    cfg["research_ledger"]["lockbox_registry"] = str(tmp_path / "lockbox_uses.yaml")
    return cfg


def test_dispatcher_full_stages_e2e(tmp_path):
    from bots_dispatcher_optuna import (
        combos, run_cluster, run_optimize, run_prep, run_select, run_synth, run_wfcpcv,
    )
    from pyeventbt.robustness.report import build_report
    from pyeventbt.runner.process_log import ProcessLog

    cfg = _tiny_setup(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = ProcessLog(run_dir / "process_logs.yaml")
    cb = combos(cfg, "AUDCAD", "H1")
    run_prep(cfg, cb, run_dir, "t", log)
    run_optimize(cfg, cb, run_dir, "t", log)
    run_cluster(cfg, cb, run_dir, "t", log)
    run_synth(cfg, cb, run_dir, "t", log)
    run_wfcpcv(cfg, cb, run_dir, "t", log)
    run_select(cfg, cb, run_dir, "t", log, auto_lockbox=False)
    assert (run_dir / "synthetic_validation" / "bbands_breakout_AUDCAD_H1.csv").exists()
    assert (run_dir / "wf_cpcv" / "AUDCAD_H1" / "pbo.json").exists()
    assert (run_dir / "decorrelation" / "AUDCAD_H1" / "gates.csv").exists()
    # Sin --auto-lockbox el FORWARD no se toca: ni registro ni promoted
    assert not (tmp_path / "lockbox_uses.yaml").exists()
    assert not (run_dir / "promoted.csv").exists()
    rep = build_report(run_dir)
    assert rep["stages"]["00_prep"] == "done"
    # Resume: segunda pasada no re-ejecuta (skip por config_hash)
    run_select(cfg, cb, run_dir, "t", log, auto_lockbox=False)
