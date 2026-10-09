"""Tests Transversal: registry de ciclo de vida + approval package + dispatcher."""
import pytest

from pyeventbt.research.registry import ALLOWED, Registry, build_approval_package


def test_registry_transitions(tmp_path):
    reg = Registry(tmp_path / "registry.yaml")
    reg.register("bb1", {"spec_hash": "abc"})
    assert reg._data["bb1"]["status"] == "RESEARCH"
    reg.transition("bb1", "CANDIDATE", "gates PASS")
    assert reg._data["bb1"]["status"] == "CANDIDATE"
    with pytest.raises(RuntimeError):
        reg.transition("bb1", "LIVE", "salto")  # transición prohibida
    with pytest.raises(ValueError):
        reg.transition("bb1", "VALIDATED", "")  # sin evidencia no hay transición


def test_approval_package(tmp_path):
    run = tmp_path / "run"
    (run / "00_prep").mkdir(parents=True)
    (run / "00_prep" / "power.json").write_text("{}")
    (run / "promoted.csv").write_text("a,b\n")
    out = build_approval_package(run, tmp_path / "approval")
    import json

    hashes = json.loads((out / "hashes.json").read_text())
    assert len(hashes) == 2 and all("sha256_16" in h for h in hashes)


def test_dispatcher_prep_outputs(tmp_path):
    """Prep punta a punta con CSV mínimo (sin tocar E:/Data)."""
    import numpy as np
    import polars as pl
    import yaml
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

    from bots_dispatcher_optuna import combos, run_prep
    from pyeventbt.runner.master_config import load_master
    from pyeventbt.runner.process_log import ProcessLog

    cfg = load_master("pipeline_master.yaml")
    cfg["universe"]["data_dir"] = str(dd)
    cfg["universe"]["windows"] = {"IS": ["2021-01-03", "2021-01-11"],
                                  "FORWARD": ["2021-01-13", "2021-01-20"]}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = ProcessLog(run_dir / "process_logs.yaml")
    run_prep(cfg, combos(cfg, "AUDCAD", "H1"), run_dir, "t", log)
    assert (run_dir / "00_prep" / "budget.json").exists()
    assert (run_dir / "00_prep" / "power.json").exists()
    assert (run_dir / "master_used.yaml").exists()
    import json

    budget = json.loads((run_dir / "00_prep" / "budget.json").read_text())
    assert budget["n_dim"] == 2 and "seed_derivation" in budget


def test_process_log_nan_counts_survive_reload(tmp_path):
    """NaN/inf en counts se guardan null: el safe_load posterior (con el
    constructor Decimal global del core) no debe reventar."""
    import yaml

    import pyeventbt.broker.mt5_broker.shared.shared_data  # noqa: F401 (registra constructor)
    from pyeventbt.runner.process_log import ProcessLog

    log = ProcessLog(tmp_path / "process_logs.yaml")
    log.init_stage("X", "H1", "wfcpcv", "abc")
    log.mark_done("X", "H1", "wfcpcv", counts={"pbo": None, "wfe": float("nan")})
    state = yaml.safe_load((tmp_path / "process_logs.yaml").read_text(encoding="utf-8"))
    assert state["X|H1|wfcpcv"]["counts"] == {"pbo": None, "wfe": None}
    assert log.is_done("X", "H1", "wfcpcv", "abc")
