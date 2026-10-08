"""Test Meta-etapa M: smoke de calibración (2 nulos + edge plantado, ventana 60d)."""
from datetime import datetime, timedelta

from pyeventbt.calibration.run import run_calibration
from pyeventbt.runner.master_config import load_master


def test_calibration_smoke(tmp_path):
    master = load_master("pipeline_master.yaml")
    master["research_ledger"]["path"] = str(tmp_path / "ledger.parquet")
    combo = {"strategy_family": "bbands_breakout", "symbol": "AUDCAD", "timeframe": "H1"}
    w0 = datetime(2021, 1, 3)
    res = run_calibration(master, combo, (w0, w0 + timedelta(days=60)),
                          tmp_path / "cal", n_nulls=2, seeds=(0,), sharpe_levels=(1.0,))
    assert "gates" in res["fpr"]
    assert res["fpr"]["gates"]["n"] == 4  # 2 kinds × 2 modos × 1 seed
    assert "1.0" in res["power"]
    import json

    rep = json.loads((tmp_path / "cal" / "calibration_report.json").read_text())
    assert rep["verdict"] in ("VALID", "INVALID")
    assert rep["master_hash"]
