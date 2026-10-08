"""Reporte agregado de gates (Transversal): `python -m pyeventbt.robustness.report
RUN_DIR [--full]`. Recorre artefactos del run y emite gate_report.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_report(run_dir: str | Path, full: bool = False) -> dict:
    run_dir = Path(run_dir)
    rep: dict = {"run": run_dir.name, "stages": {}}
    plog = run_dir / "process_logs.yaml"
    if plog.exists():
        import yaml

        rep["process_log"] = yaml.safe_load(plog.read_text(encoding="utf-8"))
    for stage_dir in ("00_prep", "optimization", "cluster", "synthetic_validation",
                      "wf_cpcv", "decorrelation", "portfolio"):
        d = run_dir / stage_dir
        rep["stages"][stage_dir] = "done" if d.exists() else "missing"
    for f in ("master_hash.txt", "promoted.csv"):
        rep[f] = "present" if (run_dir / f).exists() else "missing"
    if full:
        rep["files"] = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    rep = build_report(args.run_dir, args.full)
    out = Path(args.run_dir) / "gate_report.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
