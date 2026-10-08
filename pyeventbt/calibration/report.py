"""Reporte de calibración (Meta-etapa M): FPR por etapa + curva de potencia.

`calibration_report.json` congela los umbrales con su evidencia; el maestro lo
referencia por `master_hash`. Re-ejecutar ante cambio de umbrales/generador/
versión mayor. CI usa calibración reducida (pocos nulos).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def fpr_by_stage(results: list[dict]) -> dict[str, dict]:
    """`results`: [{null_id, kind, data_mode, stage, passed}].
    stage en {gates, lockbox_simulado, promoted_simulado}."""
    stages: dict[str, dict] = {}
    for r in results:
        st = stages.setdefault(r["stage"], {"n": 0, "passed": 0})
        st["n"] += 1
        st["passed"] += int(bool(r["passed"]))
    for st in stages.values():
        st["fpr"] = st["passed"] / st["n"] if st["n"] else 0.0
    return stages


def power_curve(results: list[dict]) -> dict[str, dict]:
    """`results`: [{sharpe_target, seed, promoted}]."""
    by_level: dict[str, dict] = {}
    for r in results:
        lv = by_level.setdefault(str(r["sharpe_target"]), {"n": 0, "promoted": 0})
        lv["n"] += 1
        lv["promoted"] += int(bool(r["promoted"]))
    for lv in by_level.values():
        lv["power"] = lv["promoted"] / lv["n"] if lv["n"] else 0.0
    return by_level


def write_report(path: str | Path, *, fpr: dict, power: dict, thresholds: dict,
                 master_hash: str, generator_version: str,
                 targets: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    overall_fpr = max((v["fpr"] for v in fpr.values()), default=0.0)
    power_at_1 = power.get("1.0", {}).get("power", 0.0)
    rep = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "master_hash": master_hash,
        "generator_version": generator_version,
        "thresholds": thresholds,
        "fpr_by_stage": fpr,
        "power_curve": power,
        "targets": targets,
        "overall_fpr": overall_fpr,
        "power_at_sharpe_1": power_at_1,
        "verdict": ("VALID" if overall_fpr <= targets.get("max_false_positive_rate", 0.05)
                    and power_at_1 >= targets.get("min_power_at_sharpe_1", 0.70)
                    else "INVALID"),
    }
    p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return p
