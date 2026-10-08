"""Registry de ciclo de vida (Transversal, gobernanza condensada).

RESEARCH → CANDIDATE → VALIDATED → APPROVED → PAPER → SHADOW → MICRO → LIVE
(+DEGRADED / QUARANTINED / RETIRED). Ningún cambio de estado sin su gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

STATES = [
    "RESEARCH", "CANDIDATE", "VALIDATED", "APPROVED",
    "PAPER", "SHADOW", "MICRO", "LIVE",
    "DEGRADED", "QUARANTINED", "RETIRED",
]

# Transiciones permitidas (grafo mínimo; RETIRED/QUARANTINED alcanzables desde casi todo)
ALLOWED = {
    "RESEARCH": ["CANDIDATE", "RETIRED"],
    "CANDIDATE": ["VALIDATED", "RETIRED"],
    "VALIDATED": ["APPROVED", "RETIRED"],
    "APPROVED": ["PAPER", "RETIRED"],
    "PAPER": ["SHADOW", "QUARANTINED", "RETIRED"],
    "SHADOW": ["MICRO", "QUARANTINED", "RETIRED"],
    "MICRO": ["LIVE", "DEGRADED", "QUARANTINED", "RETIRED"],
    "LIVE": ["DEGRADED", "QUARANTINED", "RETIRED"],
    "DEGRADED": ["PAPER", "QUARANTINED", "RETIRED"],
    "QUARANTINED": ["RESEARCH", "RETIRED"],
    "RETIRED": [],
}


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {}
        if self.path.exists():
            self._data = yaml.safe_load(open(self.path, encoding="utf-8")) or {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, sort_keys=True)

    def register(self, strategy_id: str, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("status", "RESEARCH")
        self._data[strategy_id] = record
        self.save()

    def transition(self, strategy_id: str, to: str, gate_evidence: str) -> None:
        if to not in STATES:
            raise ValueError(f"estado desconocido: {to}")
        cur = self._data.get(strategy_id, {}).get("status", "RESEARCH")
        if to not in ALLOWED.get(cur, []):
            raise RuntimeError(f"transición prohibida {cur} → {to}")
        if not gate_evidence:
            raise ValueError("toda transición exige evidencia del gate")
        self._data[strategy_id]["status"] = to
        self._data[strategy_id]["last_gate_evidence"] = gate_evidence
        self.save()
