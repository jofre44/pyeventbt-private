"""Despliegue escalonado (Etapa 6.2): paper → micro → scaled → full.

Avance solo con: paridad OK + Sharpe/DD dentro de envolvente + cero
incidentes. Champion/challenger para versiones. Nunca params en caliente
(cambiar params = re-opt = run nuevo).
"""
from __future__ import annotations


def stage_gate(stage: dict, perf: dict) -> dict:
    """`perf`: {days, trades, parity_ok, sharpe_in_envelope, dd_in_envelope,
    incidents}. Devuelve si puede avanzar."""
    reasons = []
    if perf.get("days", 0) < stage.get("min_days", 0):
        reasons.append("min_days")
    if perf.get("trades", 0) < stage.get("min_trades", 0):
        reasons.append("min_trades")
    for k in ("parity_ok", "sharpe_in_envelope", "dd_in_envelope"):
        if not perf.get(k, False):
            reasons.append(k)
    if perf.get("incidents", 0) > 0:
        reasons.append("incidents")
    return {"advance": not reasons, "blockers": reasons}


def rollout_plan(stages: list[dict], current: str) -> dict:
    names = [s["name"] for s in stages]
    i = names.index(current) if current in names else -1
    nxt = stages[i + 1] if 0 <= i + 1 < len(stages) else None
    return {"current": current, "next": (nxt["name"] if nxt else None),
            "next_lot_mult": (nxt["lot_mult"] if nxt else None)}
