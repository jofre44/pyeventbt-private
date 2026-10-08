"""Límites de riesgo y kill switches (Etapa 6.4): jerarquía por trade/día/
estrategia/cartera + operacionales. Acciones: ok | halt | flatten + kill
manual. Capa INDEPENDIENTE de la estrategia: no se confía en que se autolimite.
"""
from __future__ import annotations

import time


class KillSwitch:
    def __init__(self):
        self.engaged = False
        self.reason: str | None = None

    def engage(self, reason: str) -> None:
        self.engaged = True
        self.reason = reason

    def reset(self) -> None:
        self.engaged = False
        self.reason = None


def evaluate_limits(state: dict, limits: dict, kill: KillSwitch,
                    last_heartbeat_ts: float | None = None) -> dict:
    """`state`: {trade_risk_pct, daily_loss_pct, strategy_dd_pct,
    portfolio_dd_pct, spread_points, data_stale_s}. Devuelve acción."""
    if kill.engaged:
        return {"action": "flatten", "reason": f"kill manual: {kill.reason}"}
    now = time.time()
    if (last_heartbeat_ts is not None
            and now - last_heartbeat_ts > limits.get("heartbeat_seconds", 30) * 4):
        return {"action": "halt", "reason": "heartbeat perdido"}
    if state.get("data_stale_s", 0) > limits.get("stale_data_seconds", 120):
        return {"action": "halt", "reason": "datos stale"}
    if state.get("spread_points", 0) > limits.get("max_spread_points", 30):
        return {"action": "halt", "reason": "spread máximo excedido"}
    checks = [
        ("trade_risk_pct", "per_trade_risk_pct", "flatten"),
        ("daily_loss_pct", "daily_loss_pct", "halt"),
        ("strategy_dd_pct", "strategy_dd_pct", "halt"),
        ("portfolio_dd_pct", "portfolio_dd_pct", "flatten"),
    ]
    worst = {"action": "ok", "reason": None}
    for s_key, lim_key, action in checks:
        if state.get(s_key, 0) > limits.get(lim_key, 1e9):
            if action == "flatten" or worst["action"] == "ok":
                worst = {"action": action, "reason": f"{s_key} > {lim_key}"}
    return worst
