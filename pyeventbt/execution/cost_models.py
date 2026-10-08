"""Modelos de coste explícitos (Etapa 4.6).

El simulador aplica spread (override) + comisión/swap internos. El slippage por
tick NO es parámetro nativo: se modela como spread aditivo conservador
(1 tick ≈ 1 punto en FX 5 dígitos). `delay_bars>1` no lo soporta la señal v1
(time_generated = barra+1fija): se registra como N/A con supuesto documentado.
"""
from __future__ import annotations


def cell_cost_cfg(base_costs: dict, spread_mult: float, slippage_ticks_add: int) -> dict:
    base_spread = float(base_costs.get("base_spread_points", 20.0))
    stressed = base_spread * float(spread_mult) + float(slippage_ticks_add)
    cfg = dict(base_costs)
    cfg["base_spread_points"] = stressed
    cfg["spread_mult_applied"] = spread_mult
    cfg["slippage_ticks_add_applied"] = slippage_ticks_add
    return cfg


def stress_cells(stress_cfg: dict) -> list[dict]:
    cells = []
    for sm in stress_cfg.get("spread_mult", [1.0]):
        for sa in stress_cfg.get("slippage_ticks_add", [0]):
            for db in stress_cfg.get("delay_bars", [1]):
                cells.append({"spread_mult": sm, "slippage_ticks_add": sa, "delay_bars": db,
                              "delay_supported": db == 1})
    return cells
