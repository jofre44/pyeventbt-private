"""FORWARD lockbox: una vez, sin reintentos, con Holm (Etapa 4.8).

Solo los que PASARON puertas ven el FORWARD. Criterio: pf≥min_pf y
sharpe_net>0. Multiplicidad: PSR corregido por Holm sobre los usos del
programa (misma ventana). Cada uso se registra; segundo uso = run contaminado.
"""
from __future__ import annotations

import numpy as np


def holm(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Rechazos Holm (FWER). Devuelve máscara en orden original."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (m - rank):
            rejected[i] = True
        else:
            break
    return rejected


def lockbox_evaluate(forward_summary: dict, forward_psr_p: float,
                     program_pvals: list[float], forward_cfg: dict,
                     alpha: float = 0.05) -> dict:
    pf = forward_summary.get("pf") or 0.0
    sr = forward_summary.get("sharpe_net")
    op_ok = pf >= forward_cfg.get("min_pf", 1.1) and (sr is not None and sr > 0)
    rej = holm(program_pvals + [forward_psr_p], alpha)
    holm_ok = bool(rej[-1])
    ok = bool(op_ok and holm_ok)
    return {"verdict": "PASS" if ok else "FAIL",
            "op_filter": bool(op_ok), "holm_reject": holm_ok,
            "forward_p": forward_psr_p, "n_program_uses": len(program_pvals)}


def guard_gates_before_lockbox(gates_verdict: str) -> None:
    """Puertas después del lockbox = run inválido. Esta guarda revienta si se
    invierte el orden."""
    if gates_verdict != "PASS":
        raise RuntimeError(
            f"ORDEN INVÁLIDO: lockbox exige puertas=PASS (recibido {gates_verdict}). "
            "Puertas van ANTES del lockbox; un set que falla puertas nunca ve FORWARD."
        )
