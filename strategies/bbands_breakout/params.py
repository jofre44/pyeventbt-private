"""Params cerrados de bbands_breakout + validación estricta (Etapa 0/1).

Regla dura: `filter_params` falla ante clave desconocida o NaN.
Solo lo declarado en `search_space` es optimizable; el resto es frozen por spec.
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BBandsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")  # clave desconocida => error

    bb_period: int = Field(default=20, ge=10, le=40)
    bb_std_dev: float = Field(default=2.5, ge=1.5, le=3.0)
    # Frozen por spec (no se optimizan; viajan en el objeto para trazabilidad)
    close_hour: int = Field(default=21, ge=0, le=23)
    close_minute: int = Field(default=0, ge=0, le=59)
    order_placement_hour: int = Field(default=8, ge=0, le=23)
    order_placement_minute: int = Field(default=0, ge=0, le=59)

    @field_validator("bb_std_dev")
    @classmethod
    def _no_nan(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("bb_std_dev debe ser finito (NaN/inf rechazado)")
        return v


OPTIMIZABLE = ("bb_period", "bb_std_dev")

# Espacio canónico (el maestro puede restringirlo, nunca ampliarlo sin nuevo run)
SEARCH_SPACE = {
    "bb_period": {"type": "int", "low": 10, "high": 40, "step": 2},
    "bb_std_dev": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.1},
}


def filter_params(raw: dict[str, Any]) -> BBandsParams:
    """Valida y coacciona un dict a BBandsParams. Lanza ValueError si hay
    claves desconocidas, tipos fuera de rango o NaN/inf."""
    if not isinstance(raw, dict):
        raise ValueError(f"params debe ser dict, no {type(raw).__name__}")
    for k, v in raw.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"param '{k}' no finito: {v!r}")
    try:
        return BBandsParams(**raw)
    except Exception as exc:  # pydantic ValidationError u otros
        raise ValueError(f"params inválidos: {exc}") from exc


def params_hash(params: BBandsParams | dict[str, Any]) -> str:
    import hashlib
    import json

    d = params.model_dump() if isinstance(params, BBandsParams) else dict(params)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
