"""Carga, validación y hash del maestro (Etapa 0 + Transversal).

Reglas: cambiar el maestro = nuevo run con resume:true. `costs/windows/embargo/
seed/n_trials` son inmutables dentro de un run: se hashean al inicio y el
ProcessLog aborta si cambian.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

IMMUTABLE_PATHS = (
    ("costs",),
    ("universe", "windows"),
    ("universe", "embargo"),
    ("universe", "data_dir"),
    ("seed",),
    ("optimization", "n_trials"),
)

REQUIRED_TOP = (
    "pipeline_version", "seed", "strategy", "universe", "costs",
    "optimization", "cluster", "validation",
)


def load_master(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_TOP if k not in cfg]
    if missing:
        raise ValueError(f"pipeline_master.yaml incompleto, faltan: {missing}")
    if not cfg["strategy"].get("search_space"):
        raise ValueError("strategy.search_space vacío: la estrategia debe exponer rangos")
    return cfg


def _get(cfg: dict, path: tuple) -> Any:
    node: Any = cfg
    for k in path:
        node = node[k]
    return node


def immutable_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in IMMUTABLE_PATHS:
        try:
            out[".".join(p)] = _get(cfg, p)
        except KeyError:
            out[".".join(p)] = None
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    canon = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def immutable_hash(cfg: dict[str, Any]) -> str:
    canon = json.dumps(immutable_snapshot(cfg), sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def assert_immutable_unchanged(cfg: dict[str, Any], expected_hash: str) -> None:
    h = immutable_hash(cfg)
    if h != expected_hash:
        raise RuntimeError(
            "ABORT: costs/windows/embargo/seed/n_trials cambiaron a mitad de run "
            f"(esperado {expected_hash}, actual {h}). Relanza como run nuevo con resume:true."
        )
