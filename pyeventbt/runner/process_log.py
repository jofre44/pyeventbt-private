"""Stage base + ProcessLog atómico (Transversal).

`ProcessLog` persiste por (símbolo, TF, etapa): pending|running|done|error + counts.
Escritura atómica tmp+os.replace. Re-lanzar salta `done` cuyo config_hash coincide.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger("pyeventbt.pipeline")


@dataclass
class StageContext:
    config: dict[str, Any]
    out_dir: Path
    run_id: str
    config_hash: str
    extra: dict[str, Any] = field(default_factory=dict)


class Stage:
    name: str = "base"

    def discover_inputs(self, ctx: StageContext) -> list[dict[str, Any]]:
        raise NotImplementedError

    def run(self, combo: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
        raise NotImplementedError

    def persist(self, result: dict[str, Any], combo: dict[str, Any], ctx: StageContext) -> None:
        raise NotImplementedError


class ProcessLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.state: dict[str, Any] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.state = yaml.safe_load(f) or {}

    @staticmethod
    def key(symbol: str, timeframe: str, stage: str) -> str:
        return f"{symbol}|{timeframe}|{stage}"

    def _atomic_write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.state, f, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def init_stage(self, symbol: str, timeframe: str, stage: str, config_hash: str) -> None:
        k = self.key(symbol, timeframe, stage)
        self.state[k] = {"status": "running", "config_hash": config_hash, "counts": {}}
        self._atomic_write()

    def mark_done(self, symbol: str, timeframe: str, stage: str, counts: dict | None = None) -> None:
        k = self.key(symbol, timeframe, stage)
        entry = self.state.get(k, {})
        entry.update({"status": "done", "counts": counts or {}})
        self.state[k] = entry
        self._atomic_write()

    def mark_error(self, symbol: str, timeframe: str, stage: str, error: str) -> None:
        k = self.key(symbol, timeframe, stage)
        entry = self.state.get(k, {})
        entry.update({"status": "error", "error": error})
        self.state[k] = entry
        self._atomic_write()

    def is_done(self, symbol: str, timeframe: str, stage: str, config_hash: str) -> bool:
        e = self.state.get(self.key(symbol, timeframe, stage))
        return bool(e and e.get("status") == "done" and e.get("config_hash") == config_hash)


def run_stage(
    stage: Stage,
    combos: list[dict[str, Any]],
    ctx: StageContext,
    log: ProcessLog,
    on_combo: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results = []
    for combo in combos:
        sym, tf = combo["symbol"], combo["timeframe"]
        if log.is_done(sym, tf, stage.name, ctx.config_hash):
            logger.warning(f"[{stage.name}] {sym} {tf}: skip (done, mismo config_hash)")
            continue
        log.init_stage(sym, tf, stage.name, ctx.config_hash)
        try:
            res = (on_combo or stage.run)(combo, ctx)
            stage.persist(res, combo, ctx)
            log.mark_done(sym, tf, stage.name, counts={"ok": 1})
            results.append(res)
        except Exception as exc:
            log.mark_error(sym, tf, stage.name, str(exc))
            raise
    return results
