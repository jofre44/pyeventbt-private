"""PyEventBT pipeline — thin builder/facade over the Stage-based pipeline.

Equivalente al ``bots_dispatcher_optimizer_synthetic.BotOptimizer`` de DMRI_PyMT4:
el trabajo pesado vive en ``bots_dispatcher_optuna.run_*``. Este modulo es el
punto de entrada y la API publica: carga el maestro, crea el ``ProcessLog``,
construye el :class:`PipelineContext`, instancia las siete etapas y las expone
como ``list[Stage]`` para el runner. Metodos de compatibilidad (``optimize``,
``cluster``, etc.) delegan en las instancias para que el codigo existente siga
funcionando.

Orden canonico (Etapa 4 garantizada por construccion: puertas -> lockbox ->
sizing dentro de ``run_select``; el lockbox exige ``auto_lockbox``):

    prep -> optimize -> cluster -> synth -> wfcpcv -> select -> calibrate(M, opt-in)

Diferencia con DMRI: el runner itera por ETAPA sobre todos los combos
(stage-major), no por combo sobre todas las etapas. PBO/walkforward necesitan
la matriz completa de trials antes de evaluar supervivientes.

Resumen del proceso: :meth:`PipelineOptimizer.summary` (dict) y
``--summary`` (JSON por stdout). Sin ``print`` en pipeline salvo salida CLI
(run_dir / JSON), igual que el dispatcher.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bots_dispatcher_optuna as dispatcher
from pyeventbt.runner.master_config import config_hash, load_master
from pyeventbt.runner.process_log import ProcessLog

logger = logging.getLogger("pyeventbt.pipeline")

__all__ = [
    "PipelineOptimizer",
    "PipelineContext",
    "Stage",
    "PrepStage",
    "OptimizeStage",
    "ClusterStage",
    "SynthStage",
    "WfCpcvStage",
    "SelectStage",
    "CalibrateStage",
    "run_pipeline",
    "summarize",
]

STAGE_IDS: list[str] = list(dispatcher.STAGES)


@dataclass
class PipelineContext:
    """Contenedor de inyeccion compartido por todas las etapas."""

    master: dict[str, Any]
    run_dir: Path
    run_id: str
    log: ProcessLog
    dashboard: Any | None = None
    max_reps: int | None = None
    auto_lockbox: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """Base abstracta de etapa. Las subclases fijan ``name`` (= id del ProcessLog)."""

    name: str = "unnamed"

    def __init__(self, ctx: PipelineContext):
        self.ctx = ctx

    @property
    def master(self) -> dict[str, Any]:
        return self.ctx.master

    @property
    def run_dir(self) -> Path:
        return self.ctx.run_dir

    @property
    def log(self) -> ProcessLog:
        return self.ctx.log

    @property
    def dashboard(self):
        return self.ctx.dashboard

    def _update_state(self, **kwargs):
        if self.ctx.dashboard is not None:
            self.ctx.dashboard.update_state(**kwargs)

    @abstractmethod
    def run(self, pair: str | None = None, timeframe: str | None = None) -> None:
        """Ejecuta la etapa sobre los combos filtrados (None = todos)."""

    def discover_inputs(self, pair: str | None = None, timeframe: str | None = None) -> list[str]:
        return []

    def persist(self, pair: str, timeframe: str, **counts) -> None:
        self.log.mark_done(pair, timeframe, self.name, counts=counts or None)


def _combos(master: dict, pair: str | None, timeframe: str | None) -> list[dict]:
    return dispatcher.combos(master, pair, timeframe)


class PrepStage(Stage):
    name = "prep"

    def run(self, pair=None, timeframe=None) -> None:
        self._update_state(step=self.name)
        dispatcher.run_prep(self.master, _combos(self.master, pair, timeframe),
                            self.run_dir, self.ctx.run_id, self.log)


class OptimizeStage(Stage):
    name = "optimize"

    def run(self, pair=None, timeframe=None) -> None:
        self._update_state(step=self.name)
        dispatcher.run_optimize(self.master, _combos(self.master, pair, timeframe),
                                self.run_dir, self.ctx.run_id, self.log)


class ClusterStage(Stage):
    name = "cluster"

    def run(self, pair=None, timeframe=None) -> None:
        self._update_state(step=self.name)
        dispatcher.run_cluster(self.master, _combos(self.master, pair, timeframe),
                               self.run_dir, self.ctx.run_id, self.log)


class SynthStage(Stage):
    name = "synth"

    def run(self, pair=None, timeframe=None) -> None:
        self._update_state(step=self.name)
        dispatcher.run_synth(self.master, _combos(self.master, pair, timeframe),
                             self.run_dir, self.ctx.run_id, self.log,
                             max_reps=self.ctx.max_reps)


class WfCpcvStage(Stage):
    name = "wfcpcv"

    def run(self, pair=None, timeframe=None) -> None:
        self._update_state(step=self.name)
        dispatcher.run_wfcpcv(self.master, _combos(self.master, pair, timeframe),
                              self.run_dir, self.ctx.run_id, self.log)


class SelectStage(Stage):
    name = "select"

    def run(self, pair=None, timeframe=None) -> None:
        self._update_state(step=self.name)
        dispatcher.run_select(self.master, _combos(self.master, pair, timeframe),
                              self.run_dir, self.ctx.run_id, self.log,
                              auto_lockbox=self.ctx.auto_lockbox)


class CalibrateStage(Stage):
    name = "calibrate"

    def run(self, pair=None, timeframe=None) -> None:
        self._update_state(step=self.name)
        dispatcher.run_calibrate(self.master, _combos(self.master, pair, timeframe),
                                 self.run_dir, self.ctx.run_id, self.log)


_STAGE_CLASSES: dict[str, type[Stage]] = {
    "prep": PrepStage,
    "optimize": OptimizeStage,
    "cluster": ClusterStage,
    "synth": SynthStage,
    "wfcpcv": WfCpcvStage,
    "select": SelectStage,
    "calibrate": CalibrateStage,
}


def summarize(run_dir: str | Path) -> dict[str, Any]:
    """Resumen del proceso para un run existente: estados + artefactos.

    No necesita el maestro: lee ``process_logs.yaml`` y comprueba los
    artefactos esperados por etapa. ``promoted``/``lockbox`` se reportan si existen.
    """
    rd = Path(run_dir)
    out: dict[str, Any] = {"run_dir": str(rd), "run_id": rd.name,
                           "stages": {}, "artifacts": {},
                           "promoted_rows": None, "gate_report": None}
    log_path = rd / "process_logs.yaml"
    if log_path.exists():
        import yaml

        state = yaml.safe_load(log_path.read_text(encoding="utf-8")) or {}
        per_stage: dict[str, dict[str, int]] = {}
        for key, entry in state.items():
            try:
                _, _, stage = key.split("|")
            except ValueError:
                continue
            per_stage.setdefault(stage, {}).setdefault(entry.get("status", "?"), 0)
            per_stage[stage][entry.get("status", "?")] += 1
        out["stages"] = per_stage
    # Contenido real, no solo carpetas (el wrapper las pre-crea vacias).
    markers = {
        "00_prep": ["00_prep/power.json"],
        "optimization": ["optimization/*/*_trials.parquet"],
        "cluster": ["cluster/*/*.csv"],
        "synthetic_validation": ["synthetic_validation/*.csv"],
        "wf_cpcv": ["wf_cpcv/*/pbo.json"],
        "decorrelation": ["decorrelation/*/gates.csv"],
        "calibration": ["calibration/*/calibration_report.json"],
    }
    import glob as _glob

    for stage_dir, patterns in markers.items():
        hit = any(_glob.glob(str(rd / p)) for p in patterns)
        out["artifacts"][stage_dir] = "done" if hit else "missing"
    prom = rd / "promoted.csv"
    if prom.exists():
        try:
            import pandas as pd

            out["promoted_rows"] = len(pd.read_csv(prom))
        except Exception:
            out["promoted_rows"] = "unreadable"
    gate = rd / "gate_report.json"
    if gate.exists():
        out["gate_report"] = "done"
    mh = rd / "master_hash.txt"
    if mh.exists():
        out["master_hash"] = mh.read_text(encoding="utf-8").split()[0]
    return out


class PipelineOptimizer:
    """Builder + facade del pipeline por etapas.

    Construye con la ruta al maestro; expone ``self.stages: list[Stage]``
    para que el runner itere. Los metodos de compatibilidad delegan en la
    etapa correspondiente.
    """

    def __init__(self, master_path: str | Path, run_dir: str | Path | None = None,
                 max_reps: int | None = None, auto_lockbox: bool = False):
        self._master_path = str(master_path)
        self._master = self.load_config(self._master_path)
        self._run_dir = Path(run_dir) if run_dir else dispatcher.new_run_dir(self._master)
        self._run_id = self._run_dir.name
        self._state = self.create_log_file()
        self.ensure_stage_folders()
        self._dashboard = None
        self._ctx = PipelineContext(
            master=self._master, run_dir=self._run_dir, run_id=self._run_id,
            log=self._state, dashboard=None,
            max_reps=max_reps, auto_lockbox=auto_lockbox)
        self._stages: list[Stage] = self.build_stages()

    def build_stages(self) -> list[Stage]:
        """Construye las siete etapas en orden canonico."""
        return [cls(self._ctx) for cls in
                (PrepStage, OptimizeStage, ClusterStage, SynthStage,
                 WfCpcvStage, SelectStage, CalibrateStage)]

    @property
    def stages(self) -> list[Stage]:
        return self._stages

    @property
    def dashboard(self):
        return self._dashboard

    @dashboard.setter
    def dashboard(self, value):
        self._dashboard = value
        self._ctx.dashboard = value

    @property
    def master(self) -> dict[str, Any]:
        return self._master

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    # ------------------------------------------------------------------
    # Config / state setup
    # ------------------------------------------------------------------
    def load_config(self, master_path: str) -> dict[str, Any]:
        cfg = load_master(master_path)
        logger.warning(f"Configuration loaded from {master_path}.")
        return cfg

    def create_log_file(self) -> ProcessLog:
        return ProcessLog(self._run_dir / "process_logs.yaml")

    def ensure_stage_folders(self) -> None:
        for folder in ("00_prep/manifests", "optimization", "cluster",
                       "synthetic_validation", "wf_cpcv", "decorrelation",
                       "calibration"):
            (self._run_dir / folder).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Symbol / timeframe iteration
    # ------------------------------------------------------------------
    def list_symbols(self) -> list[str]:
        return list(self._master["universe"]["symbols"])

    def list_timeframes(self) -> list[str]:
        return list(self._master["universe"]["timeframes"])

    def combos(self, pair: str | None = None, timeframe: str | None = None) -> list[dict]:
        return _combos(self._master, pair, timeframe)

    # ------------------------------------------------------------------
    # Backward-compatible stage delegates
    # ------------------------------------------------------------------
    def _update_state(self, **kwargs):
        if self._dashboard is not None:
            self._dashboard.update_state(**kwargs)

    def _run_stage(self, stage_id: str, pair=None, timeframe=None) -> None:
        for s in self._stages:
            if s.name == stage_id:
                s.run(pair=pair, timeframe=timeframe)
                return
        raise KeyError(f"etapa desconocida: {stage_id}")

    def prep(self, pair=None, timeframe=None) -> None:
        self._run_stage("prep", pair, timeframe)

    def optimize_parameters(self, pair=None, timeframe=None) -> None:
        self._run_stage("optimize", pair, timeframe)

    def cluster_sets_selection(self, pair=None, timeframe=None) -> None:
        self._run_stage("cluster", pair, timeframe)

    def validate_on_synthetic(self, pair=None, timeframe=None) -> None:
        self._run_stage("synth", pair, timeframe)

    def wfcpcv(self, pair=None, timeframe=None) -> None:
        self._run_stage("wfcpcv", pair, timeframe)

    def decorrelation_selection(self, pair=None, timeframe=None) -> None:
        self._run_stage("select", pair, timeframe)

    def calibrate(self, pair=None, timeframe=None) -> None:
        self._run_stage("calibrate", pair, timeframe)

    def single_backtest(self, params: dict, window: tuple,
                        pair: str, timeframe: str) -> list[dict]:
        """Atajo diagnostico (no toca ledger/lockbox)."""
        cb = self.combos(pair, timeframe)
        return dispatcher.run_single_backtests(self._master, cb, params, window)

    # ------------------------------------------------------------------
    # Driver + resumen
    # ------------------------------------------------------------------
    def run(self, only: list[str] | None = None,
            pair: str | None = None, timeframe: str | None = None) -> Path:
        only = only or list(dispatcher.DEFAULT_STAGES)
        unknown = [s for s in only if s not in _STAGE_CLASSES]
        if unknown:
            raise ValueError(f"etapas desconocidas: {unknown} (validas: {STAGE_IDS})")
        n = len(self.combos(pair, timeframe)) or 1
        logger.warning(f"[pipeline] run_dir={self._run_dir} stages={only} combos={n}")
        for s in self._stages:
            if s.name in only:
                self._update_state(step=s.name)
                s.run(pair=pair, timeframe=timeframe)
        self._update_state(step="Completed")
        return self._run_dir

    def summary(self) -> dict[str, Any]:
        rep = summarize(self._run_dir)
        rep["config_hash"] = config_hash(self._master)
        return rep


def run_pipeline(master_path: str, only: list[str] | None = None,
                 pair: str | None = None, timeframe: str | None = None,
                 run_dir: str | None = None, max_reps: int | None = None,
                 auto_lockbox: bool = False) -> int:
    """Ejecuta el pipeline stage-major. Exit codes estilo POSIX (0/1/130)."""
    try:
        opt = PipelineOptimizer(master_path, run_dir=run_dir,
                                max_reps=max_reps, auto_lockbox=auto_lockbox)
        only = only or list(dispatcher.DEFAULT_STAGES)
        total = len(opt.combos(pair, timeframe))
        logger.warning(f"=== Pipeline {opt.master['strategy']['name']}: "
                       f"{total} combos x {only} ===")
        opt.run(only=only, pair=pair, timeframe=timeframe)
        rep = opt.summary()
        logger.warning(f"[summary] {rep['stages']} artifacts={rep['artifacts']} "
                       f"promoted={rep['promoted_rows']}")
        try:
            from pyeventbt.robustness.report_html import build_html_report

            html_out = build_html_report(opt.run_dir)
            logger.warning(f"[report_html] {html_out}")
        except Exception as exc:
            logger.warning(f"[report_html] omitido: {exc}")
        print(str(opt.run_dir))
        return 0
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted")
        return 130
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Wrapper/resumen del pipeline PyEventBT")
    ap.add_argument("master", nargs="?", default=None)
    ap.add_argument("--only", default=",".join(dispatcher.DEFAULT_STAGES),
                    help=f"etapas (default: {','.join(dispatcher.DEFAULT_STAGES)}; "
                         f"completo: {','.join(STAGE_IDS)})")
    ap.add_argument("--pair", default=None)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--max-reps", type=int, default=None)
    ap.add_argument("--auto-lockbox", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--params", default=None)
    ap.add_argument("--window", default="IS")
    ap.add_argument("--report", default=None, help="emite gate_report.json de RUN_DIR y sale")
    ap.add_argument("--report-html", default=None,
                    help="emite report.html offline de RUN_DIR y sale")
    ap.add_argument("--topN", type=int, default=15,
                    help="fichas expandidas en report.html (default: 15)")
    ap.add_argument("--summary", default=None, nargs="?", const="",
                    help="resumen del proceso: pasa RUN_DIR o usa --run-dir; "
                         "sin valor y sin --run-dir resume tras correr")
    args = ap.parse_args()

    if args.report:
        from pyeventbt.robustness.report import build_report

        rep = build_report(args.report, full=True)
        out = Path(args.report) / "gate_report.json"
        out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(str(out))
        return
    if args.report_html:
        from pyeventbt.robustness.report_html import build_html_report

        out = build_html_report(args.report_html, top_n=args.topN)
        print(str(out))
        return
    if args.summary is not None and (args.summary or args.run_dir) and not args.master:
        print(json.dumps(summarize(args.summary or args.run_dir), indent=1, default=str))
        return
    if not args.master:
        ap.error("se requiere MASTER (o usa solo --report/--summary RUN_DIR)")
    if args.backtest:
        if not args.pair or not args.timeframe:
            ap.error("--backtest requiere --pair y --timeframe")
        opt = PipelineOptimizer(args.master)
        rows = opt.single_backtest(
            dispatcher._parse_backtest_params(args.params),
            dispatcher._parse_backtest_window(opt.master, args.window),
            args.pair, args.timeframe)
        print(json.dumps(rows, indent=1, default=str))
        return
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    code = run_pipeline(args.master, only=only, pair=args.pair,
                        timeframe=args.timeframe, run_dir=args.run_dir,
                        max_reps=args.max_reps, auto_lockbox=args.auto_lockbox)
    if args.summary is not None:
        if args.run_dir:
            print(json.dumps(summarize(args.run_dir), indent=1, default=str))
        else:
            logger.warning("--summary post-run requiere --run-dir; omitido "
                           "(el run_dir se imprime en stdout)")
    sys.exit(code)


if __name__ == "__main__":
    main()
