# AGENTS.md — pyeventbt-private

> Fuente de verdad para agentes: este archivo + `README.md` + `documentation/PIPELINE_OPTIMIZACION_CLUSTERING_VALIDACION_FINAL.md` + `documentation/TODO_DESARROLLADOR_PIPELINE.md`. Si hay conflicto, manda el PIPELINE FINAL v3.0. Docs extra en `documentation/`.

## Qué es este repo
**PyEventBT v0.0.13** (Python 3.12, Poetry): framework event-driven de backtest + live para MetaTrader 5, en Python puro sin MQL5. Principio: **one codebase, two modes** (mock completo API MT5).
Pipeline de eventos: `BarEvent → Strategy (signal) → SignalEvent → Sizing → Risk → OrderEvent → Execution → FillEvent → Portfolio` (`pyeventbt/strategy/strategy.py`, `pyeventbt/trading_director/`).

## Qué se está desarrollando (no tocar el core sin motivo)
Pipeline robusto de selección de parámetros para **estrategia ya definida** (no data mining). Flujo:
`[0 PREP] spec+manifiesto+embargo+ledger → [1 OPTUNA] trials.parquet+daily_returns → [2 CLUSTER] DBSCAN/HDBSCAN medoide+N_eff → [3 SYNTH+EDGE] GARCH+control nulo+SPA/RC+DSR+FDR → [3B WF/CPCV/PBO] → [4 SELECT] decorrelación→jitter→tick replay→PUERTAS→LOCKBOX→sizing → [5 PORTFOLIO] → [6 LIVE] → [M CALIBRACIÓN]`. Detalle y umbrales en PIPELINE FINAL §§1-2,7.

Primera acción vigente: `run_pyeventbt_once + scoring + synthetic + objective` (P0) + 1 combo punta a punta (ej. EURUSD H1, n_trials=100, n_sim=20) + calibración mínima con 20-50 nulas.

## Comandos
- `poetry install` / `poetry run python example_ma_crossover.py`
- Pipeline (cuando exista): `poetry run python bots_dispatcher_optuna.py pipeline_master.yaml`
- Solo etapas: `poetry run python bots_dispatcher_optuna.py pipeline_master.yaml --only optimize,cluster --pair EURUSD --timeframe H1`
- Calibración / report: `poetry run python -m calibration.run pipeline_master.yaml` ; `poetry run python -m robustness.report runs/<est>/<ts>_<hash> --full`
- Tests (bloquean merge): `poetry run pytest -q` — deben existir: scoring, cluster (medoide vs moda, eps auto), params_filter, synthetic_determinism, decorr_unilateral, lockbox_integrity, optuna_resume, fdr_resolution, embargo_time, gate_order, null_control, cscv_pbo, neff, ledger_append_only, constraints_func, live_risk_limits/reconcile/idempotency.

## Estructura esperada (nueva, no rompe lo existente)
`pyeventbt/optimization|robustness|calibration|execution|analytics|selection|portfolio|live|research|runner` + `pipeline_master.yaml` + `bots_dispatcher_optuna.py` + `strategies/<est>/spec.yaml` + `runs/...` + `research/ledger.parquet` (append-only) + `research/lockbox_uses.yaml`. Ver PIPELINE FINAL §§2, "Orquestación", "Implementación".

## Reglas duras del pipeline (no negociar)
1. Todo configurable vive en `pipeline_master.yaml`. Nada hardcodeado en `.py`. Cambiar maestro = nuevo run con `resume:true`; `costs/windows/embargo/seed/n_trials` inmutables dentro del run (hash + aborta si cambian).
2. Embargo **en tiempo**: `max(max_holding, warmup)+extra_bars` en unidad del TF señal. Entre IS/FORWARD y entre folds WF/CPCV.
3. Orden obligatorio Etapa 4: selección → jitter ±10% → replay ticks + grid estrés → **PUERTAS → LOCKBOX (una vez, sin reintentos, Holm sobre S_global) → sizing/SL-TP**. Puertas después del lockbox = run inválido.
4. Ledger `research/ledger.parquet` append-only; `S_global` = trials del programa sobre misma ventana. DSR/edge/FDR se corrigen con `S_global` y `N_eff`, no solo S del run.
5. Rank sintético [10,90]: rank bajo=FAIL, rank alto=ÁMBAR (lo resuelve edge test+3B). FDR solo sobre p del edge test, con check `p_min<=q/C` o aborta. Veto `DSR(N_eff)<0.20` → FAIL.
6. Señal solo sobre **barras cerradas**; `delay_bars=1`; costes congelados (`spread real, slippage, swap, worst_case intrabarra`); `run_pyeventbt_once(params,combo,*,costs,window,cash,seed)->(pnl,trades)` valida con `params_class` pydantic (falla ante clave desconocida/NaN).
7. Gates clave: `n_trades>=100, maxdd<=25%, exposure<=80%, PF>=1.0` (Optuna laxas) → puertas `PF>=1.3, R2>=0.5, DD<365d, <=10/12 rojos, >=60% celdas régimen positivas` → lockbox `pf>=1.1+sharpe>0` → `PBO<=0.30, WFE>=0.5, CPCV mediana>0 + >=70% positivos`.
8. reproducibilidad: Docker digest + lockfile, DVC datos, MLflow tracking, CI regresión (`deterministic_mode`, n_trials=20).

## Estilo
- Python 3.12, Decimal para precios/SL-TP, `polars`/`numpy`/`numba` en indicadores; no `print` en pipeline (dashboard Rich).
- Mantener este archivo <150 líneas; detalle estadístico en PIPELINE FINAL, tareas en TODO_DESARROLLADOR_PIPELINE.md dentro de `documentation/`.
