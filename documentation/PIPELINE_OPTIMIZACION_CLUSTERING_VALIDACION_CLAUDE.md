# Pipeline Robusto v2: Optimización → Clustering → Validación (sintética + edge tests + WF/CPCV) → Decorrelación → Portfolio → Live
## Réplica mejorada del proceso DMRI_PyMT4 en PyEventBT, con config maestro, anti-overfitting state-of-the-art, calibración del propio pipeline y puente a producción

> Para **estrategia ya definida** (lógica cerrada, espacio de parámetros acotado). No data mining.
> Objetivo: **seleccionar parámetros con edge en datos futuros**, no los que mejor explican el pasado.
> Método: mismo esqueleto que `DMRI_PyMT4` (`optimize → cluster → validate_on_synthetic → decorrelate`), pero con motor **Optuna + PyEventBT** en vez de MT4 genético + `backtesting.py`, y con capas SOTA anti-overfit que DMRI no tiene (presupuesto de trials, haircut por nº de tests, DSR con N_eff, edge tests (SPA/Reality Check), PBO real (CSCV), WF/CPCV, embargo en tiempo, FDR bien planteado, ensemble por cluster).
> Todo se gobierna desde **un único archivo maestro** (`pipeline_master.yaml`). Nada hardcodeado en `.py`.

> **Versión 2.0.** Incorpora la revisión crítica de v1. Los umbrales marcados `(provisional)` son hipótesis de trabajo hasta ejecutar la **Meta-etapa M** (calibración con estrategias nulas y edge plantado); después se congelan con referencia al `calibration_report.json`.

## Cambios de v2 respecto a v1

| # | Cambio | Motivo | Dónde |
|---|---|---|---|
| 1 | Test sintético **reinterpretado**: el rank en [10,90] es un test de *consistencia*, no de edge; rank alto → ámbar (no veto); se añaden **control negativo** y **edge test** (SPA / Reality Check / Romano-Wolf / permutación) | Un GARCH con bootstrap es un modelo nulo: un edge real daría rank > 90 y v1 lo vetaba | §3.2 |
| 2 | **FDR rehecho**: p-valores de edge test (≥ 2000 remuestreos) sobre shortlist + *check de resolución* | Con 50 sims, p_min ≈ 0.02 frente a un umbral BH de q/C ≈ 0.0004: inalcanzable | §3.2 |
| 3 | **DSR con N_eff**; se guardan los **retornos diarios de cada trial** | El DSR necesita varianza del Sharpe entre trials, asimetría, curtosis y T; los trials de TPE están correlacionados | §1.4, §2.7, §3.2 |
| 4 | Nueva **Etapa 3B**: walk-forward anclado con re-optimización (WFE), CPCV y **PBO real** (CSCV). Antes "defer" | Un único corte IS/FORWARD depende de dos años concretos | Etapa 3B |
| 5 | **Lockbox**: puertas *antes* de consumirlo; contabilidad de multiplicidad; registro global de usos | v1 gastaba el FORWARD antes de aplicar las puertas; hasta 30 usos sobre la misma ventana | §4.7–4.8 |
| 6 | **Embargo en tiempo**, ≥ `max_holding + warm-up` | 50 barras < `CloseAfterNBars=96`; 50 barras H1 ≠ 50 barras H4 | §0.2 |
| 7 | **Optuna**: `constraints_func`, pruning real sobre prefijo del IS, objetivo robusto (busca mesetas), storage no-SQLite, determinismo honesto | v1 podaba tras el backtest completo, perdía evidencia y prometía un determinismo imposible con `n_jobs>1` | §1.2–1.5 |
| 8 | **Potencia estadística** (SE del Sharpe, MinTRL) y veredicto "no concluyente" | SE(Sharpe) ≈ 0.45 con 7 años | §0.4 |
| 9 | **Ledger global de investigación** (`S_global`) | El overfitting a largo plazo ocurre *entre* runs | §0.5 |
| 10 | **Meta-etapa M**: calibración del pipeline completo (FPR y potencia) | Los umbrales eran opiniones | Meta-etapa M |
| 11 | **Realismo de ejecución**: replay con ticks, orden intrabarra, costes dinámicos, grid de estrés | El backtest por barras sobreestima | §4.6, `costs` |
| 12 | **Análisis por régimen** en las puertas | "≤10/12 meses rojos" es una puerta burda | §4.7 |
| 13 | **Etapa 5 Portfolio**: asignación, exposición por divisa, correlación en estrés, DD de cartera | 5 sets del mismo EA = el mismo riesgo | Etapa 5 |
| 14 | **Etapa 6 Live**: paridad, rollout escalonado, monitorización/decay, kill switches, reconciliación, retirada | La mitad de la robustez está en producción | Etapa 6 |
| 15 | Reproducibilidad e ingeniería: Docker, DVC, MLflow, CI de regresión | Auditoría y trazabilidad | Orquestación |
| 16 | Correcciones: refs `§4.7`→`§3.2`, `max_maxdd_pct` indefinido, typo `cache/gb`, referencia a `DMRI_TRANSFER` | Consistencia | varias |

---

## Índice

1. [Arquitectura del pipeline (qué se conserva de DMRI y qué cambia)](#1-arquitectura-del-pipeline-qué-se-conserva-de-dmri-y-qué-cambia)
2. [Archivo maestro: `pipeline_master.yaml` (todo configurable)](#2-archivo-maestro-pipeline_masteryaml-todo-configurable)
3. [Etapa 0 — Preparación: spec, espacio de búsqueda, datos, presupuesto, potencia y ledger](#etapa-0--preparación-spec-espacio-de-búsqueda-datos-presupuesto-potencia-y-ledger)
4. [Etapa 1 — Optimización con Optuna (conservar solo lo que cumple criterios)](#etapa-1--optimización-con-optuna-conservar-solo-lo-que-cumple-criterios)
5. [Etapa 2 — Clustering DBSCAN/HDBSCAN de parámetros](#etapa-2--clustering-dbscanhdbscan-de-parámetros)
6. [Etapa 3 — Validación en datos sintéticos GJR-GARCH + control negativo y edge tests](#etapa-3--validación-en-datos-sintéticos-gjr-garch--control-negativo-y-edge-tests)
7. [Etapa 3B — Walk-forward anclado, CPCV y PBO real](#etapa-3b--walk-forward-anclado-cpcv-y-pbo-real)
8. [Etapa 4 — Decorrelación, estrés de ejecución, puertas y lockbox](#etapa-4--decorrelación-estrés-de-ejecución-puertas-y-lockbox)
9. [Meta-etapa M — Calibración del pipeline (nulos + edge plantado)](#meta-etapa-m--calibración-del-pipeline-nulos--edge-plantado)
10. [Etapa 5 — Portfolio: asignación, exposición y riesgo de cartera](#etapa-5--portfolio-asignación-exposición-y-riesgo-de-cartera)
11. [Etapa 6 — Live: paridad, despliegue escalonado, monitorización y retirada](#etapa-6--live-paridad-despliegue-escalonado-monitorización-y-retirada)
12. [Orquestación: stages, estado reanudable, artefactos, dashboard](#orquestación-stages-estado-reanudable-artefactos-dashboard)
13. [Tabla DMRI → PyEventBT: qué se reutiliza y qué se mejora (SOTA)](#tabla-dmri--pyeventbt-qué-se-reutiliza-y-qué-se-mejora-sota)
14. [Implementación en PyEventBT: módulos y CLI](#implementación-en-pyeventbt-módulos-y-cli)
15. [Gates numéricos y criterios de aceptación](#gates-numéricos-y-criterios-de-aceptación)
16. [Prioridades de implementación](#prioridades-de-implementación)
17. [Nota realista](#nota-realista)

---

## 1. Arquitectura del pipeline (qué se conserva de DMRI y qué cambia)

```text
pipeline_master.yaml  +  research/ledger (S_global, usos de lockbox)
        │
        ▼
[0 PREP] spec + search_space + DATA_MANIFEST (tz/DST) + cost_cfg + seed + trial_budget
        │  + potencia/MinTRL + embargo en tiempo + ledger
        ▼
[1 OPTUNA] ──IS──▶ trials.parquet + daily_returns.parquet (todos los trials)
        │  (TPE/NSGA-II, constraints_func, pruning sobre prefijo del IS, objetivo robusto)
        ▼
[2 CLUSTER] DBSCAN/HDBSCAN sobre params (± comportamiento) ──▶ representantes + N_eff
        │
        ▼
[3 SYNTH + EDGE] consistencia GJR-GARCH (rank) + control negativo + edge test (SPA/RC/perm)
        │  + DSR(N_eff) + FDR bien planteado + re-scoring estresado
        ▼
[3B WF / CPCV / PBO] walk-forward anclado con re-opt (WFE) + CSCV sobre la matriz de trials
        │
        ▼
[4 SELECT] decorrelación ──▶ jitter ──▶ replay con ticks + estrés de ejecución
        │  ──▶ PUERTAS (+ régimen) ──▶ LOCKBOX (una vez, con multiplicidad) ──▶ sizing / SL-TP
        ▼
   promoted.csv ──▶ [5 PORTFOLIO] asignación, exposición por divisa, estrés de correlación, DD cartera
        │
        ▼
[6 LIVE] paridad ──▶ paper ──▶ micro ──▶ escalado ──▶ monitorización/decay ──▶ kill switches ──▶ retirada

[M CALIBRACIÓN] estrategias nulas + edge plantado → FPR y potencia del pipeline completo
                (se ejecuta antes de fiar de un umbral y se repite al cambiar umbrales/generador/versión)
```

| Elemento DMRI | Se conserva | Cambia / mejora SOTA |
|---|---|---|
| `stages/optimize.py` barrido MT4 genético `metrics_to_test 0..8` + `nbars` | Idea de barrido por `(pair, TF)` + skip si `is_done` | Motor **Optuna** (TPE + NSGA-II multiobjetivo), `n_trials` presupuestado, pruner, constraints duras, IS con embargo; nada de `terminal.exe/.ini` |
| `tools/sets_selection.py:generate_cluster_sets` DBSCAN `eps=30, min_samples=5, manhattan` | DBSCAN + `columns_to_drop` + `groupby.mode`→medoide | Normalización robusta, auto-`eps` por k-distance, opción **HDBSCAN**, distancia mixta param+comportamiento, ensemble por cluster (no solo moda) |
| `tools/synthetic_data.py` GJR-GARCH + `fast_garch_simulation @njit` + `generate_universe(block)` | Generador + bootstrap por bloques + corrección deriva + `build_ohlc_fast` casi verbatim | `n_sims` adaptativo, stationary bootstrap, estrés de spread/slippage sintético, validación de realismo del generador, DSR + FDR/BH sobre ranks |
| `tools/metrics.py:score_percentile_rank` banda `[10,90]`, quorum `0.5` | Matemática de ranks + casos borde (`NaN/inf/varianza-cero`) + tests | Métricas netas obligatorias, quorum + **FDR**, haircut por nº trials, DSR como veto |
| `tools/decorrelation.py + stages/decorrelate.py` retornos diarios, greedy `rho<=0.7`, `max_sets=10`, score `PF+R2` | Retornos (nunca niveles), unilateralidad (negativos diversifican), `greedy/cluster`, diagnósticos | Score ampliado (`Sharpe_net, Calmar, turnover_penalty`), cap efectivo de apuestas, chequeo forward final, estabilidad local del representante |
| `stages/demo.py` puertas `PF/R2/DD/meses` + `Lots=min(...)` + `SL/TP=min/max` | Puertas y fórmula de lots | `SL/TP` por percentil/ATR (no extremos), lots con output MC p95 |
| `tools/state.py ProcessLog` + `dashboard/` + `config/bots_config.yaml` | Patrón estado/artefactos/config | Config maestro único versionado, storage Optuna RDB, dashboard con progreso por trial |

**Lo que DMRI no tiene y aquí es obligatorio:** presupuesto de trials declarado, corrección por múltiples tests, DSR/PBO, embargo temporal, FDR, generador validado, forward lockbox intocable hasta el final, ensemble por cluster en vez de "el mejor punto".

**Añadido en v2:** control negativo y edge tests (la validación sintética deja de ser el único juez), DSR con `N_eff`, PBO real (CSCV) y walk-forward con re-optimización, potencia estadística, ledger global de investigación, calibración del propio pipeline con nulos y edge plantado, realismo de ejecución (ticks, orden intrabarra, costes dinámicos), capa de portfolio y capa live (paridad, rollout, monitorización, kill switches).

---

## 2. Archivo maestro: `pipeline_master.yaml` (todo configurable)

Un solo archivo. Versionado en git. Cada `runs/` copia el master usado + `git hash`. Ningún `.py` lee constantes mágicas. Los valores marcados `(provisional)` se congelan tras la Meta-etapa M.

```yaml
# pipeline_master.yaml — MAESTRO v2. Todo lo configurable vive aquí.
pipeline_version: 2.0
seed: 1500
strategy:
  name: EA_Bollinger_Xtreme_V2
  factory: strategies.bollinger_xtreme.make_signal_fn   # (BarEvent,Modules)->SignalEvent ligado a params
  params_class: strategies.bollinger_xtreme.Params      # dataclass/BaseModel con tipos correctos
  cash: 10000
  account_currency: USD
  # Espacio de búsqueda: la ESTRATEGIA YA DEFINIDA solo expone estos rangos.
  search_space:
    Indicator_Period: {type: int, low: 10, high: 20, step: 2}
    Boll_Deviation: {type: float, low: 1.8, high: 2.8, step: 0.2}
    CCILevel: {type: int, low: 100, high: 140, step: 10}
    ADXLevel: {type: int, low: 20, high: 30, step: 5}
    CloseAfterNBars: {type: int, low: 24, high: 96, step: 24}   # time-stop por TF
    UseSLandTP: {type: categorical, choices: [true]}            # fijado por spec: siempre stop
    SL_ATR_mult: {type: float, low: 1.0, high: 2.0, step: 0.25}
  frozen: {FilterHours: true, MaxOrdenesAbiertas: 1}            # no se optimiza

universe:
  symbols: [EURUSD, GBPUSD, USDJPY]
  timeframes: [H1, H4]                       # TF señal; el filtro/régimen va en spec
  data_dir: data/m1                          # M1 formato MT5, DEL broker de producción (o contrastado con él)
  broker_server_tz: Europe/Athens            # zona horaria del servidor MT5 (DST incluido); verificada en data_check
  windows: {IS: [2015-01-01, 2021-12-31], FORWARD: [2022-01-01, 2023-12-31]}
  embargo:                                   # EN TIEMPO, no en barras (v2)
    rule: max_holding_plus_warmup            # max(CloseAfterNBars*TF, warmup_indicadores) + extra_bars
    extra_bars: 10
    apply_to: [is_forward, wf_folds, cpcv_folds]
  min_trades_IS: 100
  power:                                     # §0.4
    mintrl_check: true
    target_sharpe: 0.8
    alpha: 0.05
    inconclusive_action: flag                # flag | drop  (nunca "aprobar por defecto")

research_ledger:                             # §0.5 — GLOBAL, compartido por todos los runs/estrategias
  path: research/ledger.parquet              # append-only
  count_scope: program                       # S_global = trials de todo el programa sobre estos datos
  lockbox_registry: research/lockbox_uses.yaml

costs:                                       # CONGELADO en todo el pipeline
  commission_per_side_per_lot: 2.5
  spread_mult: 1.0
  slippage_ticks: 1
  swap_mode: from_yaml
  delay_bars: 1
  intrabar_order: worst_case                 # si SL y TP se tocan en la misma barra: gana el peor caso
  dynamic:                                   # costes dependientes del contexto (v2)
    spread_source: real_series               # serie real de spread del manifiesto, no spread plano
    rollover_window_mult: 3.0
    news_window_mult: 3.0
    slippage_vol_scaled: true                # slippage ∝ volatilidad

optimization:                                # ETAPA 1
  engine: optuna
  direction: maximize
  n_trials: 400                              # PRESUPUESTO: fijo antes de ver resultados
  n_startup_random: 40
  sampler: TPE                               # o NSGAII si multiobjetivo
  constraints_mode: sampler                  # constraints_func del sampler; NO TrialPruned tras el backtest
  pruner: median
  n_jobs: -1
  deterministic_mode: false                  # true => 1 worker (CI y test de regresión); ver §1.3
  storage: postgresql://user@host/optuna     # o JournalStorage; SQLite solo con 1 worker
  store_daily_returns: true                  # imprescindible para DSR, N_eff y CSCV/PBO
  objectives:
    primary: sharpe_net
    robust:                                  # busca mesetas en vez de picos (v2)
      enabled: true
      mode: subwindows                       # subwindows | neighborhood
      n_subwindows: 4
      lambda_std: 0.5                        # score = mean(SR_k) - lambda * std(SR_k)
    constraints:                             # se devuelven al sampler como violaciones
      min_trades: 100
      max_maxdd_pct: 25.0
      min_pf: 1.0                            # laxo aquí; el duro (1.3) va en las puertas (§4.7)
      max_exposure_pct: 80.0
  multiobjective:                            # opcional; si se activa, sampler NSGAII
    enabled: false
    names: [sharpe_net, "-maxdd_pct"]
  early_gate:                                # poda barata sobre el PREFIJO del IS (§1.2)
    enabled: true
    at_fraction: 0.5
    min_trades_fast: 30

cluster:                                     # ETAPA 2 (detalle en Etapa 2)
  enabled: true
  method: dbscan                             # dbscan | hdbscan
  feature_set: params_only                   # o params_plus_behavior (ver §2.4)
  scaler: robust                             # robust (IQR) | standard
  eps: auto                                  # auto (k-distance p90) | float
  eps_quantile: 0.90
  min_samples: 5
  metric: manhattan
  max_clusters_kept: 250                     # == DMRI cluster_sets
  representative: medoid                     # medoid | best_sharpe | ensemble3
  keep_singleton_outliers: true              # outliers buenos no se tiran
  neff: {enabled: true, method: onc_returns} # §2.7: N_eff a partir de daily_returns (siempre, aunque params_only)

validation:                                  # ETAPA 3 (detalle en §3.2)
  n_sim_screening: 20
  n_sim_final: 50
  method: block
  block_size: 10                             # + stationary: {enabled, p} ver §3.1
  stationary_bootstrap: {enabled: false, p: 0.1}
  consistency:                               # test percentile-rank DMRI = CONSISTENCIA, no edge
    percentile_low: 10.0
    percentile_high: 90.0
    low_rank_action: fail
    high_rank_action: amber                  # rank alto: sobreajuste O edge que el generador no captura
    min_metrics_pass: 0.5
    metrics:
      - {name: Sharpe Ratio, type: stat}
      - {name: Profit Factor, type: stat}
      - {name: "# Trades", type: stat}
  null_control:                              # control negativo (v2)
    enabled: true
    generator: iid_residuals                 # sin clustering de volatilidad ni estructura serial
    n_sims: 200
    max_abs_mean_sharpe: 0.15                # (provisional)
    max_pos_rate: 0.10                       # (provisional) fracción de sets nulos con Sharpe "significativo"
  edge_test:                                 # data-snooping sobre la familia de trials (v2)
    method: spa                              # spa | reality_check | romano_wolf | permutation
    resampling: stationary_bootstrap
    n_resamples_shortlist: 2000
    shortlist_size: 40
    alpha_fwer: 0.10
    family_scope: ledger                     # corrige contra S_global, no solo contra el run
  dsr:
    enabled: true
    benchmark_sr: 0.0
    n_trials_source: both                    # raw | neff | both (reportar rango)
    veto_below: 0.20                         # (provisional) DSR < 0.20 con N_eff → FAIL
  fdr:
    enabled: true
    q: 0.10
    p_source: edge_test                      # p-valores del edge test, NO de ranks
    resolution_check: abort                  # aborta si p_min alcanzable > q / C
  synth_stress: {spread_mult: 2.0, slippage_ticks_add: 1}   # re-scoring estresado (ver §3.6)

walkforward_cpcv:                            # ETAPA 3B (detalle en Etapa 3B)
  enabled: true
  pbo:  {n_groups: 10, max: 0.30, min_trials: 50}           # (provisional)
  cpcv: {n_groups: 10, k_test_groups: 2, purge: from_embargo, min_paths_positive: 0.70}
  wf:   {scheme: anchored, n_folds: 5, test_months: 12, reopt: true, n_trials_per_fold: 100,
         wfe_min: 0.5, min_folds_positive: 0.70}            # (provisional)

decorrelation:                               # ETAPA 4 (detalle en Etapa 4)
  method: greedy                             # greedy | cluster
  correlation_threshold: 0.7
  max_sets: 5                                # finalistas por (símbolo,TF) — portfolio pequeño y elite
  min_trades: 30
  resample: D
  corr_method: pearson                       # pearson | spearman
  weights: {sharpe_net: 0.4, profit_factor: 0.3, r2: 0.2, turnover_penalty: 0.1}
  pf_cap_percentile: 95.0
  stability_check: {enabled: true, jitter_pct: 10, min_green: 0.8}

execution_realism:                           # §4.6 — sobre IS, ANTES de puertas y lockbox
  tick_replay: {enabled: true, source: broker_ticks, min_coverage: 0.95}
  stress_grid:
    spread_mult: [1.0, 1.5, 2.0]
    slippage_ticks_add: [0, 1, 2]
    delay_bars: [1, 2]
  pass: {min_pf_worst_cell: 1.05, max_sharpe_decay_pct: 50}   # (provisional)
  bar_vs_tick_max_pf_gap_pct: 15

gates:                                       # §4.7 — se aplican ANTES del lockbox
  pf_min: 1.3
  pf_retest_band: [1.1, 1.3]                 # + retest 3 salidas MA/BB/RSI >= 1.3
  r2_min: 0.5
  max_dd_days: 365
  max_red_months: 10                         # de 12
  regimes: {by_year: true, by_vol_regime: true, by_session: true,
            min_share_positive: 0.60, max_single_cell_pnl_share: 0.50}

forward:                                     # §4.8 — lockbox: UN uso por (familia, símbolo, TF, set)
  min_pf: 1.1
  min_sharpe_net: 0.0
  multiplicity: {method: holm, count_scope: ledger}

promotion:
  lots: {max_dd_allow: 1200, max_trade_lost_allow: 200, mean_trade_lost_allow: 100}
  sl_tp: {metodo: percentil_MFE_MAE, p_sl: 10, p_tp: 90}
  magic_start: 11313

portfolio:                                   # ETAPA 5
  allocation: {method: hrp, covariance: ledoit_wolf, vol_target_annual: 0.10,
               kelly_fraction_cap: 0.25, max_weight_per_strategy: 0.30, rebalance: monthly}
  limits: {max_dd_portfolio_pct: 15, max_currency_net_exposure_lots: 5.0,
           max_open_positions: 12, max_leverage: 10}
  stress_corr: {method: worst_decile_days, max_rho_stress: 0.80}
  mc_envelope: {method: block_bootstrap, n_paths: 5000, percentiles: [95, 99]}

live:                                        # ETAPA 6
  rollout:
    stages:
      - {name: paper,  lot_mult: 0.0, min_days: 60, min_trades: 30}
      - {name: micro,  lot_mult: 0.1, min_days: 60}
      - {name: scaled, lot_mult: 0.5, min_days: 90}
      - {name: full,   lot_mult: 1.0}
  parity: {max_signal_mismatch_pct: 2.0, max_slippage_ticks_p95: 2.0, max_daily_tracking_error_pct: 0.5}
  monitoring:
    rolling_window_days: 60
    sharpe_alert_percentile: 5
    dd_envelope_alert: 95                    # percentil MC: alerta
    dd_envelope_stop: 99                     # percentil MC: halt
    cusum: {k: 0.5, h: 5.0}
    trade_freq_band: 0.5                     # ±50% de la frecuencia esperada
    slippage_drift_ticks: 1.0
  risk_limits:                               # capa de riesgo INDEPENDIENTE de la estrategia
    per_trade_risk_pct: 1.0
    daily_loss_pct: 3.0
    strategy_dd_pct: 10.0
    portfolio_dd_pct: 15.0
    max_spread_points: 30
    heartbeat_seconds: 30
    stale_data_seconds: 120
  reconcile: {interval_seconds: 60, on_mismatch: halt_and_alert}
  retirement:
    rule: any_of            # dd > envolvente p99 | 2 alarmas CUSUM | Sharpe rodante < p5 durante N días
    sharpe_below_p5_days: 30
    reopt_policy: none      # none | wf_quarterly (nuevo run completo, con ledger)

calibration:                                 # META-ETAPA M
  null_strategies: {n: 200, kinds: [random_entries, shuffled_signals, indicator_on_noise]}
  data_modes: [iid_synthetic, real_block_permuted]
  planted_edge: {n: 100, sharpe_levels: [0.3, 0.6, 1.0], seeds: 5}
  targets: {max_false_positive_rate: 0.05, min_power_at_sharpe_1: 0.70}
  rerun_on: [thresholds_change, generator_change, major_version]

runtime:
  robot_folder: bots_folder/{strategy}
  out_dir: runs/{strategy}/{timestamp}_{githash}
  resume: true
  reproducibility:
    container: docker
    image_digest_pinned: true
    lockfile: uv.lock
    data_versioning: dvc
    experiment_tracking: mlflow
  ci:
    regression_run: {deterministic_mode: true, n_trials: 20, expected: tests/regression/expected.json}
```

Reglas del maestro:

1. **Cambiar el maestro = nuevo run.** Nunca se edita a mitad de pipeline y se continúa; se relanza con `resume: true` (reutiliza trials/artefactos cuyo hash de config coincide).
2. **`costs`, `windows`, `embargo`, `seed`, `n_trials` son inmutables dentro de un run.** El `ProcessLog` los hashea al inicio y aborta si cambian.
3. **`search_space` solo contiene lo que la spec declara optimizable.** Añadir un parámetro a posteriori = nueva hipótesis = nuevo run + justificación en `cambios.md` (y entrada en el ledger).
4. Todo umbral de veto (`min_trades`, `veto_below`, `q`, `forward.min_pf`, `pbo.max`, ...) vive aquí, no en código.
5. **Los umbrales `(provisional)` solo se congelan con un `calibration_report.json`** (Meta-etapa M). Ablandar un umbral sin recalibrar y sin registro invalida el run.
6. **El ledger es append-only.** No se borran trials ni variantes descartadas: `S_global` se lee de ahí.
7. **Un lockbox, un uso.** Cada uso de FORWARD se registra en `lockbox_uses.yaml`; reutilizarlo tras ver el resultado deja el run contaminado.

---

## Etapa 0 — Preparación: spec, espacio de búsqueda, datos, presupuesto, potencia y ledger

**Objetivo:** que la optimización busque en el sitio correcto, con datos correctos y con un presupuesto declarado. Sin esto, Optuna es una máquina de overfitting rápida.

### 0.1 Congelar spec + search_space (de `PLAYBOOK` Fase 0)

1. `strategies/<est>/spec.yaml` cerrada: hipótesis, universo, sesión, entrada sobre **barras cerradas**, salidas (SL siempre o time-stop declarado), filtros (spread_max, ATR, noticias, `MaxOrdenes=1`), invalidez.
2. `search_space` del maestro con **tipos y pasos válidos** (el adaptador valida `step` contra `volume_step/tick_size` donde aplique). Parámetros de microstructure (spread, slippage) **nunca** en el espacio.
3. `frozen` para todo lo estructural. Si hay duda entre "buscar o fijar", fijar. Cada dimensión extra multiplica los trials necesarios y el haircut SOTA (§3.2).

### 0.2 Datos + manifiesto + embargo (en tiempo)

1. Normalizar a M1 MT5 + `DATA_MANIFEST.json` por símbolo (`fuente, filas, gaps, spread_medio/p95, hash`) **+ `broker_server_tz` y tratamiento de DST**. `data_check` verifica que el cambio de hora no crea huecos ni duplicados y que las sesiones del spec coinciden con las del servidor del broker donde se operará. Usar datos **del broker de producción** (o contrastados contra él): spreads, horarios y cierres difieren entre brokers. Reutilizar `runner/data_manifest.py + data_check.py` del playbook.
2. Ventanas del maestro: `IS` para optimizar, `FORWARD` **lockbox** (no se toca hasta la Etapa 4.8). **Embargo en tiempo, no en barras:** `embargo = max(max_holding, warm-up de indicadores) + extra_bars`, expresado en la unidad del TF de señal (con `CloseAfterNBars=96` en H4 son 16 días, no 50 barras H1). Se aplica entre IS y FORWARD y también entre folds de WF/CPCV (Etapa 3B). *(v1 usaba 50 barras fijas: menos que el holding máximo, así que un trade abierto podía cruzar la frontera, y 50 barras no son el mismo tiempo en H1 que en H4.)*
3. Chequeo `min_trades_IS` con parámetros nominales: si el nominal no da ~100 trades en IS, el espacio/TF es ilíquido para este pipeline (achicar TF o ampliar ventana, nunca bajar el umbral).

### 0.3 Presupuesto de trials + determinismo (novedad SOTA vs DMRI)

1. Fijar `n_trials` **antes** de optimizar (ej. 400) en función de dimensiones: regla práctica `50–100 trials por dimensión efectiva`, capado por coste computacional. Registrarlo en `budget.json` **y en el ledger global (§0.5)**: el haircut por nº de tests (§3.2) lo necesita.
2. `seed` única → deriva determinista por trial (`seed + trial_number`). `n_startup_random` para exploración inicial. `storage` RDB (PostgreSQL o `JournalStorage`) para resume real (DMRI resume por CSVs; aquí por trials + artefactos con hash). **Ojo con el determinismo:** con `n_jobs>1` y TPE el orden de finalización altera el muestreo; la reproducibilidad exacta solo se garantiza con `deterministic_mode: true` (§1.3).
3. Coste estimado: `n_trials × coste_backtest_IS × nº_combos`. Si excede presupuesto, reducir espacio (no `n_sim` sintético, no `min_trades`).

### 0.4 Potencia estadística (novedad v2)

Antes de optimizar hay que saber si los datos *pueden* distinguir edge de ruido.

1. **Error estándar del Sharpe anual** ≈ `√((1 + ½·SR²) / años)`. Con 7 años y SR=1 son ≈ 0.45 (IC95 ≈ ±0.9). 100 trades dan una incertidumbre del mismo orden. Un Sharpe IS de 1.0 no es, por sí solo, evidencia de nada.
2. `power.json` por combo: años efectivos, nº de trades, SE del Sharpe, **MinTRL** (años necesarios para que `target_sharpe` sea significativo a `alpha`, ajustando asimetría y curtosis) y potencia estimada para `target_sharpe`.
3. Si `años_IS < MinTRL`: el combo se marca **no concluyente**. No se aprueba ni se rechaza por métricas; solo avanza con evidencia adicional (WF/CPCV, paper alargado) o se descarta. "No concluyente" es un veredicto legítimo del pipeline, no un fallo.
4. Corolario: la solución a la falta de potencia es más datos (ventana, instrumentos relacionados, TF), nunca bajar umbrales.

### 0.5 Ledger global de investigación (novedad v2)

El overfitting más peligroso no ocurre dentro de un run, sino **entre** runs, estrategias e iteraciones sobre los mismos datos.

1. `research/ledger.parquet`, **append-only**: cada trial de cada run de cada estrategia (`run_id, strategy_family, combo, trial_number, params_hash, window, sharpe_net, n_trades, daily_returns_ref, timestamp`) más los usos de lockbox.
2. `S_global` = nº de trials del programa que tocaron la misma ventana de datos. El DSR y el edge test (§3.2) se corrigen con `S_global` (y `N_eff_global` estimado desde los retornos del ledger), no solo con el `S` del run. Una estrategia nueva sobre los mismos datos hereda la deuda estadística de las anteriores.
3. El ledger incluye las **variantes descartadas a mano** (`manual_variants` en la spec: "probé X y no funcionó"). Sin ellas, el conteo es optimista.
4. Cuando una ventana se agota (lockbox consumido para la familia), el único OOS fresco es el forward real (paper/live, Etapa 6).

**Salida Etapa 0:** `runs/.../00_prep/{spec.yaml, master_copy.yaml, manifests/, data_check.html, budget.json, power.json}` + entrada en `research/ledger.parquet` + veredicto `READY` (o `INCONCLUSIVE` por potencia).

**Qué se toma de DMRI:** `TIMEFRAME_DICT`, convención `<EST>_<PAIR>_<TF>.csv`, `ProcessLog` por combo, `historical.load_historical_data` (adaptado a M1 MT5). **Qué se añade SOTA:** embargo en tiempo, presupuesto declarado, hash de config, determinismo por trial, potencia/MinTRL, ledger global, datos del broker de producción con tz/DST verificados.

---

## Etapa 1 — Optimización con Optuna (conservar solo lo que cumple criterios)

**Objetivo:** generar `trials.parquet` completo (todos los trials, no solo ganadores — imprescindible para DSR/haircut) y filtrarlo a `candidatos.csv` con criterios duros. DMRI delegaba la búsqueda al genético de MT4 (`metrics_to_test 0..8`); aquí la búsqueda es **Optuna sobre PyEventBT** con el mismo rol pero auditable y reanudable.

### 1.1 Por qué Optuna y no grid ni genético MT4

- Grid explota combinatoriamente y desperdicia el 90% del cómputo en zonas malas. Genético MT4 no es portable a MT5/PyEventBT ni deja `trials` auditables.
- Optuna (TPE para objetivo único, NSGA-II para multiobjetivo) concentra trials donde hay señal, soporta **pruning** (mata trials malos a mitad del IS), **constraints**, espacios mixtos (`int/float/categorical`) y **resume** vía RDB. Es el estándar para `n_trials` 200–2000 con backtests de segundos-minutos.

### 1.2 Diseño del objetivo (métricas netas, constraints del sampler, pruning real, retornos guardados)

```python
# optimization/objective.py
def objective(trial, combo, master) -> float:
    cfg = master.optimization
    params = sample_search_space(trial, master)            # respeta type/step/choices
    seed = derive_seed(master.seed, trial.number)
    IS = master.universe.windows.IS

    # 1) Screening barato sobre el PREFIJO del IS.
    #    Un backtest event-driven sobre [start, mid] es prefijo exacto del backtest completo,
    #    así que la poda usa solo información intra-IS y SÍ ahorra cómputo.
    if cfg.early_gate.enabled:
        s_fast = summary(*run_pyeventbt_once(params, combo, costs=master.costs,
                                             window=prefix(IS, cfg.early_gate.at_fraction), seed=seed))
        trial.set_user_attr("n_trades_fast", s_fast.n_trades)      # el trial podado conserva evidencia
        trial.set_user_attr("sharpe_fast", s_fast.sharpe_net)
        trial.report(s_fast.sharpe_net, step=0)
        if s_fast.n_trades < cfg.early_gate.min_trades_fast or trial.should_prune():
            raise optuna.TrialPruned()

    # 2) Backtest completo (mismo cash, costes, ventana y delay para todos los trials)
    pnl, trades = run_pyeventbt_once(params, combo, costs=master.costs, window=IS, seed=seed)
    s = summary(pnl, trades)                               # analytics/stats.py
    daily = to_daily_returns(pnl)
    save_daily_returns(trial.number, daily)                # base de DSR, N_eff y CSCV/PBO

    # 3) Todo queda guardado, también si el trial viola constraints
    for k in ("pf", "n_trades", "maxdd_pct", "exposure_pct", "r2"):
        trial.set_user_attr(k, getattr(s, k))
    trial.set_user_attr("sharpe_net_full", s.sharpe_net)

    # 4) Constraints: se devuelven al sampler (<= 0 es factible). NO se lanza TrialPruned.
    c = cfg.objectives.constraints
    trial.set_user_attr("constraints", [
        c.min_trades - s.n_trades,
        s.maxdd_pct - c.max_maxdd_pct,
        c.min_pf - s.pf,
        s.exposure_pct - c.max_exposure_pct,
    ])

    # 5) Objetivo robusto: busca mesetas, no picos
    r = cfg.objectives.robust
    if r.enabled:
        sw = subwindow_sharpes(daily, n=r.n_subwindows)
        return float(np.mean(sw) - r.lambda_std * np.std(sw))
    return s.sharpe_net                                    # primario SIEMPRE neto y con costes completos

# Construcción del sampler (verifica el soporte de constraints_func en tu versión de Optuna)
sampler = optuna.samplers.TPESampler(seed=master.seed, n_startup_trials=cfg.n_startup_random,
                                     constraints_func=lambda ft: ft.user_attrs["constraints"])
```

Detalles obligatorios:

1. **Primario `sharpe_net`** (no profit, no PF solo, no win-rate): Sharpe anualizado sobre retornos diarios netos, `Rf=0`. **Con `robust.enabled` el valor optimizado es `mean(SR_k) − λ·std(SR_k)`** sobre sub-ventanas contiguas del IS: Optuna busca mesetas estables en el tiempo en vez de picos. Alternativa `mode: neighborhood` (media del Sharpe sobre vecinos ±1 step; más caro). Con multiobjetivo: `NSGAII [sharpe_net, -maxdd]`. El jitter ±10% de la Etapa 4.5 pasa a ser **confirmación**, no primer filtro de robustez.
2. **Backtest idéntico para todos los trials**: mismo `cash`, `costs`, `IS`, `delay_bars=1`, `seed` derivada. `SharedData()` aislado por trial (DMRI: `filter_strategy_params` + coerción; aquí `params_class` pydantic + `filter_params` estricto que falla ante claves desconocidas).
3. **`run_pyeventbt_once`**: escribe OHLC a temp dir en formato MT5-M1, instancia `Strategy()` + `signal_fn(params)`, `RiskPct` o `Fixed` según spec, `backtest()` → devuelve `(pnl_df, trades_df)` y los retornos diarios. Sin tocar la cola de eventos directamente.
4. **Constraints vía `constraints_func`, no `TrialPruned`.** Podar tras haber ejecutado el backtest completo no ahorra cómputo y el trial pierde sus `user_attrs`, que son evidencia para el DSR. La poda real (`TrialPruned`) solo ocurre en el screening del prefijo, y ese trial guarda al menos las métricas del prefijo.
5. **Cuándo merece la pena podar:** si el backtest completo dura < ~10 s, desactivar `early_gate` y `pruner` (el overhead no compensa) y apoyarse solo en `constraints_func`. Con backtests > 30 s, activar además `Hyperband`.
6. **Paralelo** `n_jobs=-1` a nivel trial (joblib/RDB). Prohibido paralelizar dentro del backtest y fuera a la vez (oversubscription).

### 1.3 Presupuesto, sampler, pruner, storage y determinismo (config del maestro)

- `n_trials: 400`, `n_startup_random: 40` (10%). `sampler: TPE` (o `NSGAII` si `multiobjective.enabled`). `pruner: median` (+ `Hyperband` si backtests > 30 s).
- **Storage:** SQLite no soporta bien muchos workers concurrentes (bloqueos). Con `n_jobs>1` usar **PostgreSQL o `JournalStorage`**; SQLite solo con 1 worker.
- **Determinismo honesto:** con `n_jobs>1` y TPE el orden de finalización de los trials altera el muestreo, así que `master + seed` **no** reproduce bit a bit el mismo `trials.parquet`. Dos modos: `deterministic_mode: true` (1 worker; se usa en CI y en el test de regresión) y modo paralelo (reproducible estadísticamente, no bit a bit). Lo que se audita es el **artefacto**: `trials.parquet` y `daily_returns.parquet` se hashean al terminar y el hash entra en `promoted.csv`.
- **Resume**: `storage` RDB; re-lanzar con el mismo master continúa donde quedó (DMRI: skip por `is_done`; aquí por trial hash).
- **No tocar FORWARD.** El objetivo solo ve IS. Mirar FORWARD durante la optimización invalida todo el pipeline (se detecta por `ProcessLog`: acceso a lockbox antes de Etapa 4.8 = run contaminado).

### 1.4 Conservar valores: filtrado duro + Pareto + top-k por elite (qué guarda DMRI en `optimization/*.csv`)

DMRI guarda todos los sets del genético y filtra después. Aquí igual, con criterios explícitos en el maestro:

1. Guardar **todos** los trials (también podados y no factibles) en `optimization/<EST>_<PAIR>_<TF>.csv` + `trials.parquet` (params + `sharpe_net, pf, maxdd, n_trades, exposure, r2, feasible, pruned_at` + `trial_number, seed, duration`) **+ `daily_returns.parquet`** (matriz fechas × trials). Esa matriz es la base de: DSR (varianza del Sharpe entre trials, asimetría, curtosis, T), `N_eff` (§2.7), clustering conductual (§2.4) y CSCV/PBO (Etapa 3B). Los trials podados guardan al menos las métricas del prefijo.
2. Filtro duro (`objectives.constraints`, columna `feasible`): `n_trades>=100`, `maxdd<=25%`, `exposure<=80%`, `pf>=1.0` (laxo; el 1.3 viene después).
3. Sobre los supervivientes: **Pareto** (si multiobjetivo) o **top-k por valor optimizado** con `k = 3× max_clusters_kept` (ej. 750 → cluster recorta a 250). Guardar `candidatos.csv` ordenado + `pareto.png`.
4. Registrar `S = nº de trials evaluados` y `K = nº de supervivientes` y volcarlos al **ledger global (§0.5)**: el DSR y el edge test de la Etapa 3 los necesitan. Tirar trials malos del disco es destruir evidencia contra el overfitting.

### 1.5 Anti-overfit específico de esta etapa (SOTA que DMRI no tiene)

1. **Trial budget fijo** (§0.3): más trials = más mining. Si se amplía `n_trials` a posteriori, recalcular haircut/DSR con el `S` real (y `S_global`), no con el inicial.
2. **Pruning honesto**: el pruner solo usa información intra-IS (prefijo de la ventana), nunca FORWARD.
3. **Coerción estricta de tipos** (pandas sube a `float64`, `ta`/indicadores rechazan floats): `filter_params` como DMRI `strategy_params.py` pero contra `params_class`, con error ante `NaN` o clave desconocida.
4. **Determinismo auditable**: en `deterministic_mode` el mismo `master + seed` produce el mismo `trials.parquet` (verificado por test con `n_trials=20` en CI); en modo paralelo se audita el hash del artefacto.
5. **Objetivo robusto**: la penalización por inestabilidad temporal forma parte de lo que se optimiza (§1.2), no se descubre al final.

**Salida Etapa 1:** `optimization/{<EST>_<PAIR>_<TF>.csv, trials.parquet, daily_returns.parquet, candidatos.csv, pareto.png}` + storage Optuna (PostgreSQL/Journal) + `state: optimize DONE(total_trials=S, survivors=K)` + trials volcados al ledger.

---

## Etapa 2 — Clustering DBSCAN/HDBSCAN de parámetros

**Objetivo:** convertir K candidatos correlacionados en ≤250 representantes genuinamente distintos. DMRI lo hace con `generate_cluster_sets` (DBSCAN `eps=30, min_samples=5, manhattan` + `groupby.mode`); aquí se conserva el esqueleto y se corrigen sus tres debilidades (escala sin normalizar, `eps` fijo, representante = moda frágil).

### 2.1 Entrada, features y normalización (mejora 1: escala)

1. Entrada: `candidatos.csv` (K filas). Columnas de features = claves de `search_space` menos `frozen` menos `columns_to_drop` (lista del maestro heredada de DMRI: métricas y columnas de gestión `Lots, Slippage, ...` nunca son features).
2. **Scaler robusto** (`scaler: robust`, IQR/mediana) en vez de crudo: DMRI con `eps=30` sobre params sin normalizar mezcla peras con manzanas (un `periodo 10→20` pesa 10 y una `desviación 1.8→2.8` pesa 1). Con `RobustScaler`, `eps` vive en unidades comparables.
3. Codificar `categorical` (one-hot) y `int` como numérico con su `step` respetado en el representante final (redondeo al step válido).

### 2.2 DBSCAN auto-`eps` + opción HDBSCAN (mejora 2: sin magia)

1. `method: dbscan` por defecto (paridad DMRI). `eps: auto` → curva k-distance (`k=min_samples`), `eps = cuantil(eps_quantile=0.90)` de distancias al k-ésimo vecino. Registrar `eps_auto` usado. `eps` manual solo para reproducir runs antiguos.
2. `min_samples: 5`, `metric: manhattan` (heredado DMRI; robusto en alta dimensión mixta).
3. Opción `hdbscan` (sin `eps`, `min_cluster_size=min_samples`): mejor con densidades heterogéneas (típico: una meseta grande + picos estrechos). Salida idéntica (`cluster_id`, `-1` = outlier).
4. `max_clusters_kept: 250`: si hay más clusters, quedarse con los 250 de mejor representante (mismo cap `cluster_sets` DMRI).

### 2.3 Representante por cluster (mejora 3: medoide, no moda)

DMRI usa `groupby.mode` (moda por columna): puede fabricar un punto "promedio" que no existe y que está en un valle entre dos picos. Aquí:

1. `representative: medoid` (punto real del cluster que minimiza distancia total intra-cluster). Alternativas: `best_sharpe` (mejor `sharpe_net` del cluster) o `ensemble3` (medoide + 2 vecinos para validación en ensemble, ver §2.5).
2. Redondear el medoide al `step` y `choices` válidos; re-validar con un backtest rápido que el redondeo no cambia el signo (si cambia, usar `best_sharpe`).
3. `keep_singleton_outliers: true`: outliers (`-1`) con `sharpe_net` en top-decila se conservan como singletons (DMRI los pierde o los fuerza a clusters; aquí un outlier bueno puede ser un nicho genuino, pero pasa a sintético con escrutinio extra).

### 2.4 Solo params vs params+comportamiento (opción avanzada)

- `feature_set: params_only` por defecto (paridad DMRI, barato, interpretable).
- Opción `params_plus_behavior`: concatenar params normalizados + vector de comportamiento (`retornos mensuales` o `equity mensual` normalizada, peso 0.3). Dos sets con params distintos pero misma forma de P&L acaban juntos (lo que la Etapa 4 quiere evitar). Coste: un backtest por candidato (ya hecho en Etapa 1, reutilizar `daily_returns.parquet`).

### 2.5 Ensemble por cluster (antídoto SOTA al "punto ganador")

En vez de validar un punto por cluster en la Etapa 3, validar `ensemble3` (medoide + 2 mejores del cluster) y exigir que **≥2 de 3 pasen** el sintético. Un cluster cuyo mejor punto pasa pero sus vecinos fallan es un pico, no una meseta. Coste ×3 en sintético solo para clusters finalistas (los demás, 1 punto en screening).

### 2.6 Diagnósticos obligatorios

- `cluster/<EST>_<PAIR>_<TF>.csv` (1 fila = 1 representante + `cluster_id, cluster_size, medoid_dist_p50`).
- `cluster/diagnostics/{kdistance.png, sizes_hist.png, pca2d.png, members/<id>.csv}` (PCA/t-SNE 2D con clusters coloreados + outliers).
- `state: cluster DONE(total_clusters=C)`.

### 2.7 N_eff: cuántos trials independientes hay realmente (novedad v2, alimenta el DSR)

1. Con TPE los trials se concentran en las zonas buenas: `S=400` **no son 400 hipótesis independientes**. Con la matriz `daily_returns.parquet` (Etapa 1) se agrupan los trials por correlación de retornos (ONC / clustering jerárquico sobre `1−ρ`) y se obtiene `N_eff` = nº de clusters conductuales.
2. Se calcula **siempre**, aunque `feature_set: params_only`: es un diagnóstico, no el clustering de representantes. Se guarda `cluster/neff.json {S, N_eff, method, k_range}`.
3. El DSR (§3.2d) se reporta con **ambos extremos**: `S` bruto (conservador, penaliza de más) y `N_eff` (optimista, penaliza de menos). El veto usa `N_eff`; el ranking de finalistas usa el conservador. Si la decisión cambia entre extremos, el set se marca ámbar.

**Salida Etapa 2:** ≤250 representantes auditables con su tamaño de cluster y su posición en el mapa, más `neff.json`. Nada de "top 250 por Sharpe" sin desduplicar (eso es lo que el clustering evita).

---

## Etapa 3 — Validación en datos sintéticos GJR-GARCH + control negativo y edge tests

**Objetivo:** para cada representante, responder tres preguntas distintas que v1 mezclaba en una:

- **(a) Consistencia** — ¿su rendimiento real es una muestra plausible de mercados con la misma volatilidad? (test percentile-rank de DMRI, `tools/metrics.py`).
- **(b) Sesgo** — ¿gana dinero donde no hay nada que explotar? (control negativo: calibra el backtester y la estrategia).
- **(c) Edge** — ¿su rendimiento es distinguible del azar *después* de haber probado S variantes? (edge test + DSR + FDR).

Se conserva el rank de DMRI pero se **reinterpreta**: un GARCH con bootstrap de residuos es un **modelo nulo** (conserva la volatilidad, no la estructura predictiva que explota la estrategia). Por tanto un rank real muy alto es compatible tanto con overfitting como con edge real que el generador no captura, y deja de ser veto duro por arriba. El juez del edge pasa a ser §3.2c, no el rank.

### 3.1 Generador (port DMRI + 3 endurecimientos)

1. `SyntheticMarketGenerator` (DMRI `tools/synthetic_data.py` casi verbatim): `arch GARCH(p=1,q=1,o=1,dist=t)` sobre log-retornos ×100, `std_resid`, `fast_garch_simulation @njit` con término asimétrico `gamma*res²*(res<0)`, `generate_universe(method, block_size)` con corrección de deriva a `log(Close[-1]/Open[0])`, `build_ohlc_fast` (closes por cumsum, opens retardados, mechas por ratios reales `_u/_d_ratios`).
2. **Cache por `(símbolo,TF)`** ajustado en IS (DMRI ya lo hace). `to_csv()` en M1 MT5 para que `run_pyeventbt_once` no distinga real de sintético.
3. Endurecimientos vs DMRI:
   - **Validación del generador**: antes de validar sets, comprobar que el sintético replica `volatilidad, autocorr(|r|), cola (kurtosis), rango medio diario` dentro de ±20% del real. Generador que no replica es máquina de rechazos falsos.
   - **Stationary bootstrap** opcional (`p=0.1`, longitudes geométricas) además de `block` fijo: menos artefactos de borde que `block_size=10` puro.
   - **Mechas y spread sintéticos honestos**: mechas por ratios reales (ya en DMRI) + columna `spread` = `spread_medio_real × (1 + estrés)` en el re-scoring §3.6, no spread plano 1.
   - **Generador nulo** (`iid_residuals`): mismo generador pero con residuos remuestreados iid, sin clustering de volatilidad ni estructura serial. Es el insumo del control negativo (§3.2b); sale en M1 MT5 igual que el sintético normal.

### 3.2 Protocolo de scoring (consistencia + control negativo + edge test)

**3.2a Consistencia (test DMRI, reinterpretado).**

1. Por representante: 1 backtest real (IS, `costs` maestro) + N sintéticos (`N=20` screening, `N=50` finalistas) con **idénticos params/cash/costs**.
2. `score_percentile_rank(real, synth_dist, [low=10, high=90])` por métrica (`Sharpe, PF, #Trades`): `rank=(n_below+0.5·n_equal)/n·100`. Casos borde DMRI conservados (`NaN/inf→None`, varianza-cero: igual→50 si no 0/100). **Quorum** `min_metrics_pass: 0.5`.
3. **Acción por rank:**
   - Rank **bajo** (< `low`) en el quorum → **FAIL**: en real rinde mucho peor que en mercados plausibles; algo raro en los datos reales o en la ejecución.
   - Rank **alto** (> `high`) → **ÁMBAR** (`high_rank_action: amber`): sospecha de overfitting *o* de edge que el generador no captura. Se resuelve con 3.2c y la Etapa 3B; no se veta aquí.
   - Dentro de banda → verde.
4. Interpretación explícita: este test **no prueba edge**. Detecta extremos y comportamiento anómalo.

**3.2b Control negativo (calibración del backtester y de la estrategia).**

1. Se ejecuta el pool de representantes sobre el generador nulo (`null_control.n_sims: 200`).
2. A nivel de pool: el Sharpe medio nulo debe ser ≈ 0 (`|media| ≤ max_abs_mean_sharpe`) y la fracción de sets con Sharpe nulo "significativo" ≤ `max_pos_rate`. Si el pool gana dinero de forma consistente donde no hay edge, hay **sesgo** (look-ahead, costes subestimados, orden intrabarra optimista, fills demasiado favorables): **el run se detiene y se diagnostica**; ningún resultado posterior es fiable.
3. A nivel de set: si su Sharpe medio nulo es positivo y significativo, vive de un artefacto → **FAIL**.

**3.2c Edge test sobre toda la familia de trials (data-snooping).**

1. Shortlist (`shortlist_size`): top por score entre los no-FAIL. Con `daily_returns.parquet` se aplica **White's Reality Check / Hansen SPA / Romano-Wolf stepdown** con stationary bootstrap (`n_resamples_shortlist ≥ 2000`) contra benchmark 0. La hipótesis nula es que *ningún* trial de la familia tiene edge. Alternativa `permutation` (sign-flip de retornos diarios del set).
2. Salida: p-valor **ajustado por familia** por set (`edge_p_adj`), con `alpha_fwer: 0.10`.
3. La familia contra la que se corrige es la del ledger (`S_global`, §0.5), no solo la del run.

**3.2d DSR/PSR con N_eff.**

1. PSR (con asimetría, curtosis y T reales) y **DSR** con el nº de trials de `neff.json` (§2.7). Se reporta el DSR con `S` bruto **y** con `N_eff`.
2. **Veto:** `DSR(N_eff) < veto_below` → FAIL (provisional hasta la Meta-etapa M). Aunque el resto pase, `DSR` bajo = el Sharpe se explica por haber probado muchas variantes.
3. **Haircut** por nº de tests (Harvey-Liu): reportar `Sharpe_haircut` junto al nominal; no veta solo, ordena la cola de finalistas.

**3.2e FDR bien planteado.**

1. Benjamini-Hochberg (`q: 0.10`) sobre los p-valores de `edge_p_adj` de la shortlist, **no sobre ranks**.
2. **Check de resolución:** el p mínimo alcanzable por el método debe ser ≤ `q/C`. Con 50 sims un p derivado de rank ronda `1/(n+1) ≈ 0.02`, y con `C=250`, `q=0.10` BH exigiría ≈ 0.0004: inalcanzable (por eso el FDR de v1 era decorativo o vetaba todo). Si el check falla, el pipeline **aborta con aviso**: subir remuestreos (≥ 2000 en shortlist) o reducir `C`.

**3.2f Meseta-ensemble (no es PBO).**

`ensemble3` con ≥2/3 verdes (§2.5) mide meseta local en el espacio de parámetros. Es útil, pero **no es PBO**. El PBO real (CSCV sobre la matriz trials × períodos) está en la Etapa 3B.

**Veredicto por set:**

- **VERDE:** consistencia verde (o ámbar-alto) + control negativo OK + `edge_p_adj < alpha` + `DSR` OK + FDR OK + estrés OK.
- **ÁMBAR:** cumple todo salvo uno de {estrés de costes, DSR entre extremos `S`/`N_eff`, rank alto sin edge test concluyente}.
- **ROJO:** FAIL en consistencia baja, control negativo, edge test o DSR.
- Pasan a la Etapa 3B los verdes, y los ámbares con `edge_p_adj < alpha` (con prioridad media).

### 3.3 `n_sims` adaptativo (coste bajo control)

- Screening: `n_sim_screening=20` para los C representantes → ~C·20 backtests (paralelo `n_jobs=-1`, `joblib`, conteo por worker como DMRI `_run_sims_parallel`).
- Finalistas (top por `sharpe_net` + outliers singletons): `n_sim_final=50`. Promediar coste: `C·20 + F·30` en vez de `C·50`.
- Regla: si con 20 sims el rank ya está fuera de `[5,95]` (más estricto que la banda), no gastar 50 (rechazo temprano solo por rank **bajo**; un rank alto no se rechaza, ver §3.2a).

### 3.4 Paralelo, seeds y determinismo

- `counts = reparte(n_sims, n_workers)` como DMRI; seed por `(master.seed, cluster_id, sim_idx)` → reproducible.
- `__getstate__/__setstate__` del generador (DMRI ya lo tiene para `joblib`) conservado: `model_res=None` en pickle.

### 3.5 Plots y CSVs (paridad DMRI)

- Por set: `real_vs_synth.png` (real, mediana, banda IQR, rank anotado). Por combo: `deviation_distribution.png`.
- `synthetic_validation/<EST>_<PAIR>_<TF>.csv`: filas que pasan + `ranks{sharpe,pf,trades}, diagnostics{median,iqr,n_valid}, dsr_S, dsr_neff, edge_p_adj, fdr_adj_p, haircut, null_sharpe_mean, verdict`.
- `state: synthetic DONE(total=C, passed=P)`.

### 3.6 Re-scoring estresado (novedad: costes futuros, no pasados)

- Re-evaluar los `passed` con `synth_stress: {spread_mult: 2.0, slippage_ticks_add: 1}` aplicado al ledger sintético (sin re-simular): exigir que el quorum siga verde. Un set que solo pasa con spread 2021 no tiene edge en 2026.
- Guardar `stress_pass` por set. `stress FAIL + base PASS` = ámbar (pasa a Etapa 4 con la mitad de prioridad, no con veto total).

**Salida Etapa 3:** `synthetic_validation/*.csv` con supervivientes + evidencia (ranks, control negativo, `edge_p_adj`, DSR con `S` y `N_eff`, FDR, estrés, veredicto) por set, más `null_control/` y `edge_tests/`. Lo que no pasa aquí no existe en la Etapa 3B. Aborto: control negativo fallido o check de resolución del FDR fallido ⇒ run detenido con diagnóstico.

---

## Etapa 3B — Walk-forward anclado, CPCV y PBO real

**Objetivo:** que la decisión no dependa de un único corte IS/FORWARD. Dos preguntas distintas:

1. ¿El **procedimiento** (optimizar → clusterizar → validar) produce selecciones que rinden fuera de muestra? → walk-forward **con re-optimización** (WFE).
2. ¿Cuál es la **probabilidad de que el mejor del IS sea mediocre fuera de muestra**? → PBO vía CSCV sobre la matriz de trials.

v1 dejaba walk-forward y Monte Carlo como "defer". En v2 son parte del pipeline: con un único split, el resultado depende de dos años concretos.

### 3B.1 PBO real (CSCV): barato, sin re-optimizar

1. Matriz `daily_returns` (fechas × trials) de la Etapa 1 (más ledger). Particionar en `n_groups` bloques contiguos (p.ej. 10); combinatoriamente elegir la mitad para "train" y la otra mitad para "test" (CSCV), con purga/embargo en tiempo (§0.2).
2. En cada combinación: elegir el mejor trial en train (por `sharpe_net`) y mirar su rank en test. Salida: `PBO = P(rank_OOS < mediana)`, distribución de logits λ y degradación de Sharpe IS→OOS.
3. **Gate:** `PBO ≤ pbo.max (0.30, provisional)` sobre la familia del combo. PBO alto = elegir "el mejor del IS" es esencialmente azar: el combo se invalida aunque los sets individuales pasen.

### 3B.2 CPCV: estabilidad de los finalistas

1. Para cada superviviente: distribución de Sharpe en los caminos CPCV (`n_groups`, `k_test_groups`, purga = embargo). **Gate:** mediana > 0 y ≥ `min_paths_positive` de caminos con Sharpe > 0.
2. Esto evalúa **parámetros fijos**: los parámetros ya vieron el IS, así que no es OOS estricto; mide la dependencia del período. El OOS estricto es 3B.3.

### 3B.3 Walk-forward anclado con re-optimización (WFE)

1. Esquema anclado: el train crece, el test dura `test_months` (p.ej. 12), `n_folds` ≈ 5. En cada fold se ejecuta un **mini-pipeline** (Optuna con `n_trials_per_fold` reducido + cluster + selección del representante) sobre train y se evalúa en test con embargo.
2. `WFE = Sharpe_test / Sharpe_train` (media o mediana por fold). **Gate:** `WFE ≥ wfe_min (0.5, provisional)` y ≥ `min_folds_positive` de folds con Sharpe test > 0. Mide el **procedimiento**, no un set concreto: si falla, el problema está en el espacio de búsqueda o en la estrategia, no en el parámetro elegido.
3. Coste: `n_folds × n_trials_per_fold`; se ejecuta una vez por combo, no por set.
4. Todos los trials de los folds cuentan en el ledger (§0.5).

### 3B.4 Salidas y aborto

- `wf_cpcv/<EST>_<PAIR>_<TF>.csv`: por set, distribución CPCV y caminos positivos; por combo, `PBO`, `WFE`, folds positivos. Plots: `lambda_logits.png`, `oos_degradation.png`.
- `state: wfcpcv DONE(pbo=..., wfe=..., passed=...)`.
- **Regla de aborto:** PBO o WFE fuera de gate ⇒ combo ámbar/rojo; no se fuerza la Etapa 4 sobre él.

---

## Etapa 4 — Decorrelación, estrés de ejecución, puertas y lockbox

**Objetivo:** de los P supervivientes de las Etapas 3/3B, quedarse con `k<=max_sets` (ej. 5) genuinamente distintos, robustos a la ejecución real y que pasen **todas las puertas ANTES de tocar el FORWARD**; solo entonces se gasta el lockbox, una vez. DMRI `decorrelate.py` hace la selección por `(pair,TF)` con `greedy rho<=0.7, max 10, score PF+R2`; aquí se conserva el algoritmo y se mejora el score, el cap, el orden de las puertas y el chequeo final.

**Orden en v2:** 4.1–4.4 selección → 4.5 jitter → 4.6 replay con ticks y estrés de ejecución → 4.7 puertas (+ régimen) → 4.8 lockbox → 4.9 sizing y promoción. *(En v1 las puertas se aplicaban después de consumir el FORWARD: un set que fallaba una puerta ya había gastado el lockbox en vano.)*

### 4.1 Recomputar curvas en real con idénticos costes (paridad con Etapa 3)

1. Por cada superviviente: un backtest PyEventBT en IS real con **mismo `cash/commission/date_range/costs`** que la validación (DMRI lo exige para que las curvas coincidan; aquí igual vía `run_pyeventbt_once`).
2. Derivar `n_trades, pf, r2 (equity_r2 vs ajuste lineal, 0 si <3 puntos/planos), sharpe_net, calmar, turnover (trades/mes), daily_eq → daily_returns (pct_change, drop inf)` con `resample: D`.
3. Prefiltro: `n_trades < min_trades (30)` fuera; retornos de varianza cero fuera (curvas planas no correlacionan, contaminan).

### 4.2 Matriz de correlación sobre retornos (nunca niveles)

- `returns_df.corr(method=pearson|spearman)` diaria. **Unilateral**: conservar si `rho <= threshold (0.7)` contra todo lo ya aceptado; `rho` negativo = diversificador, se queda (DMRI lo hace bien; preservarlo verbatim).
- Opción `spearman` para colas pesadas (FX con shocks): menos sensible a un día extremo común.
- Guardar `correlation.csv` + `candidates.csv` (métricas por set) como DMRI `diagnostics/`.

### 4.3 Score compuesto mejorado (PF+R2 → neto ajustado)

DMRI: `composite_scores(pf, r2, {0.5,0.5}, pf_cap p95)`. Aquí (pesos del maestro):

```text
score = 0.40·norm(sharpe_net) + 0.30·norm(pf_winsorizado p95)
      + 0.20·norm(r2) − 0.10·norm(turnover)
```

- Winsorizar `pf=inf` (sin perdedoras) al `pf_cap_percentile` antes de min-max (DMRI ya lo hace; conservar).
- `turnover_penalty`: a igual Sharpe, gana el que menos opera (menos exposición a costes futuros). Novedad vs DMRI.
- Normalización min-max sobre el pool del combo (no global): el ranking es relativo al nicho `(símbolo,TF)`.

### 4.4 Selección greedy vs cluster (mismo que DMRI, con cap elite)

- `greedy`: ordenar por score, aceptar si `max(rho vs aceptados) <= thr`, hasta `max_sets: 5`.
- `cluster`: `dist=1-rho`, `linkage(average) + fcluster(t=1-thr)`, mejor score por cluster, cap `max_sets`.
- `max_sets: 5` (no 10): portfolio de parámetros elite y operable; 10 sets de un mismo EA/TF es ilusión de diversificación (siguen siendo el mismo riesgo base).
- Cap de apuestas efectivas: reportar `n_efectivo = 1/sum(w²)` si se ponderara por score; si `n_efectivo < 0.6·k`, los k están más correlacionados de lo que `thr` sugiere (aviso ámbar).

### 4.5 Estabilidad local del representante (el "jitter test", confirmación)

- Por cada uno de los k: `jitter ±10%` en cada parámetro numérico (redondeo a step), re-backtest rápido en IS. Exigir `min_green: 0.8` (≥80% vecinos con `sharpe_net>0` y `pf>=1.0`). El que falla es pico: se sustituye por el siguiente del ranking greedy.
- Barato (k·2·d backtests) y mata los "ganadores de lotería" que sobrevivieron al sintético por azar.
- Es una **confirmación**: el primer filtro de meseta ya está en el objetivo robusto de la Etapa 1 (§1.2) y en `ensemble3` (§2.5).

### 4.6 Replay con ticks y estrés de ejecución (sobre IS, antes de puertas y lockbox)

Un backtest por barras sobreestima: ignora el orden SL/TP dentro de la barra, los gaps y el spread real de cada momento.

1. Re-ejecutar los k finalistas con **ticks del broker** (`tick_replay`) sobre el IS (o una subventana representativa si el coste es alto), con `intrabar_order: worst_case`.
2. **Grid de estrés:** `spread × {1, 1.5, 2}` × `slippage + {0, 1, 2} ticks` × `delay {1, 2} barras`. Registrar la celda peor. **Pasa** si `PF` en la peor celda ≥ `min_pf_worst_cell` y el decaimiento del Sharpe respecto al caso base ≤ `max_sharpe_decay_pct`.
3. **Divergencia barra vs ticks:** si el PF difiere más de `bar_vs_tick_max_pf_gap_pct`, el backtest por barras no es fiable para ese set (orden SL/TP, gaps): documentarlo y usar el de ticks como referencia.
4. Los costes dinámicos (spread en rollover/noticias, slippage ∝ volatilidad; `costs.dynamic`) ya se aplican desde el backtest base.
5. Va **antes** del lockbox: un set que muere por ejecución no debe haber gastado FORWARD.

### 4.7 Puertas de promoción (antes del lockbox)

1. Por set: `PF>=1.3` (o 1.1–1.3 + retest 3 salidas MA/BB/RSI con `>=1.3`), `R2>=0.5`, `DD<365d`, `≤10/12` meses rojos.
2. **Análisis por régimen** (nuevo): desglose por año, por régimen de volatilidad (terciles de ATR/volatilidad realizada) y por sesión. Exigir `≥ min_share_positive` de celdas con Sharpe > 0 y que **ninguna celda concentre más del `max_single_cell_pnl_share` del PnL** (dependencia de un único régimen). Comportamiento en estrés: peores 5% de días del universo.
3. Un set que falla puertas se descarta aquí. **Solo los que ya serían promovidos ven el lockbox.**

### 4.8 Chequeo FORWARD lockbox (una vez, al final, sin reintentos, con multiplicidad)

1. Solo los sets que pasaron 4.7 tocan `FORWARD`: un backtest por set, `forward.min_pf: 1.1` + `sharpe_net>0`. **Sin re-optimizar, sin re-elegir, sin segundo intento.** Fallar aquí = set descartado (no se vuelve a la Etapa 1 con el FORWARD ya visto; se registra y el run termina con k'<k).
2. **Multiplicidad:** con 3 símbolos × 2 TF × k=5 pueden ser hasta 30 usos sobre la misma ventana, además muy correlacionados (factor USD). Cada uso se registra en `lockbox_uses.yaml` y en el ledger, y el criterio se evalúa también con **PSR(Sharpe>0 en FORWARD) corregido por Holm** sobre el nº de usos del programa en esa ventana (no solo del run). `min_pf` es un filtro operativo, no evidencia estadística.
3. `ProcessLog` marca el acceso a lockbox; cualquier acceso anterior invalida el run (test de integridad en CI).
4. **Limitación explícita:** dos años son un solo régimen. El lockbox reduce el riesgo de overfitting, no lo elimina. Una vez usado, la ventana FORWARD queda **quemada** para esa familia de estrategias; el OOS fresco real es el paper/live (Etapa 6).
5. Guardar `forward/{...}.csv + forward_equity.png` (IS sombreado vs FORWARD) por set.

### 4.9 De k finalistas a `promoted.csv` (sizing, herencia `demo.py`)

1. `Lots = min(Lots_dd, Lots_max_los, Lots_mean_loss)` (`max_dd_allow/max_trade_lost_allow/mean_trade_lost_allow` del maestro; `0→inf`).
2. `SL/TP` por `percentil_MFE_MAE (p10/p90)` o `ATR_mult` — **nunca `min/max` extremos** (crítica `sets_to_demo_sl_tp.md` DMRI).
3. `promoted.csv`: `params + gates{...} + ranks + dsr(S y N_eff) + edge_p_adj + fdr + pbo/wfe del combo + jitter + peor celda de estrés + desglose por régimen + forward (con corrección de multiplicidad) + lots + SL/TP + cost_cfg + manifest_hashes + master_hash + ledger_snapshot_hash + commit + magic secuencial desde magic_start`. `selected_equity.png` normalizada de los k.

**Salida Etapa 4:** `decorrelation/<EST>_<PAIR>_<TF>.csv` (k filas) + `diagnostics/{candidates, correlation, selected_equity}` + `execution_realism/` + `forward/` + `promoted.csv`. Fin del **pipeline de investigación**. Lo siguiente es la construcción de cartera (Etapa 5) y la operativa en producción (Etapa 6).

---

## Meta-etapa M — Calibración del pipeline (nulos + edge plantado)

**Objetivo:** medir el comportamiento del **pipeline entero** (no de una estrategia): ¿cuántas estrategias sin edge llegan a `promoted.csv`? ¿cuántas con edge conocido sobreviven? Sin esto, `DSR 0.20`, `q=0.10`, rank `[10,90]`, `PBO 0.30` o `WFE 0.5` son opiniones. Se ejecuta antes de confiar en un umbral y se repite cuando cambian umbrales, generador de datos o versión mayor.

### M.1 Estrategias nulas → tasa de falsos positivos (FPR)

1. `n` estrategias **sin edge por construcción**: entradas aleatorias con la misma frecuencia y holding que la real, señales barajadas, indicador aplicado sobre ruido. Mismo `search_space`, mismo presupuesto de trials, mismo pipeline.
2. Datos en dos modos: **sintéticos iid** y **reales con la señal destruida** (permutación por bloques de los retornos).
3. Métrica principal: **tasa de promoción** = nulas que llegan a `promoted.csv` / nulas lanzadas. Objetivo `≤ max_false_positive_rate (5%)`.
4. Métrica diagnóstica: cuántas nulas sobreviven a *cada* etapa. Muestra qué filtros protegen de verdad y cuáles son decorativos.

### M.2 Edge plantado → potencia

1. Inyectar edge conocido: estrategia y datos donde se añade un drift condicionado a la señal para alcanzar un Sharpe objetivo (`{0.3, 0.6, 1.0}`), con varias seeds.
2. Medir la **tasa de supervivencia** por nivel. Objetivo: potencia a SR=1 `≥ 70%`.
3. Si la potencia es baja, los filtros son demasiado duros para la longitud de datos disponible (conecta con MinTRL, §0.4).

### M.3 Ajuste de umbrales

1. Los umbrales se ajustan **solo** con estas curvas (FPR/potencia), nunca mirando estrategias reales.
2. Una vez ajustados, se congelan con `calibration_report.json`, referenciado en `master_hash`. Hasta entonces permanecen `(provisional)`.

### M.4 Salidas y CI

- `calibration/{fpr_by_stage.csv, power_curve.png, calibration_report.json}` (a nivel de programa, en `research/`).
- Una calibración reducida (`n` pequeño) corre en CI para detectar que un cambio de código altera la FPR.

---

## Etapa 5 — Portfolio: asignación, exposición y riesgo de cartera

**Objetivo:** la unidad de decisión no es un set, es la **cartera**. v1 decorrelaciona por `(símbolo,TF)` y limita a 5 sets del mismo EA, que siguen siendo el mismo riesgo base. Esta etapa construye la cartera sobre todas las estrategias y símbolos promovidos.

### 5.1 Universo y correlación robusta

1. Entrada: todos los `promoted.csv` aprobados (estrategias, símbolos y TF distintos), con series de retornos diarios netos.
2. Correlación **media y en estrés** (peor decil de días): `ρ` de colas y clustering jerárquico de estrategias. La decorrelación se exige entre estrategias, no solo dentro de un combo.

### 5.2 Exposición por divisa

1. Descomponer cada posición en exposición neta por divisa (EURUSD largo = +EUR −USD) y limitar `max_currency_net_exposure`.
2. Evita que cinco pares "distintos" sean la misma apuesta corto USD.

### 5.3 Asignación

1. **HRP o risk parity** sobre covarianza robusta (shrinkage Ledoit-Wolf), **vol targeting** a `vol_target_annual`, y **Kelly fraccional** (`kelly_fraction_cap ≤ 0.25`) como techo, no como objetivo.
2. Peso máximo por estrategia (`max_weight_per_strategy`); rebalanceo mensual con banda de tolerancia.

### 5.4 Límites y envolvente de riesgo

1. Límites de cartera: DD máximo, apalancamiento, nº de posiciones, margen.
2. Monte Carlo de bloques sobre los retornos de las estrategias → **envolvente de DD (p95/p99)** de la cartera. Esa envolvente es el límite que usa la monitorización en la Etapa 6.

### 5.5 Regla de admisión y salidas

- Añadir una estrategia a la cartera exige **beneficio marginal** (Sharpe de cartera, DD) neto de costes. "Más estrategias" no es mejor por sí mismo.
- `portfolio/{weights.csv, exposure_by_currency.csv, corr_stress.png, mc_dd_envelope.csv, portfolio_report.html}`.

---

## Etapa 6 — Live: paridad, despliegue escalonado, monitorización y retirada

**Objetivo:** que lo promovido se comporte en producción como en el backtest, y detectar rápido si deja de hacerlo. Esta etapa no re-optimiza ni toca el lockbox, pero su diseño forma parte de la robustez: **nada va con dinero real sin 6.1–6.5 operativos.**

### 6.1 Paridad backtest ↔ live (shadow)

1. **Replay:** los datos reales del día pasan por el backtester con los mismos params y se compara señal a señal y fill a fill con lo que hizo (o habría hecho) el live.
2. Métricas: % de señales discordantes, slippage real vs modelado (p95), tracking error de equity diario, diferencia de spread. Gate de paridad con `live.parity`.
3. Aunque el mock de MT5 de PyEventBT comparte código entre ambos modos, hay que verificar divergencias reales: horario del servidor, rechazos/requotes, símbolos con sufijo, cambios de especificación (tick size, lot step, swaps) y cierres de mercado.

### 6.2 Despliegue escalonado

1. `paper` (≥ 60 días y ≥ 30 trades) → `micro` (lotes × 0.1) → `scaled` (× 0.5) → `full`. **Criterios de avance fijados de antemano:** paridad OK, Sharpe y DD dentro de la envolvente esperada (§6.3), cero incidentes de ejecución. También los criterios de retroceso/parada.
2. Versiones nuevas: **champion/challenger**. Nunca se cambian params en caliente.

### 6.3 Monitorización y detección de decay

1. **Sharpe rodante** (`rolling_window_days`) frente a la distribución esperada del backtest/CPCV: alerta si cae bajo el percentil `sharpe_alert_percentile`.
2. **DD en vivo** frente a la envolvente Monte Carlo (§5.4): alerta en p95, parada en p99.
3. **CUSUM / SPRT** sobre los retornos por trade frente a la media esperada.
4. **Deriva:** frecuencia de trades (banda), duración media, hit rate, slippage, spread, distribución de retornos (KS) frente al backtest y régimen de volatilidad actual frente al rango en el que se calibró.
5. Alertas con severidad (Telegram/email) y dashboard.

### 6.4 Límites de riesgo y kill switches (en código, no solo monitorización)

1. Jerarquía: por trade (riesgo máx. % equity), por día (pérdida diaria máx.), por estrategia (DD) y por cartera (DD); nº máximo de posiciones y exposición neta por divisa.
2. Operacionales: **heartbeat** con el terminal MT5, **datos stale**, spread máximo, conexión caída y hora de mercado.
3. Acciones explícitas: `halt` (no nuevas entradas) y `flatten` (cerrar), más un kill switch manual accesible.
4. La capa de riesgo es **independiente de la lógica de la estrategia**: no se confía en que la estrategia se autolimite.

### 6.5 Reconciliación e idempotencia

1. Cada `interval_seconds`: comparar posiciones, órdenes pendientes y equity internos con los del broker; discrepancia ⇒ `halt_and_alert`.
2. Órdenes **idempotentes** (identificador único vía magic + comentario; verificar antes de reenviar tras un timeout) para evitar duplicados.
3. Persistir el estado de la estrategia para reiniciar sin perder contexto.

### 6.6 Retirada y re-optimización

1. **Regla de retirada pre-registrada** (`live.retirement`): p.ej. DD > envolvente p99, dos alarmas CUSUM, o Sharpe rodante bajo p5 durante N días ⇒ la estrategia pasa a paper/shadow; no se "arregla" en caliente.
2. **Re-optimización:** política declarada de antemano (`none` o `wf_quarterly`). Toda re-optimización es un **run nuevo completo**, con los mismos gates, y consume ledger.

### 6.7 Infraestructura MT5

1. Servidor/VPS con reinicio automático, relojes sincronizados (NTP), logging inmutable y backups del estado; versión del terminal anotada.
2. Entorno reproducible (Docker donde sea posible). La librería Python oficial de MT5 suele requerir Windows: **verificar en la documentación de PyEventBT/MT5** qué plataformas soporta antes de diseñar el despliegue.

---

## Orquestación: stages, estado reanudable, artefactos, dashboard

Réplica del patrón DMRI `stages/base.py + tools/state.py + dashboard/runner.py`, adaptado a Optuna:

```text
runs/{estrategia}/{timestamp}_{githash}/
  process_logs.yaml                  # por (símbolo,TF,etapa): pending|running|done|error + counts
  master_used.yaml + master_hash
  00_prep/{spec.yaml, manifests/, data_check.html, budget.json, power.json}
  optimization/<EST>_<PAIR>_<TF>.csv + trials.parquet + daily_returns.parquet + candidatos.csv
  cluster/<EST>_<PAIR>_<TF>.csv + neff.json + diagnostics/
  synthetic_validation/<EST>_<PAIR>_<TF>.csv + null_control/ + edge_tests/ + plots/
  wf_cpcv/<EST>_<PAIR>_<TF>.csv + plots/
  decorrelation/<EST>_<PAIR>_<TF>.csv + diagnostics/ + execution_realism/ + forward/
  promoted.csv
  portfolio/{weights.csv, exposure_by_currency.csv, mc_dd_envelope.csv, ...}

research/                            # A NIVEL DE PROGRAMA (compartido por todos los runs)
  ledger.parquet                     # append-only
  lockbox_uses.yaml
  calibration_report.json + calibration/

live/{parity/, monitoring/, state/, logs/}    # servicios de producción, fuera del pipeline de investigación
```

1. `Stage` base (`name, discover_inputs, run(pair,timeframe), persist, _update_state`) + `StageContext{config, state, out_dir, ledger}`. Stages de investigación: `OptimizeOptuna → ClusterParams → ValidateSynthetic(+edge) → WalkForwardCPCV → DecorrelateSelect(+execution_realism, gates) → Lockbox → Promote → BuildPortfolio`. La Etapa 6 (live) son servicios, no stages.
2. `ProcessLog` atómico (`tmp+os.replace`), `init_stage/mark_done/mark_error` con contadores (`total_trials/survivors/clusters/passed/selected`). Re-lanzar salta `done` cuyo `config_hash` coincide; si el maestro cambió, solo re-ejecuta lo afectado (trials conservados si `search_space` no cambió).
3. Bucle `por (símbolo,TF): por stage` (DMRI `runner.py`) + `ProgressDashboard` Rich (`strategy/asset/step/progress` + logs rodantes, sin `print` en pipeline).
4. **Reglas de aborto:** `passed==0` tras sintético, control negativo fallido, check de resolución del FDR fallido, PBO/WFE fuera de gate o acceso prematuro al lockbox ⇒ el run termina en ámbar/rojo con diagnóstico. No se fuerza la siguiente etapa sobre un input vacío o inválido.
5. **Reproducibilidad e ingeniería:** entorno en contenedor (Docker con digest fijado + lockfile), datos versionados (DVC), tracking de experimentos (MLflow o similar) y **CI con run de regresión** (`deterministic_mode`, `n_trials=20`, métricas esperadas versionadas) además de la calibración reducida (Meta-etapa M).

---

## Tabla DMRI → PyEventBT: qué se reutiliza y qué se mejora (SOTA)

| Paso DMRI (archivo) | Reutilizar en PyEventBT | Mejora SOTA obligatoria |
|---|---|---|
| `stages/optimize.py` + `mt4_tools.py` (genético MT4, `metrics 0..8`, `nbars`) | Bucle por combo, skip `is_done`, `.set`→`params`, harvest CSV | **Optuna TPE/NSGA-II**, `n_trials` presupuestado, pruner, constraints, embargo, `trials.parquet` completo, RDB resume |
| `sets_selection.generate_cluster_sets` (DBSCAN `eps=30`, `mode`) | DBSCAN + `columns_to_drop` + cap 250 | RobustScaler, `eps` auto k-distance, **HDBSCAN** opcional, **medoide** (no moda), singletons buenos, `params_plus_behavior`, ensemble3 |
| `synthetic_data.SyntheticMarketGenerator` + `fast_garch_simulation` | Generador + block-bootstrap + deriva + mechas reales + cache + `__getstate__` | Validador del generador, stationary bootstrap, spread/slippage sintético, `n_sims` adaptativo 20→50, seeds por sim |
| `metrics.score_percentile_rank` + `plotting` | Rank math + edge cases + plots por set/combo | Métricas netas, **DSR veto 0.20**, **FDR q=0.10**, haircut por S, meseta-ensemble (≠PBO), control negativo, edge test (SPA/RC), PBO real en 3B, re-scoring estresado |
| `decorrelation` + `stages/decorrelate` (retornos D, greedy `0.7`, `max 10`, `PF+R2`) | Retornos (no niveles), unilateralidad, greedy/cluster, winsor `p95`, diagnósticos | Score `sharpe_net+pf+r2−turnover`, `max_sets=5`, apuestas efectivas, **jitter ±10%**, **FORWARD lockbox una vez** |
| `demo` puertas + `Lots=min` + `SL/TP=min/max` + magics + `1_for_demo` | Puertas `PF/R2/DD/meses`, fórmula lots, magics, `promoted.csv` | `SL/TP` percentil/ATR, lots con p95 MC, hash maestro+manifests en cada fila |
| `state.py + config.py + filename.py + dashboard` | `ProcessLog`, `BotConfig+ConfigError`, `TIMEFRAME_DICT`, Rich runner | Maestro único versionado, hash-check anti-cambio, test de integridad lockbox |
| *(sin equivalente en DMRI)* | — | **WF con re-opt + CPCV + PBO (CSCV)**, N_eff, potencia/MinTRL, ledger global |
| *(sin equivalente en DMRI)* | — | **Realismo de ejecución** (ticks, orden intrabarra, costes dinámicos, grid de estrés) |
| *(sin equivalente en DMRI)* | — | **Portfolio** (HRP, exposición por divisa, estrés de correlación) y **Live** (paridad, rollout, monitorización, kill switches) |
| *(sin equivalente en DMRI)* | — | **Meta-etapa M**: calibración del pipeline con nulos y edge plantado |

---

## Implementación en PyEventBT: módulos y CLI

Nuevos módulos (nada rompe lo existente):

```text
pyeventbt/
  optimization/{runner_optuna.py, params.py, cluster.py, objective.py, neff.py}
  robustness/{synthetic.py, scoring.py, validate_synthetic.py, null_control.py, edge_tests.py,
              dsr.py, fdr.py, cscv_pbo.py, cpcv.py, walkforward.py, power.py,
              decorrelation.py, monte_carlo.py, report.py}
  calibration/{null_strategies.py, planted_edge.py, report.py}
  execution/{tick_replay.py, cost_models.py, stress_grid.py}
  analytics/{stats.py, tearsheet.py, regimes.py}   # summary() con sharpe_net/pf/r2/n_trades/turnover
  selection/{gates.py, lockbox.py, lots.py, sl_tp.py, promote.py}
  portfolio/{allocation.py, exposure.py, stress_corr.py, mc_envelope.py}
  live/{parity.py, monitor.py, risk_limits.py, reconcile.py, rollout.py, retire.py}
  research/{ledger.py}
  runner/{stage.py, process_log.py, master_config.py, dashboard/}
pipeline_master.yaml
bots_dispatcher_optuna.py                     # entry: python bots_dispatcher_optuna.py pipeline_master.yaml
```

CLI:

```bash
pip install optuna arch joblib scikit-learn scipy   # + pyeventbt[opt]; psycopg2-binary (storage), mlflow, dvc
python bots_dispatcher_optuna.py pipeline_master.yaml            # todo el pipeline de investigación
python bots_dispatcher_optuna.py pipeline_master.yaml --only optimize,cluster --pair EURUSD --timeframe H1
python bots_dispatcher_optuna.py pipeline_master.yaml --only wfcpcv
python -m calibration.run pipeline_master.yaml                    # Meta-etapa M
python -m robustness.report runs/<est>/<ts>_<hash> --full         # gate_report.json agregado
python -m live.monitor pipeline_master.yaml --promoted runs/<est>/<ts>_<hash>/promoted.csv
```

Adaptador único (frontera DMRI↔PyEventBT):

```python
def run_pyeventbt_once(params, combo, *, costs, window, cash, seed) -> (pnl_df, trades_df):
    # 1. valida params contra params_class (falla ante clave desconocida/NaN)
    # 2. escribe OHLC (real o sintético) a temp dir formato MT5-M1
    # 3. Strategy() + signal_fn(params) + sizing/riesgo de spec + backtest()
    # 4. devuelve frames normalizados + summary() + retornos diarios; registra manifest_hash + cost_cfg
    # 5. modos: real | sintético | nulo (iid) | tick_replay; mismo código para todos
```

Tests que bloquean merge (patrón `tests/` DMRI):

- **Heredados:** `test_scoring` (ranks incl. bordes), `test_cluster` (medoide vs moda, eps auto), `test_params_filter` (coerción/rechazo), `test_synthetic_determinism` (misma seed → mismo universo), `test_decorr_unilateral` (negativos se quedan), `test_lockbox_integrity` (acceso temprano = fail), `test_optuna_resume` (20+20 == 40, en `deterministic_mode`).
- **Nuevos v2:** `test_fdr_resolution` (aborta si `p_min > q/C`), `test_embargo_time` (≥ `max_holding + warm-up` por TF), `test_gate_order` (las puertas se ejecutan antes del lockbox), `test_null_control` (el backtester no gana dinero en datos nulos), `test_cscv_pbo` (caso sintético con PBO conocido), `test_neff` (N_eff coherente con clusters conocidos), `test_ledger_append_only`, `test_constraints_func` (los trials infactibles conservan `user_attrs`), `test_calibration_smoke`, `test_regression_run` (run de referencia reproducible).
- **Live:** `test_live_risk_limits` (cada kill switch dispara en su condición), `test_live_reconcile` (discrepancia ⇒ `halt_and_alert`), `test_order_idempotency` (reintento tras timeout no duplica).


---

## Gates numéricos y criterios de aceptación

Defaults (el maestro los fija; endurecer por TF/símbolo, nunca ablandar sin registro y sin recalibrar). `(provisional)` = pendiente de la Meta-etapa M.

| Gate | Umbral | Fuente |
|---|---|---|
| `min_trades_IS` por trial | ≥100 | DMRI `find_valid_rules` + playbook |
| Potencia | `años_IS ≥ MinTRL(target_sharpe)`; si no, veredicto "no concluyente" | Nuevo v2 |
| Embargo | ≥ `max_holding + warm-up`, en tiempo | Nuevo v2 |
| Constraints optuna | `maxdd≤25%`, `exposure≤80%`, `pf≥1.0` vía `constraints_func` `(provisional)` | Nuevo |
| Cluster | ≤250 representantes, medoide, singletons top-decila | DMRI `cluster_sets: 250` |
| Control negativo | `|Sharpe medio nulo| ≤ 0.15`; sets nulos "significativos" ≤ 10% `(provisional)` | Nuevo v2 |
| Consistencia sintética | rank en [10,90], quorum 0.5 sobre {Sharpe, PF, #Trades}; rank bajo = FAIL, rank alto = ÁMBAR | DMRI reinterpretado |
| Edge test | `edge_p_adj < 0.10` (SPA / Reality Check / Romano-Wolf sobre `S_global`) | Nuevo v2 |
| DSR veto | `DSR(N_eff) < 0.20` → FAIL; reportar también con `S` bruto `(provisional)` | Nuevo SOTA |
| FDR | q=0.10 BH sobre p del edge test; `p_min ≤ q/C` o aborta | Nuevo v2 |
| Estrés de costes sintético | quorum verde con `spread×2 +1 tick`; si no, ámbar con mitad de prioridad | Nuevo |
| PBO (CSCV) | `≤ 0.30` `(provisional)` | Nuevo v2 |
| Walk-forward con re-opt | `WFE ≥ 0.5` y ≥70% de folds positivos `(provisional)` | Nuevo v2 |
| CPCV | mediana de Sharpe > 0 y ≥70% de caminos positivos | Nuevo v2 |
| Decorrelación | `rho≤0.7` retornos D, `max_sets=5`, `min_trades=30` | DMRI (`max 10→5` elite) |
| Jitter | ≥80% vecinos ±10% con `sharpe>0, pf≥1.0` | Nuevo |
| Estrés de ejecución | PF peor celda ≥ 1.05; decaimiento Sharpe ≤ 50%; gap barra-vs-tick ≤ 15% `(provisional)` | Nuevo v2 |
| Puertas de promoción (**antes** del lockbox) | `PF≥1.3` (o 1.1–1.3 + retest salidas ≥1.3), `R2≥0.5`, `DD<365d`, `≤10/12` rojos; ≥60% de celdas de régimen positivas; ninguna celda >50% del PnL | DMRI `filters` + nuevo |
| FORWARD lockbox | `pf≥1.1`, `sharpe_net>0`, un solo intento, PSR corregido por Holm sobre nº de usos | DMRI `end_forward_date` + lockbox estricto |
| Sizing/SLTP | `Lots=min(...)`, SL/TP percentil/ATR | DMRI fórmula + crítica extremos |
| Calibración (Meta-etapa M) | FPR del pipeline ≤ 5%; potencia a SR=1 ≥ 70% | Nuevo v2 |
| Portfolio | DD de cartera ≤ límite; exposición neta por divisa ≤ límite; beneficio marginal positivo | Nuevo v2 |
| Paridad live | ≤2% señales discordantes; slippage p95 ≤ 2 ticks; tracking error diario ≤ 0.5% | Nuevo v2 |
| Rollout | paper ≥60 d y ≥30 trades → micro ≥60 d → escalado ≥90 d → pleno | Nuevo v2 |

**Definición de DONE de investigación:** `promoted.csv` con k filas, cada una con `ranks + control negativo + edge_p_adj + dsr(S, N_eff) + fdr + pbo/wfe + jitter + estrés de ejecución + régimen + forward (con multiplicidad)` trazables + `master_hash + manifest_hashes + ledger_snapshot_hash + commit`, **y** `calibration_report.json` vigente que respalda los umbrales. Menos que eso es un informe, no una selección.

**Definición de DONE de producción:** cartera construida (Etapa 5), paridad verificada, kill switches y reconciliación probados, rollout completado según criterios fijados de antemano y monitorización con envolvente activa (Etapa 6).

---

## Prioridades de implementación

| Prioridad | Qué | Por qué primero |
|---|---|---|
| **P0** | `run_pyeventbt_once` + scoring + sintético + objetivo Optuna con `constraints_func` y retornos guardados. Corregir antes el diseño de §3.2 (semántica del test, FDR) y el orden puertas → lockbox (§4.7–4.8) | Son errores de diseño, no de implementación: costarán más cuanto más tarde se corrijan |
| **P1** | `daily_returns` + `N_eff` + DSR correcto + WF/CPCV/PBO + embargo en tiempo + potencia/MinTRL + **Meta-etapa M** (calibración) | Sin esto no se sabe cuánto de lo que sobrevive es edge |
| **P2** | Realismo de ejecución (ticks, grid de estrés), ledger global, análisis por régimen, capa de portfolio | Cierra la brecha backtest → ejecución real y evita sobreestimar la diversificación |
| **P3** | Capa live completa (paridad, rollout, monitorización, kill switches, reconciliación). Puede desarrollarse en paralelo, pero **nada va con dinero real sin paridad y kill switches** | Es donde se pierde dinero de verdad |

## Nota realista

Este pipeline reduce los falsos positivos, pero **no crea edge**. Lo que distingue a los mejores fondos no es solo el control del overfitting, sino hipótesis con fundamento económico, muchas fuentes de alfa descorrelacionadas, ejecución y gestión de riesgo. La `spec.yaml` pre-registrada con hipótesis y razón de ser es la parte más valiosa de este documento. El éxito se mide con el paper/live a largo plazo y con la calibración del propio pipeline (FPR y potencia), no con cuántos sets sobreviven.

---

*Primera acción: copiar `config/bots_config.yaml` de DMRI a `pipeline_master.yaml` con este esquema, implementar `run_pyeventbt_once + scoring + synthetic` (P0) y pasar UNA estrategia × UN combo (ej. `EURUSD H1, n_trials=100, n_sim=20`) de punta a punta, incluyendo control negativo. En paralelo, montar una calibración mínima (Meta-etapa M) con 20–50 estrategias nulas para ver qué filtros protegen de verdad. El resto es escalar lo que ya funciona, no construir más teoría.*
