# Pipeline Robusto FINAL: Optimización → Clustering → Validación (sintética + edge + WF/CPCV/PBO) → Selección → Portfolio → Live
## Réplica mejorada del proceso DMRI_PyMT4 en PyEventBT, con config maestro, anti-overfitting state-of-the-art, calibración del propio pipeline y gobernanza de ciclo de vida

> Para **estrategia ya definida** (lógica cerrada, espacio de parámetros acotado). No data mining.
> Objetivo: **seleccionar parámetros con edge en datos futuros**, no los que mejor explican el pasado.
> Todo se gobierna desde **un único archivo maestro** (`pipeline_master.yaml`). Nada hardcodeado en `.py`.

---

## 0. Veredicto sobre las tres versiones (qué se conserva y qué se descarta)

Este documento **sintetiza y sustituye** a los tres anteriores. Comparación como experto:

| Fuente | Fortaleza | Debilidad | Veredicto |
|---|---|---|---|
| **Original (v1)** | Esqueleto operativo excelente: maestro YAML, salidas por etapa, tabla DMRI→PyEventBT, gates concretos, "primera acción" pragmática | 5 errores de diseño estadístico: (1) el rank [10,90] veta por arriba lo que puede ser edge real; (2) FDR sobre ranks es inalcanzable con 50 sims; (3) DSR con S bruto ignora que los trials TPE están correlacionados; (4) puertas aplicadas DESPUÉS de gastar el lockbox; (5) embargo en barras no escala con el TF | **Base conservada**, corregida por CLAUDE |
| **CHATGPT (v2)** | Mejor visión de plataforma: ciclo de vida (registry, approval package), leakage firewall, capacity, MC lab, health score, CI/CD, roadmap P0–P5 | Abstracto y tipo checklist; **no corrige los errores estadísticos de v1**, solo añade capas encima; sobre-ingeniería en partes (data layer institucional completa, queue/market impact models, restructura total del repo) | **Se adopta la capa de gobernanza y ops condensada**; se descarta la sobre-ingeniería |
| **CLAUDE (v2)** | La revisión técnicamente correcta: reinterpreta el test sintético, añade control negativo y edge tests (SPA/Reality Check/Romano-Wolf), DSR con N_eff, PBO real (CSCV), walk-forward con re-optimización, embargo en tiempo, potencia/MinTRL, ledger global, Meta-etapa M (calibración del pipeline), realismo de ejecución, portfolio y live concretos | Denso; alcance extenso para una primera implementación (mitigado aquí con prioridades claras) | **Espina dorsal técnica de esta versión FINAL** |

### Lo que se descarta explícitamente (y por qué)

1. **Simulador de ejecución institucional completo** (queue models, market impact microestructural): no aplica a FX/CFD vía MT5 a escala retail/pro-fe pequeña. El nivel correcto es **tick replay + orden intrabarra worst-case + grid de estrés de costes** (CLAUDE §4.6). El *capacity analysis* ligero sí se conserva (Etapa 5).
2. **Data layer institucional de 6 capas** (RAW→FEATURE STORE→…): se sustituye por **manifiesto de datos + leakage firewall como tests** (CHATGPT §14.3 reducido a lo accionable). DVC cubre el versionado.
3. **Event sourcing completo**: se conserva **reconciliación + idempotencia de órdenes + persistencia de estado** (CLAUDE §6.5), que resuelve el problema real sin construir un event store.
4. **Restructura total del repositorio** (CHATGPT §26): rompería pyeventbt existente. Se mantiene el plan de módulos de CLAUDE, que encaja con la librería actual.
5. **Roadmap P5 "institutionalization"** (multi-strategy distributed, cross-asset): fuera de alcance, se anota como horizonte.

### Lo que se gana siendo hedge-fund-level

No es una métrica: es **FPR y potencia del propio pipeline medidos** (Meta-etapa M), **corrección por toda la selección histórica** (ledger global), **un solo uso del lockbox con multiplicidad**, **potencia estadística antes de prometer nada** y **paridad backtest↔live verificada antes de escalar capital**. Eso, y no un Sharpe alto, es lo que distingue a un shop serio.

---

## 1. Arquitectura del pipeline

```text
pipeline_master.yaml  +  research/ledger (S_global, usos de lockbox)
        │
        ▼
[0 PREP] spec + search_space + DATA_MANIFEST (tz/DST) + cost_cfg + seed + trial_budget
        │  + potencia/MinTRL + embargo EN TIEMPO + leakage firewall (tests)
        ▼
[1 OPTUNA] ──IS──▶ trials.parquet + daily_returns.parquet (TODOS los trials)
        │  (TPE/NSGA-II, constraints_func del sampler, pruning sobre prefijo del IS,
        │   objetivo robusto: busca mesetas, no picos)
        ▼
[2 CLUSTER] DBSCAN/HDBSCAN sobre params normalizados (± comportamiento) ──▶ representantes + N_eff
        │  (medoide por cluster + outliers buenos como singletons + ensemble3)
        ▼
[3 SYNTH + EDGE] consistencia GJR-GARCH (rank reinterpretado) + control negativo
        │  + edge test (SPA/Reality Check/Romano-Wolf) + DSR(N_eff) + FDR con check de resolución
        │  + re-scoring estresado de costes
        ▼
[3B WF / CPCV / PBO] walk-forward anclado con re-optimización (WFE) + CSCV sobre matriz de trials
        │
        ▼
[4 SELECT] decorrelación ──▶ jitter ±10% ──▶ tick replay + grid de estrés de ejecución
        │  ──▶ PUERTAS (+ régimen) ──▶ LOCKBOX (una vez, con corrección de Holm) ──▶ sizing / SL-TP
        ▼
   promoted.csv ──▶ [5 PORTFOLIO] HRP/asignación, exposición por divisa, estrés de correlación,
        │            envolvente MC de DD, capacity
        ▼
[6 LIVE] paridad ──▶ paper ──▶ micro ──▶ escalado ──▶ monitorización/decay ──▶ kill switches ──▶ retirada

[M CALIBRACIÓN] estrategias nulas + edge plantado → FPR y potencia del pipeline completo
                (antes de fiar de un umbral; se repite al cambiar umbrales/generador/versión)
```

Tres preguntas distintas que v1 mezclaba en una (contribución clave de CLAUDE):

- **(a) Consistencia** — ¿el rendimiento real es una muestra plausible de mercados con la misma volatilidad? (rank sintético)
- **(b) Sesgo** — ¿gana dinero donde no hay nada que explotar? (control negativo)
- **(c) Edge** — ¿es distinguible del azar *después* de haber probado S variantes? (edge test + DSR + FDR)

---

## 2. Archivo maestro: `pipeline_master.yaml`

Un solo archivo. Versionado en git. Cada `runs/` copia el master usado + git hash. Los umbrales marcados `(provisional)` son hipótesis de trabajo hasta la Meta-etapa M; después se congelan referenciando `calibration_report.json`.

```yaml
# pipeline_master.yaml — MAESTRO. Todo lo configurable vive aquí.
pipeline_version: 3.0
seed: 1500
strategy:
  name: EA_Bollinger_Xtreme_V2
  factory: strategies.bollinger_xtreme.make_signal_fn   # (BarEvent,Modules)->SignalEvent ligado a params
  params_class: strategies.bollinger_xtreme.Params      # pydantic/dataclass con tipos correctos
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
  data_dir: data/m1                          # M1 MT5 DEL broker de producción (o contrastado con él)
  broker_server_tz: Europe/Athens            # tz del servidor MT5 (DST incluido); verificada en data_check
  windows: {IS: [2015-01-01, 2021-12-31], FORWARD: [2022-01-01, 2023-12-31]}
  embargo:                                   # EN TIEMPO, no en barras
    rule: max_holding_plus_warmup            # max(CloseAfterNBars*TF, warmup indicadores) + extra_bars
    extra_bars: 10
    apply_to: [is_forward, wf_folds, cpcv_folds]
  min_trades_IS: 100
  power:                                     # potencia estadística (§4)
    mintrl_check: true
    target_sharpe: 0.8
    alpha: 0.05
    inconclusive_action: flag                # flag | drop (nunca "aprobar por defecto")

research_ledger:                             # GLOBAL, compartido por todos los runs/estrategias
  path: research/ledger.parquet              # append-only
  count_scope: program                       # S_global = trials de todo el programa sobre estos datos
  lockbox_registry: research/lockbox_uses.yaml

costs:                                       # CONGELADO en todo el pipeline
  commission_per_side_per_lot: 2.5
  spread_mult: 1.0
  slippage_ticks: 1
  swap_mode: from_yaml
  delay_bars: 1
  intrabar_order: worst_case                 # SL y TP en la misma barra => gana el peor caso
  dynamic:                                   # costes dependientes del contexto
    spread_source: real_series               # serie real del manifiesto, no spread plano
    rollover_window_mult: 3.0
    news_window_mult: 3.0
    slippage_vol_scaled: true

optimization:                                # ETAPA 1
  engine: optuna
  direction: maximize
  n_trials: 400                              # PRESUPUESTO: fijo antes de ver resultados
  n_startup_random: 40
  sampler: TPE                               # o NSGAII si multiobjetivo
  constraints_mode: sampler                  # constraints_func; NO TrialPruned tras el backtest completo
  pruner: median
  n_jobs: -1
  deterministic_mode: false                  # true => 1 worker (CI y regresión); ver §5
  storage: postgresql://user@host/optuna     # o JournalStorage; SQLite solo con 1 worker
  store_daily_returns: true                  # imprescindible para DSR, N_eff y CSCV/PBO
  objectives:
    primary: sharpe_net
    robust:                                  # busca mesetas en vez de picos
      enabled: true
      mode: subwindows                       # subwindows | neighborhood
      n_subwindows: 4
      lambda_std: 0.5                        # score = mean(SR_k) - lambda * std(SR_k)
    constraints:                             # se devuelven al sampler como violaciones
      min_trades: 100
      max_maxdd_pct: 25.0
      min_pf: 1.0                            # laxo aquí; el duro (1.3) va en las puertas
      max_exposure_pct: 80.0
  multiobjective:                            # opcional; si se activa, sampler NSGAII
    enabled: false
    names: [sharpe_net, "-maxdd_pct"]
  early_gate:                                # poda barata sobre el PREFIJO del IS
    enabled: true
    at_fraction: 0.5
    min_trades_fast: 30

cluster:                                     # ETAPA 2
  enabled: true
  method: dbscan                             # dbscan | hdbscan
  feature_set: params_only                   # o params_plus_behavior
  scaler: robust                             # robust (IQR) | standard
  eps: auto                                  # auto (k-distance p90) | float
  eps_quantile: 0.90
  min_samples: 5
  metric: manhattan
  max_clusters_kept: 250
  representative: medoid                     # medoid | best_sharpe | ensemble3
  keep_singleton_outliers: true
  neff: {enabled: true, method: onc_returns} # N_eff desde daily_returns (siempre)

validation:                                  # ETAPA 3
  n_sim_screening: 20
  n_sim_final: 50
  method: block
  block_size: 10
  stationary_bootstrap: {enabled: false, p: 0.1}
  consistency:                               # test rank DMRI = CONSISTENCIA, no edge
    percentile_low: 10.0
    percentile_high: 90.0
    low_rank_action: fail
    high_rank_action: amber                  # rank alto: overfitting O edge que el generador no captura
    min_metrics_pass: 0.5
    metrics:
      - {name: Sharpe Ratio, type: stat}
      - {name: Profit Factor, type: stat}
      - {name: "# Trades", type: stat}
  null_control:                              # control negativo (calibra backtester + estrategia)
    enabled: true
    generator: iid_residuals                 # sin clustering de volatilidad ni estructura serial
    n_sims: 200
    max_abs_mean_sharpe: 0.15                # (provisional)
    max_pos_rate: 0.10                       # (provisional)
  edge_test:                                 # data-snooping sobre la familia de trials
    method: spa                              # spa | reality_check | romano_wolf | permutation
    resampling: stationary_bootstrap
    n_resamples_shortlist: 2000
    shortlist_size: 40
    alpha_fwer: 0.10
    family_scope: ledger                     # corrige contra S_global, no solo el run
  dsr:
    enabled: true
    benchmark_sr: 0.0
    n_trials_source: both                    # raw | neff | both (reportar rango)
    veto_below: 0.20                         # (provisional) DSR < 0.20 con N_eff => FAIL
  fdr:
    enabled: true
    q: 0.10
    p_source: edge_test                      # p-valores del edge test, NO de ranks
    resolution_check: abort                  # aborta si p_min alcanzable > q / C
  synth_stress: {spread_mult: 2.0, slippage_ticks_add: 1}

walkforward_cpcv:                            # ETAPA 3B
  enabled: true
  pbo:  {n_groups: 10, max: 0.30, min_trials: 50}            # (provisional)
  cpcv: {n_groups: 10, k_test_groups: 2, purge: from_embargo, min_paths_positive: 0.70}
  wf:   {scheme: anchored, n_folds: 5, test_months: 12, reopt: true, n_trials_per_fold: 100,
         wfe_min: 0.5, min_folds_positive: 0.70}             # (provisional)

decorrelation:                               # ETAPA 4
  method: greedy                             # greedy | cluster
  correlation_threshold: 0.7
  max_sets: 5                                # portfolio pequeño y elite (no 10)
  min_trades: 30
  resample: D
  corr_method: pearson                       # pearson | spearman
  weights: {sharpe_net: 0.4, profit_factor: 0.3, r2: 0.2, turnover_penalty: 0.1}
  pf_cap_percentile: 95.0
  stability_check: {enabled: true, jitter_pct: 10, min_green: 0.8}

execution_realism:                           # sobre IS, ANTES de puertas y lockbox
  tick_replay: {enabled: true, source: broker_ticks, min_coverage: 0.95}
  stress_grid:
    spread_mult: [1.0, 1.5, 2.0]
    slippage_ticks_add: [0, 1, 2]
    delay_bars: [1, 2]
  pass: {min_pf_worst_cell: 1.05, max_sharpe_decay_pct: 50}  # (provisional)
  bar_vs_tick_max_pf_gap_pct: 15

gates:                                       # se aplican ANTES del lockbox
  pf_min: 1.3
  pf_retest_band: [1.1, 1.3]                 # + retest 3 salidas MA/BB/RSI >= 1.3
  r2_min: 0.5
  max_dd_days: 365
  max_red_months: 10                         # de 12
  regimes: {by_year: true, by_vol_regime: true, by_session: true,
            min_share_positive: 0.60, max_single_cell_pnl_share: 0.50}

forward:                                     # lockbox: UN uso por (familia, símbolo, TF, set)
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
  capacity: {capital_grid: [10k, 50k, 250k, 1M], metric: sharpe_decay_pct, max_decay_pct: 25}

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
    rule: any_of             # dd > envolvente p99 | 2 alarmas CUSUM | Sharpe rodante < p5 N días
    sharpe_below_p5_days: 30
    reopt_policy: none       # none | wf_quarterly (siempre run nuevo completo, con ledger)

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
    container: docker                        # digest fijado + lockfile
    data_versioning: dvc
    experiment_tracking: mlflow
  ci:
    regression_run: {deterministic_mode: true, n_trials: 20, expected: tests/regression/expected.json}
```

Reglas del maestro:

1. **Cambiar el maestro = nuevo run.** Nunca se edita a mitad de pipeline y se continúa; se relanza con `resume: true` (reutiliza artefactos cuyo hash de config coincide).
2. **`costs`, `windows`, `embargo`, `seed`, `n_trials` son inmutables dentro de un run.** El `ProcessLog` los hashea al inicio y aborta si cambian.
3. **`search_space` solo contiene lo que la spec declara optimizable.** Añadir un parámetro a posteriori = nueva hipótesis = nuevo run + entrada en el ledger + justificación en `cambios.md`.
4. Todo umbral de veto vive aquí, no en código.
5. **Los umbrales `(provisional)` solo se congelan con un `calibration_report.json`** (Meta-etapa M). Ablandar un umbral sin recalibrar y sin registro invalida el run.
6. **El ledger es append-only.** No se borran trials ni variantes descartadas: `S_global` se lee de ahí.
7. **Un lockbox, un uso.** Cada uso de FORWARD se registra en `lockbox_uses.yaml`; reutilizarlo tras ver el resultado deja el run contaminado.

---

## Etapa 0 — Preparación: spec, datos, presupuesto, potencia y ledger

**Objetivo:** que la optimización busque en el sitio correcto, con datos correctos y con un presupuesto declarado. Sin esto, Optuna es una máquina de overfitting rápida.

### 0.1 Congelar spec + search_space

1. `strategies/<est>/spec.yaml` cerrada: hipótesis (con fundamento económico: por qué debería existir este edge), universo, sesión, entrada sobre **barras cerradas**, salidas (SL siempre o time-stop declarado), filtros (spread_max, ATR, noticias, `MaxOrdenes=1`), invalidez. **La spec pre-registrada es el activo más valioso del framework.**
2. `search_space` con **tipos y pasos válidos**. Parámetros de microestructura (spread, slippage) **nunca** en el espacio.
3. `frozen` para todo lo estructural. Si hay duda entre "buscar o fijar", fijar: cada dimensión extra multiplica trials y haircut.

### 0.2 Datos + manifiesto + embargo EN TIEMPO + leakage firewall

1. Normalizar a M1 MT5 + `DATA_MANIFEST.json` por símbolo (`fuente, filas, gaps, spread_medio/p95, hash`) **+ `broker_server_tz` y tratamiento de DST**. `data_check` verifica que el cambio de hora no crea huecos ni duplicados y que las sesiones del spec coinciden con las del servidor del broker donde se operará. Usar datos **del broker de producción** (o contrastados contra él).
2. Ventanas: `IS` para optimizar, `FORWARD` **lockbox** (no se toca hasta 4.8). **Embargo en tiempo, no en barras:** `embargo = max(max_holding, warm-up de indicadores) + extra_bars`, expresado en la unidad del TF de señal (con `CloseAfterNBars=96` en H4 son 16 días, no 50 barras H1). Se aplica entre IS y FORWARD y entre folds de WF/CPCV.
3. **Leakage firewall (tests que bloquean, no warnings):** look-ahead (señal sobre barra no cerrada), timestamps desordenados, barras incompletas, duplicados, joins temporales incorrectos, uso accidental del lockbox, survivorship en el universo. Un fallo de integridad produce **FAIL**.
4. Chequeo `min_trades_IS` con parámetros nominales: si el nominal no da ~100 trades en IS, el espacio/TF es ilíquido para este pipeline (achicar TF o ampliar ventana, nunca bajar el umbral).

### 0.3 Presupuesto de trials + determinismo

1. Fijar `n_trials` **antes** de optimizar: regla práctica `50–100 trials por dimensión efectiva`. Registrarlo en `budget.json` **y en el ledger global**: el haircut/DSR lo necesita.
2. `seed` única → deriva determinista por trial (`seed + trial_number`). Storage RDB (PostgreSQL o `JournalStorage`); SQLite solo con 1 worker.
3. Coste estimado: `n_trials × coste_backtest_IS × nº_combos`. Si excede presupuesto, reducir espacio (no `n_sim` sintético, no `min_trades`).

### 0.4 Potencia estadística (antes de optimizar hay que saber si los datos pueden distinguir edge de ruido)

1. **SE del Sharpe anual ≈ √((1 + ½·SR²) / años)**. Con 7 años y SR=1 son ≈ 0.45 (IC95 ≈ ±0.9). Un Sharpe IS de 1.0 no es, por sí solo, evidencia de nada.
2. `power.json` por combo: años efectivos, nº de trades, SE del Sharpe, **MinTRL** (años para que `target_sharpe` sea significativo a `alpha`, ajustando asimetría y curtosis) y potencia estimada.
3. Si `años_IS < MinTRL`: el combo se marca **no concluyente**. No se aprueba ni se rechaza por métricas; avanza solo con evidencia adicional (WF/CPCV, paper alargado) o se descarta. "No concluyente" es un veredicto legítimo.
4. Corolario: la solución a la falta de potencia es más datos, nunca bajar umbrales.

### 0.5 Ledger global de investigación

El overfitting más peligroso no ocurre dentro de un run, sino **entre** runs y estrategias sobre los mismos datos.

1. `research/ledger.parquet`, **append-only**: cada trial de cada run (`run_id, strategy_family, combo, trial_number, params_hash, window, sharpe_net, n_trades, daily_returns_ref, timestamp`) más los usos de lockbox.
2. `S_global` = nº de trials del programa que tocaron la misma ventana. DSR y edge test se corrigen con `S_global` (y `N_eff_global`), no solo con el `S` del run. Una estrategia nueva sobre los mismos datos hereda la deuda estadística de las anteriores.
3. El ledger incluye las **variantes descartadas a mano** ("probé X y no funcionó"). Sin ellas, el conteo es optimista.
4. Cuando una ventana se agota (lockbox consumido para la familia), el único OOS fresco es el forward real (paper/live).

**Salida Etapa 0:** `runs/.../00_prep/{spec.yaml, master_copy.yaml, manifests/, data_check.html, budget.json, power.json}` + entrada en ledger + veredicto `READY` (o `INCONCLUSIVE` por potencia).

---

## Etapa 1 — Optimización con Optuna

**Objetivo:** generar `trials.parquet` + `daily_returns.parquet` completos (todos los trials — imprescindible para DSR/N_eff/PBO) y filtrarlos a `candidatos.csv`.

### 1.1 Diseño del objetivo (constraints del sampler, pruning real, objetivo robusto)

```python
# optimization/objective.py
def objective(trial, combo, master) -> float:
    cfg = master.optimization
    params = sample_search_space(trial, master)            # respeta type/step/choices
    seed = derive_seed(master.seed, trial.number)
    IS = master.universe.windows.IS

    # 1) Screening barato sobre el PREFIJO del IS.
    #    Un backtest event-driven sobre [start, mid] es prefijo exacto del completo:
    #    la poda usa solo información intra-IS y SÍ ahorra cómputo.
    if cfg.early_gate.enabled:
        s_fast = summary(*run_pyeventbt_once(params, combo, costs=master.costs,
                                             window=prefix(IS, cfg.early_gate.at_fraction), seed=seed))
        trial.set_user_attr("n_trades_fast", s_fast.n_trades)   # el trial podado conserva evidencia
        trial.report(s_fast.sharpe_net, step=0)
        if s_fast.n_trades < cfg.early_gate.min_trades_fast or trial.should_prune():
            raise optuna.TrialPruned()

    # 2) Backtest completo (mismo cash, costes, ventana y delay para todos los trials)
    pnl, trades = run_pyeventbt_once(params, combo, costs=master.costs, window=IS, seed=seed)
    s = summary(pnl, trades)
    daily = to_daily_returns(pnl)
    save_daily_returns(trial.number, daily)   # base de DSR, N_eff y CSCV/PBO

    # 3) Todo queda guardado, también si viola constraints
    for k in ("pf", "n_trades", "maxdd_pct", "exposure_pct", "r2"):
        trial.set_user_attr(k, getattr(s, k))
    trial.set_user_attr("sharpe_net_full", s.sharpe_net)

    # 4) Constraints: se devuelven al sampler (<= 0 es factible). NO TrialPruned.
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
    return s.sharpe_net

# Sampler (verificar soporte de constraints_func en la versión de Optuna instalada)
sampler = optuna.samplers.TPESampler(seed=master.seed, n_startup_trials=cfg.n_startup_random,
                                     constraints_func=lambda ft: ft.user_attrs["constraints"])
```

Detalles obligatorios:

1. **Primario `sharpe_net`** (no profit, no PF solo, no win-rate): anualizado sobre retornos diarios netos, `Rf=0`. Con `robust.enabled` se optimiza `mean(SR_k) − λ·std(SR_k)` sobre sub-ventanas contiguas: **Optuna busca mesetas estables en el tiempo**. El jitter ±10% de la Etapa 4 pasa a ser confirmación, no primer filtro.
2. **Backtest idéntico para todos los trials**: mismo `cash`, `costs`, `IS`, `delay_bars=1`, seed derivada. `params_class` pydantic + `filter_params` estricto que falla ante claves desconocidas o `NaN`.
3. **Constraints vía `constraints_func`, no `TrialPruned`**: podar tras el backtest completo no ahorra cómputo y destruye evidencia para el DSR. La poda real solo ocurre en el screening del prefijo.
4. **Cuándo podar:** si el backtest completo dura < ~10 s, desactivar `early_gate` y `pruner` (el overhead no compensa). Con backtests > 30 s, añadir `Hyperband`.
5. **Paralelo** `n_jobs=-1` a nivel trial. Prohibido paralelizar dentro del backtest y fuera a la vez.

### 1.2 Storage, determinismo honesto, filtrado

- **Storage:** SQLite no soporta bien workers concurrentes; con `n_jobs>1` usar PostgreSQL o `JournalStorage`.
- **Determinismo honesto:** con `n_jobs>1` y TPE el orden de finalización altera el muestreo; `master+seed` no reproduce bit a bit. Dos modos: `deterministic_mode: true` (1 worker; CI y regresión) y modo paralelo (reproducible estadísticamente). Lo que se audita es el **artefacto**: hash de `trials.parquet` y `daily_returns.parquet` en `promoted.csv`.
- Guardar **todos** los trials (también podados e infactibles, con columna `feasible` y `pruned_at`) **+ `daily_returns.parquet`** (matriz fechas × trials): base de DSR (varianza del Sharpe entre trials, skew, kurtosis, T), `N_eff`, clustering conductual y CSCV/PBO. Tirar trials malos del disco es destruir evidencia contra el overfitting.
- Filtro duro: `n_trades>=100`, `maxdd<=25%`, `exposure<=80%`, `pf>=1.0` (laxo). Sobre supervivientes: **Pareto** o **top-k** con `k = 3× max_clusters_kept` → `candidatos.csv`.
- Registrar `S` y `K` y volcarlos al **ledger global**.

**Salida Etapa 1:** `optimization/{<EST>_<PAIR>_<TF>.csv, trials.parquet, daily_returns.parquet, candidatos.csv, pareto.png}` + storage + `state: optimize DONE(total_trials=S, survivors=K)` + trials en ledger.

---

## Etapa 2 — Clustering DBSCAN/HDBSCAN de parámetros

**Objetivo:** convertir K candidatos correlacionados en ≤250 representantes genuinamente distintos. Se corrigen las tres debilidades de DMRI (escala sin normalizar, `eps` fijo, representante = moda frágil).

1. **Scaler robusto** (IQR/mediana): DMRI con `eps=30` sin normalizar mezcla peras con manzanas. Categorical one-hot; redondeo del representante al `step` válido con re-validación rápida de que el redondeo no cambia el signo.
2. **DBSCAN auto-`eps`** (curva k-distance, cuantil p90; registrar `eps_auto`) + opción **HDBSCAN** para densidades heterogéneas. `min_samples: 5`, `metric: manhattan`.
3. **Representante: medoide** (punto real que minimiza la distancia intra-cluster), no moda (la moda fabrica puntos inexistentes en valles entre picos). Alternativas: `best_sharpe`, `ensemble3`.
4. **`keep_singleton_outliers: true`**: outliers (`-1`) con `sharpe_net` en top-decila se conservan como singletons con escrutinio extra.
5. **`params_plus_behavior` (opcional):** concatenar params normalizados + vector de comportamiento (retornos mensuales, peso 0.3). Dos sets con params distintos pero misma forma de P&L acaban juntos. Reutiliza `daily_returns.parquet`.
6. **Ensemble por cluster:** validar `ensemble3` (medoide + 2 mejores) y exigir **≥2 de 3 verdes**. Un cluster cuyo mejor punto pasa pero sus vecinos fallan es un pico, no una meseta. Mide meseta local; **no es PBO** (el real está en 3B).
7. **N_eff (novedad, alimenta el DSR):** con TPE los trials se concentran en zonas buenas: `S=400` no son 400 hipótesis independientes. Agrupar trials por correlación de retornos (ONC / jerárquico sobre `1−ρ`) → `N_eff` = nº de clusters conductuales. `cluster/neff.json {S, N_eff, method}`. DSR se reporta con ambos extremos: `S` bruto (conservador) y `N_eff` (optimista); el veto usa `N_eff`, el ranking el conservador. Si la decisión cambia entre extremos, el set es ámbar.
8. Diagnósticos: `cluster/<...>.csv` + `diagnostics/{kdistance.png, sizes_hist.png, pca2d.png, members/}`.

---

## Etapa 3 — Validación sintética GJR-GARCH + control negativo + edge tests

### 3.1 Generador (port DMRI + endurecimientos)

1. `SyntheticMarketGenerator` (DMRI casi verbatim): `GARCH(1,1,1)-t` sobre log-retornos ×100, `fast_garch_simulation @njit` con término asimétrico, `generate_universe` con corrección de deriva, `build_ohlc_fast` con mechas por ratios reales.
2. Cache por `(símbolo,TF)` ajustado en IS; `to_csv()` M1 MT5 para que `run_pyeventbt_once` no distinga real de sintético.
3. Endurecimientos:
   - **Validación del generador**: el sintético debe replicar `volatilidad, autocorr(|r|), kurtosis, rango medio diario` dentro de ±20% del real. Generador que no replica es máquina de rechazos falsos.
   - **Stationary bootstrap** opcional (`p=0.1`): menos artefactos de borde que `block_size=10` puro.
   - **Spread sintético honesto**: `spread_medio_real × (1 + estrés)` en el re-scoring, no spread plano.
   - **Generador nulo** (`iid_residuals`): residuos remuestreados iid, sin clustering de volatilidad. Insumo del control negativo.

### 3.2 Protocolo de scoring (consistencia + control negativo + edge test)

**3.2a Consistencia (test DMRI, reinterpretado).**

1. Por representante: 1 backtest real (IS, `costs` maestro) + N sintéticos (`N=20` screening, `N=50` finalistas) con idénticos params/cash/costs.
2. `score_percentile_rank` por métrica (Sharpe, PF, #Trades), banda `[10,90]`, quorum 0.5, casos borde conservados (`NaN/inf→None`, varianza-cero→50).
3. **Interpretación correcta:** un GARCH con bootstrap de residuos es un **modelo nulo** (conserva la volatilidad, no la estructura predictiva). Por tanto:
   - Rank **bajo** (<10) → **FAIL**: en real rinde peor que en mercados plausibles; algo raro en datos o ejecución.
   - Rank **alto** (>90) → **ÁMBAR** (no veto): sospecha de overfitting *o* de edge real que el generador no captura. Lo resuelve el edge test (3.2c) y la Etapa 3B.
   - El juez del edge NO es el rank.

**3.2b Control negativo (calibra el backtester y la estrategia).**

1. Ejecutar el pool de representantes sobre el generador nulo (`n_sims: 200`).
2. A nivel de pool: Sharpe medio nulo ≈ 0 (`|media| ≤ 0.15`) y fracción de sets nulos "significativos" ≤ 10%. Si el pool gana donde no hay nada que explotar hay **sesgo** (look-ahead, costes subestimados, intrabarra optimista, fills demasiado favorables): **el run se detiene y se diagnostica**; ningún resultado posterior es fiable.
3. A nivel de set: Sharpe medio nulo positivo y significativo → vive de un artefacto → **FAIL**.

**3.2c Edge test sobre toda la familia de trials (data-snooping).**

1. Shortlist (top 40 por score entre no-FAIL). Con `daily_returns.parquet`: **White's Reality Check / Hansen SPA / Romano-Wolf stepdown** con stationary bootstrap (`≥ 2000` remuestreos) contra benchmark 0. H₀: *ningún* trial de la familia tiene edge. Alternativa: permutación (sign-flip de retornos diarios).
2. Salida: p-valor **ajustado por familia** (`edge_p_adj`), `alpha_fwer: 0.10`.
3. La familia de corrección es la del ledger (`S_global`), no solo el run.

**3.2d DSR/PSR con N_eff.**

1. PSR (con asimetría, curtosis y T reales) y **DSR** con `N_eff` de §2.7. Reportar con `S` bruto **y** `N_eff`.
2. **Veto:** `DSR(N_eff) < 0.20` → FAIL (provisional hasta Meta-etapa M).
3. **Haircut** por nº de tests (Harvey-Liu): reportar `Sharpe_haircut`; no veta solo, ordena la cola de finalistas.

**3.2e FDR bien planteado.**

1. Benjamini-Hochberg (`q: 0.10`) sobre los p-valores del **edge test**, no de ranks. *(Con 50 sims, un p derivado de rank ronda 1/(n+1) ≈ 0.02 y con C=250 BH exigiría ≈ 0.0004: inalcanzable — el FDR de v1 era decorativo.)*
2. **Check de resolución:** el p mínimo alcanzable del método debe ser ≤ `q/C`. Si falla, el pipeline **aborta con aviso**: subir remuestreos o reducir `C`.

**Veredicto por set:**

- **VERDE:** consistencia verde (o ámbar-alto) + control negativo OK + `edge_p_adj < alpha` + DSR OK + FDR OK + estrés OK.
- **ÁMBAR:** todo salvo uno de {estrés de costes, DSR entre extremos, rank alto sin edge concluyente}.
- **ROJO:** FAIL en consistencia baja, control negativo, edge test o DSR.
- Pasan a 3B los verdes + ámbares con `edge_p_adj < alpha`.

### 3.3 Coste, paralelismo, plots

- `n_sims` adaptativo: screening 20 para C representantes; 50 solo para finalistas. Rechazo temprano solo por rank **bajo** (<5 fuera de banda).
- Seeds por `(master.seed, cluster_id, sim_idx)`; `__getstate__/__setstate__` del generador para joblib.
- Plots: `real_vs_synth.png`, `deviation_distribution.png`.
- `synthetic_validation/<...>.csv`: `ranks, diagnostics, dsr_S, dsr_neff, edge_p_adj, fdr_adj_p, haircut, null_sharpe_mean, verdict`.
- **Re-scoring estresado:** re-evaluar los `passed` con `spread×2, +1 tick` sobre el ledger sintético (sin re-simular). `stress FAIL + base PASS` = ámbar con mitad de prioridad.

**Salida Etapa 3:** supervivientes + evidencia + `null_control/` + `edge_tests/`. **Abortos:** control negativo fallido o check de resolución FDR fallido ⇒ run detenido con diagnóstico.

---

## Etapa 3B — Walk-forward anclado, CPCV y PBO real

**Objetivo:** que la decisión no dependa de un único corte IS/FORWARD (que depende de dos años concretos).

1. **PBO real (CSCV): barato, sin re-optimizar.** Matriz `daily_returns` (fechas × trials) → particionar en 10 bloques contiguos → combinaciones mitades train/test (con purga/embargo) → en cada combinación, elegir el mejor trial en train y medir su rank en test. Salida: `PBO = P(rank_OOS < mediana)`, logits, degradación Sharpe IS→OOS. **Gate:** `PBO ≤ 0.30` sobre la familia del combo. PBO alto = elegir "el mejor del IS" es esencialmente azar.
2. **CPCV:** por superviviente, distribución de Sharpe en los caminos CPCV (purga = embargo). **Gate:** mediana > 0 y ≥70% de caminos positivos. Evalúa parámetros fijos (los parámetros ya vieron el IS); el OOS estricto es el punto siguiente.
3. **Walk-forward anclado con re-optimización (WFE):** train creciente, test de 12 meses, ~5 folds. En cada fold se ejecuta un **mini-pipeline completo** (Optuna reducido + cluster + selección) sobre train y se evalúa en test con embargo. `WFE = Sharpe_test / Sharpe_train`. **Gate:** `WFE ≥ 0.5` y ≥70% de folds positivos. Mide el **procedimiento**, no un set: si falla, el problema está en el espacio de búsqueda o en la estrategia. Coste: `n_folds × n_trials_per_fold`, una vez por combo. Todos los trials de los folds cuentan en el ledger.
4. Salidas: `wf_cpcv/<...>.csv` + `lambda_logits.png` + `oos_degradation.png`. **Regla de aborto:** PBO o WFE fuera de gate ⇒ combo ámbar/rojo; no se fuerza la Etapa 4.

---

## Etapa 4 — Decorrelación, estrés de ejecución, puertas y lockbox

**Orden crítico (corrección de v1):** 4.1–4.4 selección → 4.5 jitter → 4.6 replay con ticks → 4.7 **puertas** → 4.8 **lockbox** → 4.9 sizing. *(En v1 las puertas se aplicaban después de consumir el FORWARD: un set que fallaba una puerta ya había gastado el lockbox en vano.)*

1. **Recomputar curvas en real con idénticos costes** (paridad con Etapa 3). Derivar `n_trades, pf, r2, sharpe_net, calmar, turnover` con `resample: D`. Prefiltro: `n_trades < 30` fuera; varianza cero fuera.
2. **Correlación sobre retornos (nunca niveles):** diaria, pearson|spearman. **Unilateral**: `rho <= 0.7` contra lo ya aceptado; `rho` negativo = diversificador, se queda. Opción `spearman` para colas pesadas (FX).
3. **Score compuesto:**
   ```text
   score = 0.40·norm(sharpe_net) + 0.30·norm(pf_winsorizado p95)
         + 0.20·norm(r2) − 0.10·norm(turnover)
   ```
   Winsorizar `pf=inf` al p95. `turnover_penalty`: a igual Sharpe, gana el que menos opera. Normalización min-max por combo (no global).
4. **Selección greedy/cluster:** aceptar si `max(rho vs aceptados) <= thr`, hasta `max_sets: 5` (no 10: 10 sets del mismo EA/TF es ilusión de diversificación, mismo riesgo base). Reportar `n_efectivo = 1/sum(w²)`; si `< 0.6·k`, aviso ámbar.
5. **Jitter test (confirmación):** ±10% por parámetro, ≥80% vecinos con `sharpe_net>0, pf≥1.0`. El que falla es pico: se sustituye por el siguiente del ranking.
6. **Replay con ticks y estrés de ejecución (sobre IS, antes de lockbox):** un backtest por barras sobreestima (ignora orden SL/TP intrabarra, gaps, spread real). Re-ejecutar los k con **ticks del broker** (`intrabar_order: worst_case`). **Grid de estrés:** `spread ×{1, 1.5, 2}` × `slippage +{0,1,2}` × `delay {1,2}`. Pasa si PF en la peor celda ≥ 1.05 y decaimiento de Sharpe ≤ 50%. Si PF barra-vs-ticks difiere >15%, el backtest por barras no es fiable para ese set (documentarlo; ticks como referencia).
7. **Puertas (antes del lockbox):** `PF≥1.3` (o 1.1–1.3 + retest 3 salidas MA/BB/RSI ≥1.3), `R2≥0.5`, `DD<365d`, `≤10/12` meses rojos. **Análisis por régimen:** desglose por año, terciles de volatilidad y sesión; exigir ≥60% de celdas con Sharpe>0 y que ninguna celda concentre >50% del PnL (dependencia de un único régimen). Un set que falla puertas se descarta aquí: **solo los que ya serían promovidos ven el lockbox.**
8. **FORWARD lockbox (una vez, sin reintentos, con multiplicidad):** solo los que pasaron puertas: `pf≥1.1` + `sharpe_net>0`. Sin re-optimizar ni segundo intento. **Multiplicidad:** con 3 símbolos × 2 TF × k=5 pueden ser 30 usos correlacionados (factor USD) sobre la misma ventana; cada uso se registra y el criterio se evalúa con **PSR corregido por Holm** sobre los usos del programa. `min_pf` es filtro operativo, no evidencia estadística. **Limitación explícita:** dos años son un solo régimen; una vez usado, FORWARD queda quemado para esa familia. Test de integridad en CI: acceso previo al lockbox invalida el run.
9. **Promoción:** `Lots = min(Lots_dd, Lots_max_los, Lots_mean_loss)`; `SL/TP` por percentil MFE/MAE (p10/p90) o ATR — **nunca min/max extremos**. `promoted.csv`: `params + gates + ranks + dsr(S,N_eff) + edge_p_adj + fdr + pbo/wfe + jitter + peor celda de estrés + régimen + forward(Holm) + lots + SL/TP + cost_cfg + manifest_hashes + master_hash + ledger_snapshot_hash + commit + magic`.

---

## Meta-etapa M — Calibración del pipeline (nulos + edge plantado)

**Objetivo:** medir el comportamiento del **pipeline entero**: ¿cuántas estrategias sin edge llegan a `promoted.csv`? ¿cuántas con edge conocido sobreviven? Sin esto, `DSR 0.20`, `q=0.10`, `PBO 0.30` o `WFE 0.5` son opiniones.

1. **Estrategias nulas → FPR:** 200 estrategias **sin edge por construcción** (entradas aleatorias con la misma frecuencia/holding, señales barajadas, indicador sobre ruido). Mismo search_space, mismo presupuesto, mismo pipeline. Datos: sintéticos iid y reales con señal destruida (permutación por bloques). Métrica: **tasa de promoción de nulas** ≤ 5%. Diagnóstico: cuántas nulas sobreviven a cada etapa (muestra qué filtros protegen de verdad y cuáles son decorativos).
2. **Edge plantado → potencia:** inyectar drift condicionado a la señal para Sharpe objetivo `{0.3, 0.6, 1.0}`, varias seeds. Objetivo: potencia a SR=1 ≥ 70%. Potencia baja = filtros demasiado duros para la longitud de datos (conecta con MinTRL).
3. **Ajuste de umbrales:** solo con estas curvas (FPR/potencia), **nunca mirando estrategias reales**. Congelados en `calibration_report.json` referenciado por `master_hash`.
4. **CI:** calibración reducida (n pequeño) para detectar que un cambio de código altera la FPR. Salidas: `calibration/{fpr_by_stage.csv, power_curve.png, calibration_report.json}` a nivel de programa.

---

## Etapa 5 — Portfolio: asignación, exposición y riesgo de cartera

**Objetivo:** la unidad de decisión no es un set, es la **cartera**. 5 sets del mismo EA son el mismo riesgo base.

1. **Universo y correlación robusta:** todos los `promoted.csv` aprobados (estrategias/símbolos/TF distintos), retornos diarios netos. Correlación **media y en estrés** (peor decil de días) + clustering jerárquico de estrategias. La decorrelación se exige **entre estrategias**, no solo dentro de un combo.
2. **Exposición por divisa:** descomponer cada posición en exposición neta por divisa (EURUSD largo = +EUR −USD) y limitar `max_currency_net_exposure`. Evita que cinco pares "distintos" sean la misma apuesta corto USD.
3. **Asignación:** **HRP o risk parity** sobre covarianza shrinkage (Ledoit-Wolf), **vol targeting** (`vol_target_annual`), **Kelly fraccional** (cap 0.25) como techo, no objetivo. Peso máximo por estrategia; rebalanceo mensual con banda.
4. **Límites y envolvente:** DD máximo de cartera, apalancamiento, nº de posiciones, margen. Monte Carlo de bloques → **envolvente de DD (p95/p99)**, que es el límite que usa la monitorización (Etapa 6). Salidas del MC: `P(DD>X)`, `P(pérdida anual>X)`, expected shortfall, máximo periodo perdedor, recovery time.
5. **Capacity (ligero):** curva `capital → Sharpe/decaimiento` vía grid de estrés de costes (`{10k, 50k, 250k, 1M}`). Una estrategia con Sharpe alto pero capacidad de €20k no se presenta como de €1M.
6. **Regla de admisión:** añadir una estrategia exige **beneficio marginal** (Sharpe/DD de cartera neto de costes). "Más estrategias" no es mejor por sí mismo.

---

## Etapa 6 — Live: paridad, despliegue escalonado, monitorización y retirada

**Objetivo:** que lo promovido se comporte en producción como en el backtest, y detectar rápido si deja de hacerlo. Nada va con dinero real sin 6.1–6.5 operativos.

1. **Paridad backtest ↔ live (shadow):** replay diario — los datos reales del día pasan por el backtester con los mismos params y se compara señal a señal y fill a fill con el live. Métricas: % señales discordantes, slippage real vs modelado (p95), tracking error de equity. Verificar divergencias reales: horario del servidor, rechazos/requotes, sufijos de símbolos, cambios de especificación (tick size, lot step, swaps), cierres de mercado.
2. **Despliegue escalonado con criterios pre-fijados:** `paper` (≥60 días y ≥30 trades) → `micro` (×0.1) → `scaled` (×0.5) → `full`. Avance solo con: paridad OK, Sharpe/DD dentro de envolvente, cero incidentes de ejecución. Versiones nuevas: **champion/challenger**. Nunca params en caliente.
3. **Monitorización y decay (health score):** Sharpe rodante vs distribución esperada del backtest/CPCV (alerta bajo p5); **DD en vivo vs envolvente MC** (alerta p95, parada p99); **CUSUM/SPRT** sobre retornos por trade; deriva de frecuencia, duración media, hit rate, slippage, spread, distribución de retornos (KS). Health score = Performance + Execution + Statistical + Regime: `GREEN/YELLOW/ORANGE/RED`.
4. **Límites de riesgo y kill switches (en código, no solo monitorización):** jerarquía por trade/día/estrategia/cartera + operacionales (heartbeat con MT5, datos stale, spread máximo, conexión). Acciones explícitas: `halt` (no nuevas entradas) y `flatten` (cerrar) + kill switch manual. **La capa de riesgo es independiente de la lógica de la estrategia**: no se confía en que la estrategia se autolimite.
5. **Reconciliación e idempotencia:** cada 60 s comparar posiciones/órdenes/equity internos vs broker; discrepancia ⇒ `halt_and_alert`. Órdenes **idempotentes** (magic + comentario único; verificar antes de reenviar tras timeout) para evitar duplicados. Persistir estado para reiniciar sin perder contexto.
6. **Retirada y re-optimización:** regla de retirada **pre-registrada**: DD > envolvente p99, 2 alarmas CUSUM, o Sharpe rodante < p5 durante 30 días ⇒ pasa a paper/shadow; no se "arregla" en caliente. Toda re-optimización es un **run nuevo completo** con los mismos gates y consumo de ledger.
7. **Infraestructura MT5:** VPS con reinicio automático, NTP, logging inmutable, backups de estado; verificar soporte de plataforma de la librería MT5 de PyEventBT antes de diseñar el despliegue.

---

## Orquestación: stages, estado reanudable, artefactos

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
  portfolio/{weights.csv, exposure_by_currency.csv, mc_dd_envelope.csv, capacity.csv, ...}

research/                            # A NIVEL DE PROGRAMA (compartido por todos los runs)
  ledger.parquet                     # append-only
  lockbox_uses.yaml
  calibration_report.json + calibration/

live/{parity/, monitoring/, state/, logs/}    # servicios de producción, fuera del pipeline de investigación
```

1. `Stage` base (`name, discover_inputs, run(pair,timeframe), persist, _update_state`) + `StageContext{config, state, out_dir, ledger}`. Stages de investigación: `OptimizeOptuna → ClusterParams → ValidateSynthetic(+edge) → WalkForwardCPCV → DecorrelateSelect(+execution_realism, gates) → Lockbox → Promote → BuildPortfolio`. La Etapa 6 son servicios, no stages.
2. `ProcessLog` atómico (`tmp+os.replace`), `init_stage/mark_done/mark_error` con contadores. Re-lanzar salta `done` cuyo `config_hash` coincide; si el maestro cambió, solo re-ejecuta lo afectado (trials conservados si `search_space` no cambió).
3. Dashboard Rich (`strategy/asset/step/progress` + logs rodantes; sin `print` en pipeline).
4. **Reglas de aborto:** `passed==0` tras sintético, control negativo fallido, check de resolución FDR fallido, PBO/WFE fuera de gate o acceso prematuro al lockbox ⇒ run termina en ámbar/rojo con diagnóstico. Nunca se fuerza la etapa siguiente sobre input vacío o inválido.
5. **Reproducibilidad:** Docker con digest fijado + lockfile, datos con DVC, tracking con MLflow, **CI con run de regresión** (`deterministic_mode`, `n_trials=20`, métricas esperadas versionadas). **Behavioural regression:** un cambio que mejora las métricas del run de referencia demasiado (p.ej. Sharpe 1.8→2.4) levanta alerta y exige explicación — puede ser un bug o leakage, no una mejora.

---

## Gobernanza: ciclo de vida del strategy registry (de CHATGPT, condensado)

```text
RESEARCH → CANDIDATE → VALIDATED → APPROVED → PAPER → SHADOW → MICRO_LIVE → LIVE
                                                            ├── DEGRADED
                                                            ├── QUARANTINED
                                                            └── RETIRED
```

Ninguna estrategia cambia de estado sin el gate correspondiente. Registro versionado por estrategia: `strategy_id, version, status, spec_hash, code_commit, data_manifest_hash, research_lineage, validation_report, portfolio_assignment, risk_budget, execution_profile, approved_by/at, last_health_check`. Un cambio de lógica/features/universo/costes crea **nueva versión y nueva genealogía**, nunca una edición silenciosa.

**Strategy Approval Package** (artefacto de promoción final, con hashes):

```text
approval_package/
├── strategy_spec.yaml + master_used.yaml + data_manifests/
├── research_ledger.json + trials.parquet + clustering.parquet
├── synthetic_validation.parquet + cpcv_report.json + pbo_report.json + dsr_report.json
├── walkforward.parquet + monte_carlo.parquet + stress_report.json
├── portfolio_report.json + capacity_report.json + execution_report.json
├── forward_lockbox_report.json + promoted.csv + risk_limits.yaml
└── approval.json + hashes.json
```

> **Criterio final:** si una tercera persona no puede reconstruir por qué una estrategia fue seleccionada, con qué datos, qué alternativas se probaron, qué costes se asumieron, qué controles superó y qué versión exacta se desplegó, el sistema todavía no está al nivel objetivo.

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
  analytics/{stats.py, tearsheet.py, regimes.py}   # summary(): sharpe_net/pf/r2/n_trades/turnover
  selection/{gates.py, lockbox.py, lots.py, sl_tp.py, promote.py}
  portfolio/{allocation.py, exposure.py, stress_corr.py, mc_envelope.py, capacity.py}
  live/{parity.py, monitor.py, risk_limits.py, reconcile.py, rollout.py, retire.py}
  research/{ledger.py, registry.py}
  runner/{stage.py, process_log.py, master_config.py, dashboard/}
pipeline_master.yaml
bots_dispatcher_optuna.py                     # entry: python bots_dispatcher_optuna.py pipeline_master.yaml
```

CLI:

```bash
pip install optuna arch joblib psycopg2-binary mlflow dvc   # + pyeventbt[opt]
python bots_dispatcher_optuna.py pipeline_master.yaml            # pipeline de investigación completo
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

Tests que bloquean merge:

- **Núcleo:** `test_scoring` (ranks incl. bordes), `test_cluster` (medoide vs moda, eps auto), `test_params_filter` (coerción/rechazo), `test_synthetic_determinism` (misma seed → mismo universo), `test_decorr_unilateral` (negativos se quedan), `test_lockbox_integrity` (acceso temprano = fail), `test_optuna_resume` (20+20 == 40, `deterministic_mode`).
- **Estadística v2:** `test_fdr_resolution` (aborta si `p_min > q/C`), `test_embargo_time` (≥ `max_holding + warm-up` por TF), `test_gate_order` (puertas antes del lockbox), `test_null_control` (el backtester no gana en datos nulos), `test_cscv_pbo` (caso sintético con PBO conocido), `test_neff`, `test_ledger_append_only`, `test_constraints_func` (infactibles conservan `user_attrs`), `test_calibration_smoke`, `test_regression_run`.
- **Leakage firewall:** señal sobre barra no cerrada, timestamps desordenados, barras incompletas → FAIL.
- **Live:** `test_live_risk_limits` (cada kill switch dispara en su condición), `test_live_reconcile` (discrepancia ⇒ `halt_and_alert`), `test_order_idempotency` (reintento tras timeout no duplica).

---

## Gates numéricos y criterios de aceptación

Defaults (el maestro los fija; endurecer por TF/símbolo, nunca ablandar sin registro y sin recalibrar). `(provisional)` = pendiente de Meta-etapa M.

| Gate | Umbral | Fuente |
|---|---|---|
| `min_trades_IS` por trial | ≥100 | DMRI + playbook |
| Potencia | `años_IS ≥ MinTRL(target_sharpe)`; si no, "no concluyente" | Nuevo |
| Embargo | ≥ `max_holding + warm-up`, en tiempo | Nuevo |
| Leakage firewall | Cualquier violación = FAIL | Nuevo |
| Constraints optuna | `maxdd≤25%`, `exposure≤80%`, `pf≥1.0` vía `constraints_func` | Nuevo |
| Cluster | ≤250 representantes, medoide, singletons top-decila | DMRI |
| Control negativo | `|Sharpe medio nulo| ≤ 0.15`; nulos "significativos" ≤ 10% `(provisional)` | Nuevo |
| Consistencia sintética | rank en [10,90], quorum 0.5; rank bajo = FAIL, rank alto = ÁMBAR | DMRI reinterpretado |
| Edge test | `edge_p_adj < 0.10` (SPA/RC/Romano-Wolf sobre `S_global`) | Nuevo |
| DSR veto | `DSR(N_eff) < 0.20` → FAIL; reportar también con S bruto `(provisional)` | Nuevo |
| FDR | q=0.10 BH sobre p del edge test; `p_min ≤ q/C` o aborta | Nuevo |
| Estrés de costes sintético | quorum verde con `spread×2 +1 tick`; si no, ámbar | Nuevo |
| PBO (CSCV) | ≤ 0.30 `(provisional)` | Nuevo |
| WF con re-opt | `WFE ≥ 0.5` y ≥70% folds positivos `(provisional)` | Nuevo |
| CPCV | mediana Sharpe > 0 y ≥70% caminos positivos | Nuevo |
| Decorrelación | `rho≤0.7` retornos D, `max_sets=5`, `min_trades=30` | DMRI (elite) |
| Jitter | ≥80% vecinos ±10% con `sharpe>0, pf≥1.0` | Nuevo |
| Estrés de ejecución | PF peor celda ≥ 1.05; decay Sharpe ≤ 50%; gap barra-vs-tick ≤ 15% `(provisional)` | Nuevo |
| Puertas (**antes** del lockbox) | `PF≥1.3` (o 1.1–1.3 + retest ≥1.3), `R2≥0.5`, `DD<365d`, `≤10/12` rojos; ≥60% celdas de régimen positivas; ninguna celda >50% del PnL | DMRI + nuevo |
| FORWARD lockbox | `pf≥1.1`, `sharpe_net>0`, un intento, PSR corregido por Holm | DMRI + nuevo |
| Sizing/SLTP | `Lots=min(...)`, SL/TP percentil/ATR | DMRI |
| Calibración (M) | FPR ≤ 5%; potencia a SR=1 ≥ 70% | Nuevo |
| Portfolio | DD cartera ≤ límite; exposición por divisa ≤ límite; beneficio marginal positivo | Nuevo |
| Capacity | decaimiento de Sharpe ≤ 25% en el grid de capital | Nuevo |
| Paridad live | ≤2% señales discordantes; slippage p95 ≤ 2 ticks; tracking error ≤ 0.5% | Nuevo |
| Rollout | paper ≥60d y ≥30 trades → micro ≥60d → escalado ≥90d → pleno | Nuevo |

**Definición de DONE de investigación:** `promoted.csv` con k filas, cada una con `ranks + control negativo + edge_p_adj + dsr(S, N_eff) + fdr + pbo/wfe + jitter + estrés de ejecución + régimen + forward (Holm)` trazables + `master_hash + manifest_hashes + ledger_snapshot_hash + commit`, **y** `calibration_report.json` vigente que respalda los umbrales. Menos que eso es un informe, no una selección.

**Definición de DONE de producción:** cartera construida (Etapa 5), paridad verificada, kill switches y reconciliación probados, rollout completado según criterios pre-fijados y monitorización con envolvente activa (Etapa 6).

---

## Prioridades de implementación

| Prioridad | Qué | Por qué primero |
|---|---|---|
| **P0** | `run_pyeventbt_once` + scoring + sintético + objetivo Optuna con `constraints_func` y retornos guardados. Corregir desde el día 1: semántica del test sintético (§3.2), FDR con check de resolución, orden puertas → lockbox, embargo en tiempo | Son errores de diseño, no de implementación: costan más cuanto más tarde se corrijan |
| **P1** | `daily_returns` + `N_eff` + DSR correcto + WF/CPCV/PBO + potencia/MinTRL + **Meta-etapa M** (calibración con 20–50 nulas) + ledger global | Sin esto no se sabe cuánto de lo que sobrevive es edge |
| **P2** | Realismo de ejecución (ticks, grid de estrés), análisis por régimen, capa de portfolio (HRP, exposición por divisa, envolvente MC, capacity) | Cierra la brecha backtest → ejecución real y evita sobreestimar la diversificación |
| **P3** | Capa live completa (paridad, rollout, monitorización, kill switches, reconciliación). Puede desarrollarse en paralelo, pero **nada va con dinero real sin paridad y kill switches** | Es donde se pierde dinero de verdad |
| **Defer** | Simulador institucional completo (market impact/queue), data layer multi-capas, event store completo, multi-estrategia distribuida | No aplican a la escala actual; no bloquean nada de lo anterior |

---

## Nota realista

Este pipeline reduce los falsos positivos, pero **no crea edge**. Lo que distingue a los mejores fondos no es solo el control del overfitting, sino hipótesis con fundamento económico, muchas fuentes de alfa descorrelacionadas, ejecución y gestión de riesgo. La `spec.yaml` pre-registrada con hipótesis y razón de ser es la parte más valiosa de este documento. El éxito se mide con el paper/live a largo plazo y con la calibración del propio pipeline (FPR y potencia), no con cuántos sets sobreviven.

---

*Primera acción: copiar el esquema del maestro a `pipeline_master.yaml`, implementar `run_pyeventbt_once + scoring + synthetic + objective` (P0) y pasar UNA estrategia × UN combo (ej. `EURUSD H1, n_trials=100, n_sim=20`) de punta a punta, incluyendo control negativo. En paralelo, montar una calibración mínima (Meta-etapa M) con 20–50 estrategias nulas para ver qué filtros protegen de verdad. El resto es escalar lo que ya funciona, no construir más teoría.*
