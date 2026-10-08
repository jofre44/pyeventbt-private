# Pipeline Robusto v2: Optimización → Validación → Portfolio → Ejecución → Live

**Documento base:** Pipeline Robusto: Optimización → Clustering → Validación Sintética → Decorrelación
## Réplica mejorada del proceso DMRI_PyMT4 en PyEventBT, con config maestro y anti-overfitting state-of-the-art

> Para **estrategia ya definida** (lógica cerrada, espacio de parámetros acotado). No data mining.
> Objetivo: **seleccionar parámetros con edge en datos futuros**, no los que mejor explican el pasado.
> Método: mismo esqueleto que `DMRI_PyMT4` (`optimize → cluster → validate_on_synthetic → decorrelate`), pero con motor **Optuna + PyEventBT** en vez de MT4 genético + `backtesting.py`, y con capas SOTA anti-overfit que DMRI no tiene (presupuesto de trials, haircut por nº de tests, DSR, PBO-lite, purged/embargo, FDR, ensemble por cluster).
> Todo se gobierna desde **un único archivo maestro** (`pipeline_master.yaml`). Nada hardcodeado en `.py`.

---

## Índice

1. [Arquitectura del pipeline (qué se conserva de DMRI y qué cambia)](#1-arquitectura-del-pipeline-qué-se-conserva-de-dmri-y-qué-cambia)
2. [Archivo maestro: `pipeline_master.yaml` (todo configurable)](#2-archivo-maestro-pipeline_masteryaml-todo-configurable)
3. [Etapa 0 — Preparación: spec, espacio de búsqueda, datos, presupuesto](#etapa-0--preparación-spec-espacio-de-búsqueda-datos-presupuesto)
4. [Etapa 1 — Optimización con Optuna (conservar solo lo que cumple criterios)](#etapa-1--optimización-con-optuna-conservar-solo-lo-que-cumple-criterios)
5. [Etapa 2 — Clustering DBSCAN/HDBSCAN de parámetros](#etapa-2--clustering-dbscan-hdbscan-de-parámetros)
6. [Etapa 3 — Validación en datos sintéticos GJR-GARCH](#etapa-3--validación-en-datos-sintéticos-gjr-garch)
7. [Etapa 4 — Decorrelación y selección final de parámetros](#etapa-4--decorrelación-y-selección-final-de-parámetros)
8. [Orquestación: stages, estado reanudable, artefactos, dashboard](#orquestación-stages-estado-reanudable-artefactos-dashboard)
9. [Tabla DMRI → PyEventBT: qué se reutiliza y qué se mejora (SOTA)](#tabla-dmri--pyeventbt-qué-se-reutiliza-y-qué-se-mejora-sota)
10. [Implementación en PyEventBT: módulos y CLI](#implementación-en-pyeventbt-módulos-y-cli)
11. [Gates numéricos y criterios de aceptación](#gates-numéricos-y-criterios-de-aceptación)

---

## 1. Arquitectura del pipeline (qué se conserva de DMRI y qué cambia)

```text
pipeline_master.yaml
        │
        ▼
[0 PREP] spec + search_space + DATA_MANIFEST + cost_cfg + seed + trial_budget
        │  optimization/<EST>_<PAIR>_<TF>.csv (todos los trials, no solo ganadores)
        ▼
[1 OPTUNA] ──IS purged/embargo──▶ trials.parquet + filtrado duro ──▶ candidatos.csv
        │  (TPE/NSGA-II, pruner, constraints, multi-objetivo, n_trials fijo)
        ▼
[2 CLUSTER] DBSCAN/HDBSCAN sobre params normalizados ──▶ cluster/<...>.csv (1 fila = 1 representante)
        │  (medoide por cluster + outliers buenos como singletons)
        ▼
[3 SYNTH] GJR-GARCH(1,1,1-t) + block-bootstrap ──▶ synthetic_validation/<...>.csv (passed + ranks)
        │  (percentile-rank + DSR + FDR, n=20 screening → 50 finalistas)
        ▼
[4 DECORR] retornos diarios, greedy/cluster, rho<=thr ──▶ decorrelation/<...>.csv (k finalistas)
        │  (+ chequeo FORWARD lockbox solo al final + stability ±10%)
        ▼
   promoted.csv (params + gates + lots + SL/TP + cost_cfg + hashes + commit)
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

**Lo que DMRI no tiene y aquí es obligatorio:** presupuesto de trials declarado, corrección por múltiples tests, DSR/PBO-lite, embargo temporal, FDR, generador validado, forward lockbox intocable hasta el final, ensemble por cluster en vez de "el mejor punto".

---

## 2. Archivo maestro: `pipeline_master.yaml` (todo configurable)

Un solo archivo. Versionado en git. Cada `runs/` copia el master usado + `git hash`. Ningún `.py` lee constantes mágicas.

```yaml
# pipeline_master.yaml — MAESTRO. Todo lo configurable vive aquí.
pipeline_version: 1.0
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
  data_dir: data/m1                            # M1 formato MT5
  windows: {IS: [2015-01-01, 2021-12-31], FORWARD: [2022-01-01, 2023-12-31]}
  embargo_bars: 50                            # barras de embargo tras IS (anti-leak por autocorrelación)
  min_trades_IS: 100

costs:                                        # CONGELADO en todo el pipeline
  commission_per_side_per_lot: 2.5
  spread_mult: 1.0
  slippage_ticks: 1
  swap_mode: from_yaml
  delay_bars: 1

optimization:                                 # ETAPA 1
  engine: optuna
  direction: maximize
  n_trials: 400                               # PRESUPUESTO: fijo antes de ver resultados
  n_startup_random: 40
  sampler: TPE                                 # o NSGAII si multiobjetivo
  pruner: median                                # + Hyperband opcional
  n_jobs: -1
  storage: sqlite:///runs/optuna.db            # resume real
  objectives:                                  # primario + restricciones
    primary: sharpe_net
    constraints:
      min_trades: 100
      max_maxdd_pct: 25.0
      min_pf: 1.0                              # filtro laxo aquí; el duro (1.3) va en Etapa 4/puertas
      max_exposure_pct: 80.0
  multiobjective:                              # opcional; si se activa, sampler NSGAII
    enabled: false
    names: [sharpe_net, "-maxdd_pct"]
  early_gate:                                  # poda barata antes del backtest completo
    min_trades_fast: 30

cluster:                                      # ETAPA 2 (detalle en §5)
  enabled: true
  method: dbscan                               # dbscan | hdbscan
  feature_set: params_only                     # o params_plus_behavior (ver §5.4)
  scaler: robust                               # robust (IQR) | standard
  eps: auto                                    # auto (k-distance p90) | float
  eps_quantile: 0.90
  min_samples: 5
  metric: manhattan
  max_clusters_kept: 250                       # == DMRI cluster_sets
  representative: medoid                       # medoid | best_sharpe | ensemble3
  keep_singleton_outliers: true                # outliers buenos no se tiran

validation:                                   # ETAPA 3 (detalle en §6)
  n_sim_screening: 20
  n_sim_final: 50
  method: block
  block_size: 10                               # + stationary: {enabled, p} ver §6.3
  stationary_bootstrap: {enabled: false, p: 0.1}
  percentile_low: 10.0
  percentile_high: 90.0
  min_metrics_pass: 0.5
  metrics:
    - {name: Sharpe Ratio, type: stat}
    - {name: Profit Factor, type: stat}
    - {name: "# Trades", type: stat}
  dsr:
    enabled: true
    benchmark_sr: 0.0
    veto_below: 0.20                           # DSR < 0.20 → FAIL aunque ranks pasen
  fdr: {enabled: true, q: 0.10}                # Benjamini-Hochberg sobre p-valores de ranks
  synth_stress: {spread_mult: 2.0, slippage_ticks_add: 1}  # re-scoring estresado (ver §6.6)

decorrelation:                                # ETAPA 4 (detalle en §7)
  method: greedy                               # greedy | cluster
  correlation_threshold: 0.7
  max_sets: 5                                  # finalistas por (símbolo,TF) — portfolio pequeño y elite
  min_trades: 30
  resample: D
  corr_method: pearson                         # pearson | spearman
  weights: {sharpe_net: 0.4, profit_factor: 0.3, r2: 0.2, turnover_penalty: 0.1}
  pf_cap_percentile: 95.0
  forward_check: true                          # solo al final, una vez, sobre FORWARD lockbox
  forward_min_pf: 1.1
  stability_check: {enabled: true, jitter_pct: 10, min_green: 0.8}

promotion:
  lots: {max_dd_allow: 1200, max_trade_lost_allow: 200, mean_trade_lost_allow: 100}
  sl_tp: {metodo: percentil_MFE_MAE, p_sl: 10, p_tp: 90}
  magic_start: 11313

runtime:
  robot_folder: bots_folder/{strategy}
  out_dir: runs/{strategy}/{timestamp}_{githash}
  resume: true
```

Reglas del maestro:

1. **Cambiar el maestro = nuevo run.** Nunca se edita a mitad de pipeline y se continúa; se relanza con `resume: true` (reutiliza trials/artefactos cuyo hash de config coincide).
2. **`costs`, `windows`, `seed`, `n_trials` son inmutables dentro de un run.** El `ProcessLog` los hashea al inicio y aborta si cambian.
3. **`search_space` solo contiene lo que la spec declara optimizable.** Añadir un parámetro a posteriori = nueva hipótesis = nuevo run + justificación en `cambios.md`.
4. Todo umbral de veto (`min_trades`, `veto_below`, `q`, `forward_min_pf`) vive aquí, no en código.

---

## Etapa 0 — Preparación: spec, espacio de búsqueda, datos, presupuesto

**Objetivo:** que la optimización busque en el sitio correcto, con datos correctos y con un presupuesto declarado. Sin esto, Optuna es una máquina de overfitting rápida.

### 0.1 Congelar spec + search_space (de `PLAYBOOK` Fase 0)

1. `strategies/<est>/spec.yaml` cerrada: hipótesis, universo, sesión, entrada sobre **barras cerradas**, salidas (SL siempre o time-stop declarado), filtros (spread_max, ATR, noticias, `MaxOrdenes=1`), invalidez.
2. `search_space` del maestro con **tipos y pasos válidos** (el adaptador valida `step` contra `volume_step/tick_size` donde aplique). Parámetros de microstructure (spread, slippage) **nunca** en el espacio.
3. `frozen` para todo lo estructural. Si hay duda entre "buscar o fijar", fijar. Cada dimensión extra multiplica los trials necesarios y el haircut SOTA (§4.7).

### 0.2 Datos + manifiesto + embargo

1. Normalizar a M1 MT5 + `DATA_MANIFEST.json` por símbolo (`fuente, filas, gaps, spread_medio/p95, hash`). Reutilizar `runner/data_manifest.py + data_check.py` del playbook.
2. Ventanas del maestro: `IS` para optimizar, `FORWARD` **lockbox** (no se toca hasta la Etapa 4.6). `embargo_bars: 50` entre IS y FORWARD para que la autocorrelación serial no filtre información (López de Prado: purged + embargo; DMRI no lo hace → aquí es obligatorio).
3. Chequeo `min_trades_IS` con parámetros nominales: si el nominal no da ~100 trades en IS, el espacio/TF es ilíquido para este pipeline (achicar TF o ampliar ventana, nunca bajar el umbral).

### 0.3 Presupuesto de trials + determinismo (novedad SOTA vs DMRI)

1. Fijar `n_trials` **antes** de optimizar (ej. 400) en función de dimensiones: regla práctica `50–100 trials por dimensión efectiva`, capado por coste computacional. Registrarlo: el haircut por nº de tests (§4.7) lo necesita.
2. `seed` única → deriva determinista por trial (`seed + trial_number`). `n_startup_random` para exploración inicial. `storage` RDB para resume real (DMRI resume por CSVs; aquí por trials + artefactos con hash).
3. Coste estimado: `n_trials × coste_backtest_IS × nº_combos`. Si excede presupuesto, reducir espacio (no `n_sim` sintético, no `min_trades`).

**Salida Etapa 0:** `runs/.../00_prep/{spec.yaml, master_copy.yaml, manifests/, data_check.html, budget.json}` + veredicto `READY`.

**Qué se toma de DMRI:** `TIMEFRAME_DICT`, convención `<EST>_<PAIR>_<TF>.csv`, `ProcessLog` por combo, `historical.load_historical_data` (adaptado a M1 MT5). **Qué se añade SOTA:** embargo, presupuesto declarado, hash de config, determinismo por trial.

---

## Etapa 1 — Optimización con Optuna (conservar solo lo que cumple criterios)

**Objetivo:** generar `trials.parquet` completo (todos los trials, no solo ganadores — imprescindible para DSR/haircut) y filtrarlo a `candidatos.csv` con criterios duros. DMRI delegaba la búsqueda al genético de MT4 (`metrics_to_test 0..8`); aquí la búsqueda es **Optuna sobre PyEventBT** con el mismo rol pero auditable y reanudable.

### 1.1 Por qué Optuna y no grid ni genético MT4

- Grid explota combinatoriamente y desperdicia el 90% del cómputo en zonas malas. Genético MT4 no es portable a MT5/PyEventBT ni deja `trials` auditables.
- Optuna (TPE para objetivo único, NSGA-II para multiobjetivo) concentra trials donde hay señal, soporta **pruning** (mata trials malos a mitad del IS), **constraints**, espacios mixtos (`int/float/categorical`) y **resume** vía RDB. Es el estándar para `n_trials` 200–2000 con backtests de segundos-minutos.

### 1.2 Diseño del objetivo (una sola función, métricas netas)

```python
# optimization/objective.py
def objective(trial, combo, master) -> float:
    params = sample_search_space(trial, master)          # respeta type/step/choices
    pnl, trades = run_pyeventbt_once(params, combo, costs=master.costs, window=master.windows.IS)
    s = summary(pnl, trades)                            # analytics/stats.py: sharpe_net, pf, maxdd, n_trades...
    trial.set_user_attr("pf", s.pf); trial.set_user_attr("n_trades", s.n_trades)  # todo queda guardado
    # Constraints duras: violaciones → trial fallido podado (no puntúa)
    if s.n_trades < master.optimization.objectives.constraints.min_trades: raise TrialPruned()
    if s.maxdd_pct > max_maxdd_pct: raise TrialPruned()
    # Early gate barato: a mitad de IS, si n_trades_fast < umbral → prune (ahorra ~50% cómputo)
    return s.sharpe_net                                  # primario SIEMPRE neto y con costes completos
```

Detalles obligatorios:

1. **Primario `sharpe_net`** (no profit, no PF solo, no win-rate). Sharpe anualizado sobre retornos diarios netos, `Rf=0`. Alternativa si `max_sets` multiobjetivo: `NSGAII [sharpe_net, -maxdd]`.
2. **Backtest idéntico para todos los trials**: mismo `cash`, `costs`, `IS`, `delay_bars=1`, `seed` derivada. `SharedData()` aislado por trial (DMRI: `filter_strategy_params` + coerción; aquí `params_class` pydantic + `filter_params` estricto que falla ante claves desconocidas).
3. **`run_pyeventbt_once`**: escribe OHLC a temp dir en formato MT5-M1, instancia `Strategy()` + `signal_fn(params)`, `RiskPct` o `Fixed` según spec, `backtest()` → devuelve `(pnl_df, trades_df)`. Sin tocar la cola de eventos directamente.
4. **Paralelo** `n_jobs=-1` a nivel trial (joblib/RDB). Prohibido paralelizar dentro del backtest y fuera a la vez (oversubscription).

### 1.3 Presupuesto, sampler, pruner (config del maestro)

- `n_trials: 400`, `n_startup_random: 40` (10%). `sampler: TPE` (o `NSGAII` si `multiobjective.enabled`). `pruner: median` (+ `Hyperband` si backtests >30s).
- **Resume**: `storage: sqlite:///runs/optuna.db`; re-lanzar con el mismo master continúa donde quedó (DMRI: skip por `is_done`; aquí por trial hash).
- **No tocar FORWARD.** El objetivo solo ve IS. Mirar FORWARD durante la optimización invalida todo el pipeline (se detecta por `ProcessLog`: acceso a lockbox antes de Etapa 4.6 = run contaminado).

### 1.4 Conservar valores: filtrado duro + Pareto + top-k por elite (qué guarda DMRI en `optimization/*.csv`)

DMRI guarda todos los sets del genético y filtra después. Aquí igual, pero con criterios explícitos en el maestro:

1. Guardar **todos** los trials en `optimization/<EST>_<PAIR>_<TF>.csv` + `trials.parquet` (params + `sharpe_net, pf, maxdd, n_trades, exposure, r2` + `trial_number, seed, duration`).
2. Filtro duro (`objectives.constraints`): `n_trades>=100`, `maxdd<=25%`, `exposure<=80%`, `pf>=1.0` (laxo; el 1.3 viene después).
3. Sobre los supervivientes: **Pareto** (si multiobjetivo) o **top-k por `sharpe_net`** con `k = 3× max_clusters_kept` (ej. 750 → cluster recorta a 250). Guardar `candidatos.csv` ordenado + `pareto.png`.
4. Registrar `S = nº de trials evaluados` y `K = nº de supervivientes`: el haircut de la Etapa 3 los necesita. Tirar trials malos del disco es destruir evidencia contra el overfitting.

### 1.5 Anti-overfit específico de esta etapa (SOTA que DMRI no tiene)

1. **Trial budget fijo** (§0.3): más trials = más mining. Si se amplía `n_trials` a posteriori, recalcular haircut/DSR con el `S` real, no con el inicial.
2. **Pruning honesto**: el pruner solo usa información intra-IS (mitad de la ventana), nunca FORWARD.
3. **Coerción estricta de tipos** (pandas sube a `float64`, `ta`/indicadores rechazan floats): `filter_params` como DMRI `strategy_params.py` pero contra `params_class`, con error ante `NaN` o clave desconocida.
4. **Determinismo**: mismo `master + seed` → mismo `trials.parquet` (verificado por test con `n_trials=20` en CI).

**Salida Etapa 1:** `optimization/{<EST>_<PAIR>_<TF>.csv, trials.parquet, candidatos.csv, pareto.png, optuna.db}` + `state: optimize DONE(total_trials=S, survivors=K)`.

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
- Opción `params_plus_behavior`: concatenar params normalizados + vector de comportamiento (`retornos mensuales` o `equity mensual` normalizada, peso 0.3). Dos sets con params distintos pero misma forma de P&L acaban juntos (lo que la Etapa 4 quiere evitar). Coste: un backtest por candidato (ya hecho en Etapa 1, reutilizar `pnl` cacheada).

### 2.5 Ensemble por cluster (antídoto SOTA al "punto ganador")

En vez de validar un punto por cluster en la Etapa 3, validar `ensemble3` (medoide + 2 mejores del cluster) y exigir que **≥2 de 3 pasen** el sintético. Un cluster cuyo mejor punto pasa pero sus vecinos fallan es un pico, no una meseta. Coste ×3 en sintético solo para clusters finalistas (los demás, 1 punto en screening).

### 2.6 Diagnósticos obligatorios

- `cluster/<EST>_<PAIR>_<TF>.csv` (1 fila = 1 representante + `cluster_id, cluster_size, medoid_dist_p50`).
- `cluster/diagnostics/{kdistance.png, sizes_hist.png, pca2d.png, members/<id>.csv}` (PCA/t-SNE 2D con clusters coloreados + outliers).
- `state: cluster DONE(total_clusters=C)`.

**Salida Etapa 2:** ≤250 representantes auditables con su tamaño de cluster y su posición en el mapa. Nada de "top 250 por Sharpe" sin desduplicar (eso es lo que el clustering evita).

---

## Etapa 3 — Validación en datos sintéticos GJR-GARCH

**Objetivo:** para cada representante, responder "¿mi Sharpe/PF/n_trades real es una muestra plausible del ensemble de mercados con la misma volatilidad, o un extremo overfiteado?" Se conserva el test percentile-rank de DMRI (`tools/metrics.py`) y se le añaden los vetos SOTA que le faltan (DSR, FDR, haircut, estrés de costes).

### 3.1 Generador (port DMRI + 3 endurecimientos)

1. `SyntheticMarketGenerator` (DMRI `tools/synthetic_data.py` casi verbatim): `arch GARCH(p=1,q=1,o=1,dist=t)` sobre log-retornos ×100, `std_resid`, `fast_garch_simulation @njit` con término asimétrico `gamma*res²*(res<0)`, `generate_universe(method, block_size)` con corrección de deriva a `log(Close[-1]/Open[0])`, `build_ohlc_fast` (closes por cumsum, opens retardados, mechas por ratios reales `_u/_d_ratios`).
2. **Cache por `(símbolo,TF)`** ajustado en IS (DMRI ya lo hace). `to_csv()` en M1 MT5 para que `run_pyeventbt_once` no distinga real de sintético.
3. Endurecimientos vs DMRI:
   - **Validación del generador**: antes de validar sets, comprobar que el sintético replica `volatilidad, autocorr(|r|), cola (kurtosis), rango medio diario` dentro de ±20% del real. Generador que no replica es máquina de rechazos falsos.
   - **Stationary bootstrap** opcional (`p=0.1`, longitudes geométricas) además de `block` fijo: menos artefactos de borde que `block_size=10` puro.
   - **Mechas y spread sintéticos honestos**: mechas por ratios reales (ya en DMRI) + columna `spread` = `spread_medio_real × (1 + estrés)` en el re-scoring §3.6, no spread plano 1.

### 3.2 Protocolo de scoring (DMRI + vetos)

1. Por representante: 1 backtest real (IS, `costs` maestro) + N sintéticos (`N=20` screening, `N=50` finalistas) con **idénticos params/cash/costs**.
2. `score_percentile_rank(real, synth_dist, [low=10, high=90])` por métrica (`Sharpe, PF, #Trades`): `rank=(n_below+0.5·n_equal)/n·100`, PASS si dentro de banda. Casos borde DMRI conservados (`NaN/inf→None`, varianza-cero: igual→50 si no 0/100).
3. **Quorum** `min_metrics_pass: 0.5` (fracción) como DMRI. Sin quorum laxo no hay pipeline viable; sin quorum no hay control.
4. **Vetos SOTA que DMRI no tiene (todos en el maestro):**
   - **DSR veto** (`dsr.veto_below: 0.20`): Deflated Sharpe sobre `trials.parquet` (usa `S` real de la Etapa 1). Aunque los ranks pasen, `DSR<0.20` = FAIL (el Sharpe se explica por haber probado S variantes).
   - **FDR** (`fdr.q: 0.10` Benjamini-Hochberg sobre p-valores bilaterales derivados de ranks): con C clusters testeados, esperar ~C·q falsos positivos; ajustar la lista de `passed` por FDR antes de la Etapa 4.
   - **Haircut por nº de tests**: reportar `Sharpe_haircut = f(S)` junto al Sharpe nominal (Bonferroni/Harvey-Liu); no veta solo, pero ordena la cola de finalistas.
   - **PBO-lite**: si `representative: ensemble3`, exigir ≥2/3 verdes (§2.5). Es el Combinatorial-Symmetric-CV reducido a nivel cluster.

### 3.3 `n_sims` adaptativo (coste bajo control)

- Screening: `n_sim_screening=20` para los C representantes → ~C·20 backtests (paralelo `n_jobs=-1`, `joblib`, conteo por worker como DMRI `_run_sims_parallel`).
- Finalistas (top por `sharpe_net` + outliers singletons): `n_sim_final=50`. Promediar coste: `C·20 + F·30` en vez de `C·50`.
- Regla: si con 20 sims el rank ya está fuera de `[5,95]` (más estricto que la banda), no gastar 50 (rechazo temprano).

### 3.4 Paralelo, seeds y determinismo

- `counts = reparte(n_sims, n_workers)` como DMRI; seed por `(master.seed, cluster_id, sim_idx)` → reproducible.
- `__getstate__/__setstate__` del generador (DMRI ya lo tiene para `joblib`) conservado: `model_res=None` en pickle.

### 3.5 Plots y CSVs (paridad DMRI)

- Por set: `real_vs_synth.png` (real, mediana, banda IQR, rank anotado). Por combo: `deviation_distribution.png`.
- `synthetic_validation/<EST>_<PAIR>_<TF>.csv`: filas que pasan + `ranks{sharpe,pf,trades}, diagnostics{median,iqr,n_valid}, dsr, fdr_adj_p, haircut`.
- `state: synthetic DONE(total=C, passed=P)`.

### 3.6 Re-scoring estresado (novedad: costes futuros, no pasados)

- Re-evaluar los `passed` con `synth_stress: {spread_mult: 2.0, slippage_ticks_add: 1}` aplicado al ledger sintético (sin re-simular): exigir que el quorum siga verde. Un set que solo pasa con spread 2021 no tiene edge en 2026.
- Guardar `stress_pass` por set. `stress FAIL + base PASS` = ámbar (pasa a Etapa 4 con la mitad de prioridad, no con veto total).

**Salida Etapa 3:** `synthetic_validation/*.csv` con P supervivientes + evidencia (ranks, DSR, FDR, estrés) por set. Lo que no pasa aquí no existe en la Etapa 4.

---

## Etapa 4 — Decorrelación y selección final de parámetros

**Objetivo:** de P supervivientes sintéticos, quedarse con `k<=max_sets` (ej. 5) genuinamente distintos y con mejor neto ajustado a turnover, verificados una sola vez en FORWARD. DMRI `decorrelate.py` hace esto por `(pair,TF)` con `greedy rho<=0.7, max 10, score PF+R2`; aquí se conserva el algoritmo y se mejora el score, el cap y el chequeo final.

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

### 4.5 Estabilidad local del representante (novedad: el "jitter test")

- Por cada uno de los k: `jitter ±10%` en cada parámetro numérico (redondeo a step), re-backtest rápido en IS. Exigir `min_green: 0.8` (≥80% vecinos con `sharpe_net>0` y `pf>=1.0`). El que falla es pico: se sustituye por el siguiente del ranking greedy.
- Barato (k·2·d backtests) y mata los "ganadores de lotería" que sobrevivieron al sintético por azar.

### 4.6 Chequeo FORWARD lockbox (una vez, al final, sin reintentos)

- Solo los k finalistas tocan `FORWARD`: un backtest por set, `forward_min_pf: 1.1` + `sharpe_net>0`. **Sin re-optimizar, sin re-elegir, sin segundo intento.** Fallar aquí = set descartado (no se vuelve a la Etapa 1 con el FORWARD ya visto; se registra y el run termina con k'<k).
- `ProcessLog` marca acceso a lockbox; cualquier acceso anterior invalida el run (test de integridad en CI).
- Guardar `forward/{...}.csv + forward_equity.png` (IS sombreado vs FORWARD) por set.

### 4.7 De k finalistas a `promoted.csv` (puertas + sizing, herencia `demo.py`)

1. Puertas por set (de `PLAYBOOK` Fase 9, aplicadas aquí al finalista, no antes): `PF>=1.3` (o 1.1–1.3 + retest 3 salidas MA/BB/RSI con `>=1.3`), `R2>=0.5`, `DD<365d`, `≤10/12` meses rojos.
2. `Lots = min(Lots_dd, Lots_max_los, Lots_mean_loss)` (`max_dd_allow/max_trade_lost_allow/mean_trade_lost_allow` del maestro; `0→inf`).
3. `SL/TP` por `percentil_MFE_MAE (p10/p90)` o `ATR_mult` — **nunca `min/max` extremos** (crítica `sets_to_demo_sl_tp.md` DMRI).
4. `promoted.csv`: `params + gates{...} + lots + SL/TP + cost_cfg + manifest_hashes + master_hash + commit + magic secuencial desde magic_start`. `selected_equity.png` normalizada de los k.

**Salida Etapa 4:** `decorrelation/<EST>_<PAIR>_<TF>.csv` (k filas) + `diagnostics/{candidates, correlation, selected_equity}` + `forward/` + `promoted.csv`. Fin del pipeline. Lo siguiente (paper/paridad/producción) es `PLAYBOOK` Fases 11–13, no este documento.

---

## Orquestación: stages, estado reanudable, artefactos, dashboard

Réplica del patrón DMRI `stages/base.py + tools/state.py + dashboard/runner.py`, adaptado a Optuna:

```text
bots_folder/{estrategia}/           # o runs/{estrategia}/ (recomendado nuevo)
  process_logs.yaml                  # por (símbolo,TF,etapa): pending|running|done|error + counts
  optuna.db                          # storage resume
  master_used.yaml + master_hash
  optimization/<EST>_<PAIR>_<TF>.csv + trials.parquet + candidatos.csv
  cluster/<EST>_<PAIR>_<TF>.csv + diagnostics/
  synthetic_validation/<EST>_<PAIR>_<TF>.csv + plots/
  decorrelation/<EST>_<PAIR>_<TF>.csv + diagnostics/ + forward/
  promoted.csv
```

1. `Stage` base (`name, discover_inputs, run(pair,timeframe), persist, _update_state`) + `StageContext{config, state, out_dir}`. Cinco stages: `OptimizeOptuna → ClusterParams → ValidateSynthetic → DecorrelateSelect → PromoteGates`.
2. `ProcessLog` atómico (`tmp+os.replace`), `init_stage/mark_done/mark_error` con contadores (`total_trials/survivors/clusters/passed/selected`). Re-lanzar salta `done` cuyo `config_hash` coincide; si el maestro cambió, solo re-ejecuta lo afectado (trials conservados si `search_space` no cambió).
3. Bucle `por (símbolo,TF): por stage` (DMRI `runner.py`) + `ProgressDashboard` Rich (`strategy/asset/step/progress` + logs rodantes, sin `print` en pipeline).
4. Regla de aborto: `passed==0` tras sintético → run termina en ámbar con diagnóstico (no se fuerza decorrelación sobre vacío; `sets_to_demo` DMRI ya hace fallback correcto: sin input no hay output).

---

## Tabla DMRI → PyEventBT: qué se reutiliza y qué se mejora (SOTA)

| Paso DMRI (archivo) | Reutilizar en PyEventBT | Mejora SOTA obligatoria |
|---|---|---|
| `stages/optimize.py` + `mt4_tools.py` (genético MT4, `metrics 0..8`, `nbars`) | Bucle por combo, skip `is_done`, `.set`→`params`, harvest CSV | **Optuna TPE/NSGA-II**, `n_trials` presupuestado, pruner, constraints, embargo, `trials.parquet` completo, RDB resume |
| `sets_selection.generate_cluster_sets` (DBSCAN `eps=30`, `mode`) | DBSCAN + `columns_to_drop` + cap 250 | RobustScaler, `eps` auto k-distance, **HDBSCAN** opcional, **medoide** (no moda), singletons buenos, `params_plus_behavior`, ensemble3 |
| `synthetic_data.SyntheticMarketGenerator` + `fast_garch_simulation` | Generador + block-bootstrap + deriva + mechas reales + cache/gb`__getstate__` | Validador del generador, stationary bootstrap, spread/slippage sintético, `n_sims` adaptativo 20→50, seeds por sim |
| `metrics.score_percentile_rank` + `plotting` | Rank math + edge cases + plots por set/combo | Métricas netas, **DSR veto 0.20**, **FDR q=0.10**, haircut por S, PBO-lite ensemble, re-scoring estresado |
| `decorrelation` + `stages/decorrelate` (retornos D, greedy `0.7`, `max 10`, `PF+R2`) | Retornos (no niveles), unilateralidad, greedy/cluster, winsor `p95`, diagnósticos | Score `sharpe_net+pf+r2−turnover`, `max_sets=5`, apuestas efectivas, **jitter ±10%**, **FORWARD lockbox una vez** |
| `demo` puertas + `Lots=min` + `SL/TP=min/max` + magics + `1_for_demo` | Puertas `PF/R2/DD/meses`, fórmula lots, magics, `promoted.csv` | `SL/TP` percentil/ATR, lots con p95 MC, hash maestro+manifests en cada fila |
| `state.py + config.py + filename.py + dashboard` | `ProcessLog`, `BotConfig+ConfigError`, `TIMEFRAME_DICT`, Rich runner | Maestro único versionado, hash-check anti-cambio, test de integridad lockbox |

---

## Implementación en PyEventBT: módulos y CLI

Nuevos módulos (nada rompe lo existente):

```text
pyeventbt/
  optimization/{runner_optuna.py, params.py, cluster.py, objective.py}
  robustness/{synthetic.py, scoring.py, validate_synthetic.py, decorrelation.py,
              walkforward.py(defer), monte_carlo.py(defer), report.py}
  analytics/{stats.py, tearsheet.py}          # summary() con sharpe_net/pf/r2/n_trades/turnover
  selection/{gates.py, lots.py, sl_tp.py, promote.py}
  runner/{stage.py, process_log.py, master_config.py, dashboard/}
pipeline_master.yaml
bots_dispatcher_optuna.py                     # entry: python bots_dispatcher_optuna.py pipeline_master.yaml
```

CLI:

```bash
pip install optuna arch joblib scikit-learn scipy   # + pyeventbt[opt]
python bots_dispatcher_optuna.py pipeline_master.yaml            # todo el pipeline
python bots_dispatcher_optuna.py pipeline_master.yaml --only optimize,cluster --pair EURUSD --timeframe H1
python -m robustness.report runs/<est>/<ts>_<hash> --full        # gate_report.json agregado
```

Adaptador único (frontera DMRI↔PyEventBT, §4 DMRI_TRANSFER):

```python
def run_pyeventbt_once(params, combo, *, costs, window, cash, seed) -> (pnl_df, trades_df):
    # 1. valida params contra params_class (falla ante clave desconocida/NaN)
    # 2. escribe OHLC (real o sintético) a temp dir formato MT5-M1
    # 3. Strategy() + signal_fn(params) + sizing/riesgo de spec + backtest()
    # 4. devuelve frames normalizados + summary(); registra manifest_hash + cost_cfg
```

Tests que bloquean merge (patrón `tests/` DMRI): `test_scoring` (ranks incl. bordes), `test_cluster` (medoide vs moda, eps auto), `test_params_filter` (coerción/rechazo), `test_synthetic_determinism` (misma seed → mismo universo), `test_decorr_unilateral` (negativos se quedan), `test_lockbox_integrity` (acceso temprano = fail), `test_optuna_resume` (20+20 == 40).

---

## Gates numéricos y criterios de aceptación

Defaults (maestro los fija; endurecer por TF/símbolo, nunca ablandar sin registro):

| Gate | Umbral | Fuente |
|---|---|---|
| `min_trades_IS` por trial | ≥100 | DMRI `find_valid_rules` + playbook |
| Constraints optuna | `maxdd≤25%`, `exposure≤80%`, `pf≥1.0` | Nuevo (laxo; lo duro va después) |
| Cluster | ≤250 representantes, medoide, singletons top-decila | DMRI `cluster_sets: 250` |
| Sintético ranks | dentro de [10,90], quorum 0.5 sobre {Sharpe, PF, #Trades} | DMRI `validation` verbatim |
| DSR veto | <0.20 → FAIL | Nuevo SOTA (usa S real) |
| FDR | q=0.10 BH sobre shortlist | Nuevo SOTA |
| Estrés costes | quorum verde con `spread×2 +1 tick` o ámbar con mitad de prioridad | Nuevo |
| Decorrelación | `rho≤0.7` retornos D, `max_sets=5`, `min_trades=30` | DMRI (`max 10→5` elite) |
| Jitter | ≥80% vecinos ±10% con `sharpe>0, pf≥1.0` | Nuevo |
| FORWARD | `pf≥1.1`, `sharpe_net>0`, un solo intento | DMRI `end_forward_date` + lockbox estricto |
| Promoción | `PF≥1.3` (o 1.1–1.3 + retest salidas ≥1.3), `R2≥0.5`, `DD<365d`, `≤10/12` rojos | DMRI `filters` verbatim |
| Sizing/SLTP | `Lots=min(...)`, SL/TP percentil/ATR | DMRI fórmula + crítica extremos |

**Definición de DONE del pipeline:** `promoted.csv` con k filas, cada una con `ranks + dsr + fdr + jitter + forward` trazables + `master_hash + manifest_hashes + commit`. Menos que eso es un informe, no una selección.


---

# ARQUITECTURA v2 — De Pipeline de Research a Plataforma Quant End-to-End

> **Extensión del documento original.** Se conserva el pipeline DMRI → Optuna → Clustering → Synthetic → Decorrelation → Forward, pero se añaden las capas necesarias para que el mismo código pueda evolucionar desde investigación reproducible hasta portfolio, ejecución, paper trading y producción live.

## 12. Objetivo de arquitectura: no solo optimizar estrategias

El sistema deja de considerarse únicamente un *backtesting/validation pipeline*. El objetivo de v2 es construir una **Quant Research & Trading Platform** donde una estrategia pueda recorrer un ciclo de vida controlado:

```text
RESEARCH / DISCOVERY
        │
        ▼
STRATEGY SPEC + VERSION FREEZE
        │
        ▼
VALIDATION
  Purged CV / CPCV / PBO / DSR
  Monte Carlo / Synthetic / Stress
        │
        ▼
WALK-FORWARD
        │
        ▼
FINAL HOLDOUT / LOCKBOX
        │
        ▼
PORTFOLIO CONSTRUCTION
  Correlation / Factors / CVaR / Capacity
        │
        ▼
RISK APPROVAL
        │
        ▼
PAPER → SHADOW → MICRO-LIVE
        │
        ▼
LIVE
        │
        ├── Monitoring / Drift / Reconciliation
        ├── Risk limits / Kill switches
        └── Degradation → Quarantine → Retirement
```

Principio central:

> **El sistema debe hacer difícil engañarse a uno mismo.** Un backtest espectacular no es una estrategia aprobada; es solo una hipótesis que debe sobrevivir a múltiples controles independientes.

---

## 13. Separación estricta: Discovery vs Validation

El documento original está diseñado principalmente para una estrategia cuya lógica ya está definida. En v2 se formaliza una frontera adicional: **Research/Discovery puede explorar hipótesis; Validation solo puede validar una hipótesis congelada.**

### 13.1 Research / Discovery

Puede incluir:

- nuevas reglas de entrada/salida;
- nuevos indicadores/features;
- nuevos universos;
- nuevos timeframes;
- nuevas transformaciones;
- feature engineering;
- tests exploratorios;
- data mining explícito.

Cada experimento debe registrar:

```yaml
experiment_id:
parent_experiment_id:
hypothesis:
researcher:
created_at:
code_commit:
data_manifest_hash:
features_version:
search_space_hash:
trial_budget:
notes:
result:
```

### 13.2 Strategy Spec Freeze

Cuando una hipótesis pasa a validación se crea una versión inmutable:

```text
strategy_id = EURUSD_15M_XYZ
strategy_version = 003
spec_hash = ...
code_commit = ...
data_manifest_hash = ...
search_space_hash = ...
```

Desde ese momento no se puede modificar silenciosamente:

- lógica;
- features;
- universo;
- costes;
- ventanas;
- espacio de parámetros;
- definición de labels;
- reglas de ejecución.

Un cambio crea una nueva versión y una nueva genealogía experimental.

---

## 14. Data Layer institucional y Data Leakage Firewall

La calidad de los datos debe ser un componente independiente del backtester.

### 14.1 Capas de datos

```text
RAW
 │
 ▼
NORMALIZED
 │
 ▼
CLEAN / VALIDATED
 │
 ▼
FEATURE STORE
 │
 ▼
BACKTEST DATASET
 │
 ▼
LIVE DATA
```

Cada dataset debe conservar:

- proveedor;
- instrumento/símbolo;
- timezone;
- calendario;
- timestamp precision;
- corporate actions cuando aplique;
- ajustes;
- gaps;
- duplicados;
- revisiones de datos;
- filas;
- rango temporal;
- hash;
- versión del pipeline de transformación.

### 14.2 Data Manifest ampliado

```yaml
dataset_id:
source:
version:
asset:
timeframe:
timezone:
start:
end:
rows:
gaps:
duplicates:
adjustments:
timestamp_policy:
calendar:
feature_version:
content_hash:
created_at:
```

### 14.3 Data Leakage Firewall

Antes de cualquier backtest se deben bloquear automáticamente:

- look-ahead bias;
- features calculadas usando datos futuros;
- joins temporales incorrectos;
- timestamps fuera de orden;
- revisiones futuras de macrodatos;
- survivorship bias;
- cambios de símbolo no tratados;
- duplicados;
- barras incompletas;
- uso accidental del futuro/lockbox;
- señales calculadas con barra aún no cerrada.

Un fallo de integridad de datos debe producir **FAIL**, no warning.

---

## 15. Validación estadística v2

El pipeline actual incorpora DSR, FDR, embargo y un PBO-lite. Para una validación institucional se añade una batería estadística completa.

### 15.1 Walk-Forward obligatorio

`walkforward.py` deja de ser `defer`. Debe ser P0/P1 porque mide si la estrategia mantiene capacidad predictiva cuando los parámetros se entrenan únicamente sobre pasado.

Ejemplo:

```text
Train 2015-2018 → Test 2019
Train 2015-2019 → Test 2020
Train 2015-2020 → Test 2021
Train 2015-2021 → Test 2022
...
```

Registrar por ventana:

- parámetros elegidos;
- Sharpe;
- PF;
- DD;
- trades;
- turnover;
- costes;
- régimen;
- estabilidad paramétrica.

El resultado final debe incluir **consistencia entre ventanas**, no solo el promedio.

### 15.2 Purged K-Fold + embargo

Cuando exista solapamiento entre observaciones, las muestras de entrenamiento que puedan contaminar el test deben purgarse. El embargo añade una separación temporal adicional.

La implementación debe aceptar:

```python
purged_kfold(
    timestamps,
    n_splits=5,
    embargo_pct=0.01,
    label_horizon=...
)
```

### 15.3 CPCV + PBO real

El `PBO-lite` del pipeline se mantiene como filtro barato, pero se añade **Combinatorial Purged Cross-Validation** para estimar de forma más rigurosa la Probability of Backtest Overfitting.

Artefactos mínimos:

```text
validation/cpcv/
  partitions.parquet
  performance_matrix.parquet
  logit_matrix.parquet
  pbo.json
```

### 15.4 DSR y PSR

DSR no debe ser un número aislado. El sistema debe conservar:

- Sharpe observado;
- Sharpe esperado bajo selección múltiple;
- número efectivo de trials;
- skewness;
- kurtosis;
- PSR;
- DSR.

### 15.5 Minimum Backtest Length / Track Record

Una estrategia no puede promocionarse simplemente porque supera los gates con pocos datos. El registry debe imponer mínimos de:

- longitud temporal;
- número de trades;
- diferentes regímenes;
- años/calendarios;
- episodios de estrés.

---

## 16. Monte Carlo y Robustness Lab

La validación sintética GJR-GARCH se conserva, pero se convierte en un **Robustness Lab** con varios generadores independientes.

### 16.1 Familias de perturbación

```text
Historical Bootstrap
       ├── iid trade reshuffle
       ├── block bootstrap
       └── stationary bootstrap

Model-based
       ├── GARCH / GJR-GARCH
       ├── stochastic volatility
       └── regime-conditioned generators

Trade-level
       ├── reshuffle
       ├── parameter perturbation
       └── execution perturbation

Stress
       ├── spread shock
       ├── slippage shock
       ├── latency shock
       ├── volatility shock
       └── gap/shock scenarios
```

### 16.2 Resultados que deben producirse

No basta con un percentile de Sharpe. Estimar:

- P(DD > X);
- P(pérdida anual > X);
- P(ruina);
- expected shortfall;
- máximo periodo perdedor;
- recovery time;
- distribución de CAGR;
- distribución de Sharpe;
- sensibilidad a costes;
- sensibilidad a parámetros.

---

## 17. Execution Simulator y Capacity Model

El backtest actual debe evolucionar de un modelo de costes fijo a un simulador de ejecución.

### 17.1 Variables de ejecución

```yaml
execution:
  spread_model: empirical
  slippage_model: empirical
  latency_model: empirical
  partial_fills: true
  rejection_probability: true
  market_impact: true
  queue_model: optional
  gap_model: true
  session_liquidity: true
```

Registrar por trade:

- signal timestamp;
- order timestamp;
- broker acknowledgement;
- fill timestamp;
- requested price;
- executed price;
- spread;
- slippage;
- latency;
- fill ratio;
- fees;
- rejection.

### 17.2 Capacity

Toda estrategia promovida debe tener una curva:

```text
Capital → Expected Return / Sharpe / DD / Slippage / Market Impact
```

El sistema debe estimar:

- capital máximo antes de degradación;
- participación relativa en el mercado;
- sensibilidad a slippage;
- capital óptimo;
- capacidad por instrumento y sesión.

Una estrategia con Sharpe alto pero capacidad de €20k no debe presentarse como una estrategia de €1M.

---

## 18. Portfolio Construction y Risk Engine

La selección actual de parámetros por decorrelación pasa a ser solo una parte de la construcción de portfolio.

### 18.1 Risk dimensions

Medir correlación en:

- retornos diarios;
- retornos semanales;
- drawdowns;
- colas;
- diferentes regímenes;
- factores comunes.

### 18.2 Métodos de asignación

El framework debe soportar, como mínimo:

- equal weight;
- volatility scaling;
- risk parity / Equal Risk Contribution;
- minimum variance;
- maximum diversification;
- CVaR/Expected Shortfall constrained;
- volatility target.

La estrategia nunca debe decidir por sí sola el riesgo total del portfolio.

### 18.3 Risk Budget

```yaml
risk_budget:
  max_portfolio_dd_pct: 15
  max_strategy_dd_pct: 10
  max_daily_loss_pct: 2
  max_weekly_loss_pct: 4
  max_asset_exposure_pct: 25
  max_correlated_exposure_pct: 40
  max_currency_exposure_pct: 30
```

Los límites de riesgo son **gates de seguridad**, no parámetros que Optuna pueda optimizar para mejorar el backtest.

---

## 19. Strategy Registry y Lifecycle State Machine

Cada estrategia debe existir como una entidad versionada y trazable.

```text
RESEARCH
  ↓
CANDIDATE
  ↓
VALIDATED
  ↓
APPROVED
  ↓
PAPER
  ↓
SHADOW
  ↓
LIVE_SMALL
  ↓
LIVE
  ├── DEGRADED
  ├── QUARANTINED
  └── RETIRED
```

### 19.1 Registry record

```yaml
strategy_id:
version:
status:
spec_hash:
code_commit:
data_manifest_hash:
research_lineage:
validation_report:
portfolio_assignment:
risk_budget:
execution_profile:
approved_by:
approved_at:
last_health_check:
```

Ninguna estrategia puede pasar de estado sin cumplir el gate correspondiente.

---

## 20. Paper → Shadow → Micro-live → Live

El salto directo de backtest a producción queda prohibido.

### PAPER

Simulación con datos live y ejecución simulada. Se compara continuamente contra el backtest.

### SHADOW

El sistema genera señales y órdenes reales pero no ejecuta. Permite comparar:

```text
expected signal
expected order
expected fill
vs
market actual
```

### MICRO-LIVE

Capital mínimo para validar:

- conectividad;
- fills;
- costes;
- latencia;
- reconciliación;
- P&L real.

### LIVE

Solo tras superar los gates de operación y riesgo. El capital aumenta progresivamente.

---

## 21. Live Monitoring y Strategy Health Score

El sistema debe detectar que una estrategia puede estar deteriorándose antes de que llegue a su máximo drawdown histórico.

### 21.1 Variables monitorizadas

```text
Performance
 ├── PnL
 ├── Sharpe
 ├── PF
 └── Drawdown

Execution
 ├── Spread
 ├── Slippage
 ├── Fill rate
 └── Latency

Behaviour
 ├── Trade frequency
 ├── Holding time
 ├── Exposure
 └── Regime distribution

Model/Data
 ├── Feature drift
 ├── Distribution drift
 ├── Missing data
 └── Signal drift
```

### 21.2 Health Score

Crear un score independiente del P&L:

```text
Health = Performance + Execution + Statistical + Regime + Liquidity + Risk
```

Ejemplo de estados:

```text
GREEN   → normal
YELLOW  → vigilancia
ORANGE  → reducción de riesgo
RED     → quarantine / kill switch
```

---

## 22. Reconciliation y Event Sourcing

Una plataforma live debe poder reconstruir exactamente qué ocurrió.

### 22.1 Reconciliation

Comparar periódicamente:

```text
Internal state
      vs
Broker state
```

Para:

- cash;
- posiciones;
- órdenes abiertas;
- fills;
- fees;
- realized P&L;
- unrealized P&L.

Cualquier diferencia material produce alerta y, si supera un umbral, bloqueo de nuevas órdenes.

### 22.2 Event sourcing

Guardar cada evento:

```text
MarketEvent
SignalEvent
RiskDecision
OrderCreated
OrderSubmitted
OrderAccepted
OrderRejected
OrderPartiallyFilled
OrderFilled
PositionUpdated
RiskLimitBreached
KillSwitchTriggered
```

Con esto una operación completa debe poder reconstruirse sin depender de logs humanos.

---

## 23. Kill Switches, seguridad y resiliencia

### 23.1 Kill switches

Obligatorios:

- pérdida diaria máxima;
- drawdown máximo;
- slippage anormal;
- spread anormal;
- broker desconectado;
- datos stale;
- heartbeat perdido;
- posición inesperada;
- órdenes duplicadas;
- breach de riesgo;
- divergencia broker/internal state.

### 23.2 Seguridad

- secretos fuera del repositorio;
- API keys con mínimo privilegio;
- credenciales research y live separadas;
- rotación de claves;
- MFA donde sea posible;
- backups;
- logs inmutables;
- auditoría;
- rollback;
- disaster recovery;
- incident response.

### 23.3 Recovery

El sistema debe soportar reinicio sin duplicar órdenes. Para ello cada orden debe tener un `client_order_id` idempotente y el estado debe poder reconstruirse desde eventos + broker reconciliation.

---

## 24. CI/CD y Strategy Regression Testing

Cada cambio de código debe pasar tests antes de poder promocionarse.

### 24.1 Test layers

```text
Unit tests
   ↓
Data integrity tests
   ↓
Leakage tests
   ↓
Backtest determinism
   ↓
Backtest regression
   ↓
Risk tests
   ↓
Execution simulator tests
   ↓
Integration tests
   ↓
Paper / Shadow
```

### 24.2 Regression test de estrategia

Un cambio que produzca, por ejemplo:

```text
Sharpe 1.83 → 2.41
MaxDD 18%  → 11%
```

no debe celebrarse automáticamente. Debe levantar una alerta de **behavioural regression** y exigir explicación del cambio. Mejoras demasiado grandes son potencialmente evidencia de un bug o leakage.

### 24.3 Determinismo

El mismo:

```text
code_commit + data_hash + config_hash + seed
```

debe producir el mismo resultado dentro de una tolerancia definida.

---

## 25. Experiment Lineage y Research Budget

El número de Optuna trials no es el único número que importa. También cuentan:

- número de estrategias probadas;
- número de features;
- número de universos;
- número de timeframes;
- número de espacios de parámetros;
- número de ventanas;
- decisiones manuales;
- número de variantes descartadas;
- número de datasets inspeccionados.

El sistema debe registrar un **Research Ledger**:

```text
Research budget
 ├── hypotheses tested
 ├── datasets tested
 ├── feature families
 ├── parameter trials
 ├── model variants
 ├── manual selections
 └── final selected hypothesis
```

Esto alimenta DSR/PBO y permite saber cuánta selección ha sufrido una estrategia antes de declarar que existe edge.

---

## 26. Estructura de repositorio v2

La estructura recomendada pasa a ser:

```text
pyeventbt/
├── core/
│   ├── events/
│   ├── strategy/
│   ├── portfolio/
│   └── execution/
├── data/
│   ├── ingestion/
│   ├── normalization/
│   ├── validation/
│   ├── lineage/
│   └── leakage_firewall/
├── research/
│   ├── experiments/
│   ├── discovery/
│   ├── features/
│   ├── optimization/
│   └── lineage/
├── validation/
│   ├── purged_cv/
│   ├── cpcv/
│   ├── pbo/
│   ├── dsr/
│   ├── psr/
│   ├── walk_forward/
│   ├── monte_carlo/
│   ├── synthetic/
│   ├── stress/
│   └── robustness/
├── portfolio/
│   ├── construction/
│   ├── allocation/
│   ├── correlation/
│   ├── factors/
│   ├── risk/
│   └── capacity/
├── execution/
│   ├── simulator/
│   ├── mt5/
│   ├── brokers/
│   ├── reconciliation/
│   ├── routing/
│   └── slippage/
├── production/
│   ├── deployment/
│   ├── paper/
│   ├── shadow/
│   ├── live/
│   ├── monitoring/
│   ├── health/
│   ├── alerts/
│   ├── kill_switch/
│   └── recovery/
├── governance/
│   ├── registry/
│   ├── approvals/
│   ├── audit/
│   ├── lineage/
│   └── research_budget/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── leakage/
│   ├── regression/
│   ├── execution/
│   └── production/
└── configs/
    ├── pipeline_master.yaml
    ├── risk.yaml
    ├── execution.yaml
    └── environments/
        ├── research.yaml
        ├── paper.yaml
        └── live.yaml
```

---

## 27. Nuevos contratos/interfaces Python

La arquitectura debe depender de interfaces, no de implementaciones concretas.

```python
class ExecutionBroker(Protocol):
    def get_balance(self): ...
    def get_positions(self): ...
    def get_orders(self): ...
    def place_order(self, order): ...
    def cancel_order(self, order_id): ...


class RiskEngine(Protocol):
    def check_order(self, order, portfolio): ...
    def check_portfolio(self, portfolio): ...


class StrategyRegistry(Protocol):
    def register(self, strategy_version): ...
    def promote(self, strategy_id, target_state): ...
    def retire(self, strategy_id): ...


class DataProvider(Protocol):
    def load(self, dataset_id, window): ...
    def manifest(self, dataset_id): ...


class ExecutionSimulator(Protocol):
    def simulate(self, order, market_state): ...
```

Esto permite conectar posteriormente:

- PyEventBT / MT5;
- brokers tradicionales;
- crypto exchanges;
- prediction markets;
- otros adapters;

sin modificar strategy/risk/portfolio.

---

## 28. Roadmap definitivo P0 → P5

### P0 — Research Engine

**Objetivo:** hacer reproducible el pipeline actual.

- pipeline_master.yaml;
- Optuna;
- trials completos;
- clustering;
- synthetic;
- DSR/FDR;
- decorrelation;
- lockbox;
- tests básicos;
- hashes.

### P1 — Statistical Validation

**Objetivo:** saber si el edge sobrevive a validación estadística seria.

- Walk-forward;
- Purged K-Fold;
- CPCV;
- PBO;
- PSR/DSR;
- Monte Carlo;
- parameter stability;
- regime analysis;
- research ledger.

### P2 — Portfolio & Risk

**Objetivo:** pasar de estrategia individual a cartera.

- risk engine;
- risk budgets;
- CVaR;
- factor exposure;
- allocation;
- capacity;
- stress testing;
- portfolio-level Monte Carlo.

### P3 — Execution

**Objetivo:** que el backtest represente la realidad.

- execution simulator;
- empirical spread/slippage;
- latency;
- partial fills;
- rejections;
- market impact;
- MT5 adapter;
- broker abstraction;
- reconciliation.

### P4 — Production

**Objetivo:** operar sin depender de intervención manual constante.

- strategy registry;
- paper;
- shadow;
- micro-live;
- live;
- monitoring;
- health score;
- kill switches;
- alerts;
- audit trail;
- recovery.

### P5 — Institutionalization

**Objetivo:** que el sistema pueda gestionar una cartera significativa y múltiples estrategias.

- multi-strategy allocation;
- capital allocation engine;
- cross-asset risk;
- capacity optimization;
- distributed execution;
- stronger event store;
- immutable research artifacts;
- disaster recovery;
- full CI/CD;
- operational governance.

---

## 29. Definition of Done v2

Una estrategia **NO está lista para live** aunque tenga un `promoted.csv`. Debe existir evidencia completa en todas estas capas:

```text
[✓] Data integrity
[✓] Strategy spec frozen
[✓] Research lineage
[✓] Trial budget recorded
[✓] Optimization complete
[✓] Clustering / parameter stability
[✓] Synthetic robustness
[✓] DSR / PSR / multiple-testing controls
[✓] Purged CV / CPCV / PBO
[✓] Walk-forward
[✓] Monte Carlo / stress
[✓] Final lockbox
[✓] Portfolio correlation / factor / CVaR
[✓] Capacity analysis
[✓] Execution simulation
[✓] Paper trading
[✓] Shadow trading
[✓] Micro-live
[✓] Reconciliation
[✓] Risk limits
[✓] Kill switches
[✓] Monitoring / health score
[✓] Audit trail
[✓] Rollback / recovery
```

La promoción final debe producir un **Strategy Approval Package** que contenga, como mínimo:

```text
approval_package/
├── strategy_spec.yaml
├── master_used.yaml
├── data_manifests/
├── research_ledger.json
├── trials.parquet
├── clustering.parquet
├── synthetic_validation.parquet
├── cpcv_report.json
├── pbo_report.json
├── dsr_report.json
├── walkforward.parquet
├── monte_carlo.parquet
├── stress_report.json
├── portfolio_report.json
├── capacity_report.json
├── execution_report.json
├── forward_lockbox_report.json
├── promoted.csv
├── risk_limits.yaml
├── approval.json
└── hashes.json
```

> **Criterio final:** si una tercera persona no puede reconstruir por qué una estrategia fue seleccionada, con qué datos, qué alternativas se probaron, qué costes se asumieron, qué controles superó y qué versión exacta se desplegó, el sistema todavía no está al nivel objetivo.

---

## 30. Cambios respecto a v1 y prioridad de implementación

| Área | Estado original | v2 | Prioridad |
|---|---|---|---|
| Optuna / clustering / synthetic | Implementado en diseño | Mantener | P0 |
| Trial budget / hashes / lineage | Parcial | Formalizar | P0 |
| Walk-forward | Defer | Obligatorio | P1 |
| Purged CV / CPCV | Parcial | Completo | P1 |
| PBO | Lite | Completo | P1 |
| DSR / PSR / FDR | Parcial | Completo | P1 |
| Monte Carlo | Defer | Robustness Lab | P1 |
| Data lineage | Manifest | Data Layer completo | P0/P1 |
| Leakage firewall | Parcial | Obligatorio | P0 |
| Portfolio risk | Correlation | Risk Engine completo | P2 |
| Capacity | No | Obligatorio | P2 |
| Execution | Cost model | Simulator + adapters | P3 |
| Paper/Shadow/Micro-live | No | Lifecycle obligatorio | P4 |
| Monitoring | Dashboard research | Live monitoring | P4 |
| Reconciliation | No | Obligatorio | P3/P4 |
| Kill switches | No | Obligatorio | P4 |
| CI/CD | Tests puntuales | Regression framework | P3/P4 |
| Registry/governance | promoted.csv | Strategy Registry | P4 |
| Disaster recovery | No | Obligatorio para capital relevante | P5 |

---

## 31. Principio operativo final

El framework no debe optimizar para producir el mayor Sharpe histórico. Debe optimizar para producir **estrategias que tengan una alta probabilidad de seguir funcionando después de haber sido descubiertas, seleccionadas, ejecutadas y sometidas a costes reales**.

Por tanto, la métrica implícita de calidad del framework es: 

```text
Research Quality
× Statistical Robustness
× Execution Realism
× Portfolio Robustness
× Operational Reliability
```

Si cualquiera de estos factores es cercano a cero, el sistema completo es cercano a cero.


---

*Primera acción: copiar `config/bots_config.yaml` de DMRI a `pipeline_master.yaml` con este esquema, implementar `run_pyeventbt_once + scoring + synthetic` (P0) y pasar UNA estrategia × UN combo (ej. `EURUSD H1, n_trials=100, n_sim=20`) de punta a punta. El resto es escalar lo que ya funciona, no construir más teoría.*
