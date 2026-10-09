# Guía de onboarding y ejecución del framework PyEventBT + Pipeline robusto

> Para quién es: alguien nuevo en el proyecto que necesita ejecutar el
> framework de principio a fin. Cero conocimiento previo asumido salvo
> Python básico y nociones de trading (qué es un backtest).
> Idioma de comandos: PowerShell en Windows (el entorno de este proyecto).

---

## 0. Qué es esto (el mapa mental en 2 minutos)

Este repo tiene **dos capas** que no debes mezclar:

1. **Motor de backtest/live (`pyeventbt/`)** — framework event-driven para
   MetaTrader 5 en Python puro, sin MQL5.Principio: *one codebase, two modes*:
   el mismo código de estrategia corre en backtest y en live (mock completo
   de la API MT5). Pipeline de eventos:
   `BarEvent → Strategy (signal) → SignalEvent → Sizing → Risk → OrderEvent → Execution → FillEvent → Portfolio`.
   Los ejemplos `example_*.py` de la raíz usan **solo** esta capa.
2. **Pipeline de investigación (carpetas nuevas sobre el motor)** — proceso
   anti-overfitting para **una estrategia ya definida** (lógica cerrada,
   espacio de parámetros acotado). NO es data mining: no busca estrategias,
   busca *parámetros con edge en datos futuros*. Flujo:
   `[0 PREP] → [1 OPTUNA] → [2 CLUSTER] → [3 SYNTH+EDGE] → [3B WF/CPCV/PBO] → [4 SELECT] → [5 PORTFOLIO] → [6 LIVE] → [M CALIBRACIÓN]`.

Documentos de referencia (léelos en este orden):

| Doc | Para qué |
|---|---|
| `documentation/RESUMEN_NO_TECNICO_PIPELINE.md` | La idea en lenguaje no técnico |
| `documentation/PIPELINE_OPTIMIZACION_CLUSTERING_VALIDACION_FINAL.md` | **La teoría (v3.0, manda si hay conflicto)** |
| `documentation/TODO_DESARROLLADOR_PIPELINE.md` | Estado de implementación por etapas |
| `documentation/ASSUMPTIONS_DECISIONS.md` | Supuestos y decisiones tomadas (secciones A–T) |
| `AGENTS.md` | Instrucciones del repo (útil pero con matices: ver §10) |

---

## 1. Requisitos e instalación

- **Python 3.12** (`requires-python = ">=3.12,<3.13"`, ver `pyproject.toml` y `.python-version`).
- **Poetry** como gestor recomendado (`AGENTS.md`: `poetry install`).
- Datos históricos en disco (ver §3). Sin datos no hay pipeline.

```powershell
# 1) Clonar y entrar
git clone <repo> pyeventbt-private
cd pyeventbt-private

# 2) Instalar (motor + pipeline: optuna, joblib van en dependencias base)
poetry install

# 3) Extras opcionales del pipeline (arch, hdbscan, mlflow, dvc, psycopg2, pytest, rich)
poetry run pip install "pyeventbt[pipeline]"
# o con pip directamente:
pip install -e ".[pipeline]"
```

> Nota: `pytest` y `rich` viven en el extra `pipeline`, no en base.
> Para correr los tests necesitas al menos `pip install pytest`
> (los tests usan `tests/conftest.py`, que añade la raíz al `sys.path`).

**Regla de oro del entorno:** ejecuta SIEMPRE desde la raíz del repo.
`strategies/` no tiene `__init__.py` y el dispatcher la importa como paquete
de namespace (`from strategies.bbands_breakout.params import ...`); si corres
desde otro directorio, fallará el import.

Verificación rápida de que el entorno está sano:

```powershell
poetry run python -c "import pyeventbt, optuna, polars, sklearn, scipy; print('env OK')"
```

---

## 2. Mapa del repositorio (qué toca cada cosa)

```
pyeventbt-private/
├── pyeventbt/                  # MOTOR (no tocar sin motivo)
│   ├── strategy/               # Strategy + custom_signal_engine + backtest()
│   ├── data_provider/          # conector CSV MT5 (lee {SYMBOL}.csv)
│   ├── execution_engine/       # simulador MT5 (spread/comisión/swap internos)
│   ├── optimization/           # PIPELINE E1-2: adapter.py (run_pyeventbt_once),
│   │                           #   objective.py, runner_optuna.py, cluster.py, neff.py
│   ├── robustness/             # PIPELINE E3-3B: synthetic.py, scoring.py,
│   │                           #   validate_synthetic.py, null_control.py,
│   │                           #   edge_tests.py, dsr.py, fdr.py, cscv_pbo.py,
│   │                           #   cpcv.py, walkforward.py, decorrelation.py,
│   │                           #   power.py, report.py
│   ├── selection/              # PIPELINE E4: gates.py, lockbox.py, lots.py,
│   │                           #   sl_tp.py, promote.py
│   ├── calibration/            # PIPELINE M: null_strategies.py, planted_edge.py,
│   │                           #   report.py, run.py
│   ├── execution/              # PIPELINE E4: cost_models.py, stress_grid.py,
│   │                           #   tick_replay.py
│   ├── analytics/              # stats.py (summary canónico), regimes.py
│   ├── portfolio/              # PIPELINE E5: allocation.py, exposure.py,
│   │                           #   stress_corr.py, mc_envelope.py, capacity.py
│   ├── live/                   # PIPELINE E6: parity.py, rollout.py, monitor.py,
│   │                           #   risk_limits.py, reconcile.py  (librería, sin CLI)
│   ├── research/               # ledger.py (append-only), registry.py
│   └── runner/                 # master_config.py, process_log.py, data_manifest.py
├── strategies/
│   └── bbands_breakout/        # Estrategia de referencia del pipeline
│       ├── spec.yaml           # hipótesis pre-registrada + invalidez + frozen
│       ├── params.py           # BBandsParams (pydantic, extra=forbid) + SEARCH_SPACE
│       └── signal.py           # make_signal_fn() — factory pura, sin estado global
├── pipeline_master.yaml        # EL MAESTRO: todo configurable vive aquí (v3.0)
├── bots_dispatcher_optuna.py   # CLI del pipeline (etapas prep/optimize/cluster)
├── tests/                      # 66 tests (bloquean merge)
├── research/                   # (se crea al correr) ledger.parquet + lockbox_uses.yaml
├── runs/                       # (se crea al correr) artefactos por run
├── example_ma_crossover.py     # Ejemplos SOLO-motor (no usan el pipeline)
├── example_bbands_breakout.py  # Origen de strategies/bbands_breakout (ver §5.4)
└── documentation/              # Teoría + estado + supuestos + esta guía
```

---

## 3. Los datos (léelo antes de tocar nada)

### 3.1 Dónde viven y qué formato tienen

- Ruta configurada: `universe.data_dir` = `E:/Data_csv/MT5`.
- Un CSV por símbolo: `E:/Data_csv/MT5/AUDCAD.csv`.
  - La carpeta contiene **26 pares** (AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD,
    CADCHF, CHFJPY, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD, GBPAUD,
    GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD,
    USDCAD, USDCHF, USDJPY, ~130 MB cada uno, 2021-01/02 → 2026-10).
  - El maestro de referencia usa los 26 × M30/H1/H4/D1 (104 combos).
    Empieza con `--pair/--timeframe` antes del barrido completo.
- Formato MT5 **sin cabecera**: `YYYY.MM.DD,HH:MM,open,high,low,close,tickvol,volume,spread`
  (la hora viene **sin segundos**; el adaptador normaliza a `HH:MM:SS` al
  escribir su temp dir — supuesto A1).

### 3.2 El detalle que más dolores de cabeza da: pares de conversión

El simulador exige el par de conversión de divisa para cuenta USD
(ej. AUDCAD necesita AUDUSD y USDCAD). Esos CSV **sí existen** en la carpeta
actual, así que se usan los reales automáticamente.
Comportamiento del adaptador (`_ensure_aux_csvs`):

1. Si el CSV auxiliar existe en `data_dir`, lo recorta a la ventana y lo usa.
2. Si no existe, genera serie de **tasa constante** desde
   `universe.aux_fallback_rates` (AUDUSD 0.66, USDCAD 1.36) y lo registra en
   `aux_origins`.

Es neutro para *ranking* (Sharpe/PF/maxdd% invariantes a la constante), pero
la escala absoluta en $ queda aproximada. El fallback solo cubre huecos
puntuales. **Puerta:** el sizing de Etapa 4 y cualquier cuenta live exigen
tipos reales (supuesto B4).

---

## 4. Pruebas previas obligatorias (no te las saltes)

### 4.1 Suite completa: debe estar en verde antes de cualquier run

```powershell
poetry run pytest -q
# o sin poetry:
python -m pytest tests -q
```

- Son **66 tests** (verificado con `--collect-only`).
- Incluyen los que bloquean merge según `AGENTS.md`: scoring, cluster
  (medoide vs moda, eps auto), params_filter, synthetic_determinism,
  decorr_unilateral, lockbox_integrity, optuna_resume, fdr_resolution,
  embargo_time, gate_order, null_control, cscv_pbo, neff,
  ledger_append_only, constraints_func, live_risk_limits/reconcile/idempotency.
- Dos tests tocan el motor real y tardan unos segundos:
  `test_regression_run.py` (determinismo bit a bit del backtest) y
  `test_calibration_smoke.py` (calibración reducida, ~15 s). Si fallan,
  **para todo**: hay un problema de entorno o del core, no del pipeline.

### 4.2 Lo que los tests NO cubren (verifícalo a mano)

1. Que `E:/Data_csv/MT5/<SIMBOLO>.csv` existe y cubre tus ventanas IS/FORWARD.
2. Que `broker_server_tz` coincide con el servidor del bróker de producción
   (supuesto A3: `Europe/Athens` está **asumida, no verificada**).
3. Que las horas de sesión de tu estrategia (08:00 / 21:00) son hora servidor.

---

## 5. Configuración: el maestro y la estrategia

### 5.1 `pipeline_master.yaml` — el único sitio donde se configura

> Regla dura nº 1: **todo configurable vive en el maestro, nada hardcodeado
> en `.py`**. Si necesitas cambiar algo y no está en el YAML, el código está
> mal: añade el parámetro al maestro.

Bloques que **sí** vas a tocar como usuario nuevo:

| Bloque | Qué contiene | Ejemplo actual |
|---|---|---|
| `seed` | Semilla única (se deriva `seed+trial_number` por trial) | `1500` |
| `strategy` | nombre, `factory`, `params_class`, `spec`, `cash`, `search_space`, `frozen` | bbands_breakout, cash 10000 |
| `universe` | símbolos, TFs, `data_dir`, `windows` IS/FORWARD, `embargo`, `min_trades_IS`, `power` | AUDCAD H1, IS 2021-2023 |
| `costs` | **CONGELADOS en todo el pipeline** (comisión, spread base, slippage, delay, worst-case) | spread base 28.0 |
| `optimization` | `n_trials`, sampler, pruner, `constraints` laxas, objetivo robusto | 100 trials, TPE |
| `cluster`, `validation`, `walkforward_cpcv`, `decorrelation`, `execution_realism`, `gates`, `forward`, `promotion`, `portfolio`, `live`, `calibration` | Umbrales de cada etapa (ver §7) | — |

Bloques **inmutables dentro de un run**: `costs`, `windows`, `embargo`,
`seed`, `n_trials`. El `ProcessLog` los hashea al inicio
(`immutable_hash.txt`) y **aborta si cambian**. ¿Quieres probar otro valor?
Es un **run nuevo** con `resume: true` (reutiliza artefactos cuyo
`config_hash` coincida). Nunca edites el maestro a mitad de pipeline.

### 5.2 Ventanas, embargo y potencia (los tres conceptos que más confunden)

- **IS** = in-sample (se optimiza aquí). **FORWARD** = lockbox (no se toca
  hasta la Etapa 4.8, una sola vez, sin reintentos).
- **Embargo en tiempo** (no en barras): `max(max_holding, warmup) + extra_bars`
  en unidades del TF de señal. Se aplica entre IS/FORWARD y entre folds de
  WF/CPCV. Ejemplo real: el firewall detectó que FORWARD 2024-01-01 violaba el
  embargo de 40 h tras IS 2023-12-31 → se movió a 2024-01-03 (supuesto C4).
- **Potencia/MinTRL**: con `target_sharpe: 0.8` el MinTRL sale ≈ 3.8 años;
  con ~3 años de IS el combo nace probablemente **INCONCLUSIVE**. Eso es el
  veredicto honesto, no un fallo: la evidencia adicional debe venir de
  WF/CPCV + paper, **nunca de bajar umbrales** (supuesto C3).

### 5.3 Dar de alta una estrategia nueva (checklist)

La estrategia de referencia es `strategies/bbands_breakout/` (3 ficheros).
Para una estrategia nueva `strategies/<nombre>/`:

1. **`spec.yaml`** — pre-registro obligatorio: hipótesis con fundamento
   económico, universo, sesión, entrada **solo sobre barras cerradas**,
   salidas (SL siempre o time-stop declarado), filtros, `MaxOrdenes`,
   criterio de invalidez. **Cambiar la spec = nueva hipótesis = nuevo run.**
2. **`params.py`** — clase pydantic con `extra="forbid"` (clave desconocida o
   NaN ⇒ error), constantes `OPTIMIZABLE` y `SEARCH_SPACE` canónico, y
   funciones `filter_params()` + `params_hash()`. Todo lo estructural va
   como campo *frozen* con default (viaja para trazabilidad, no se optimiza).
3. **`signal.py`** — `make_signal_fn(params, strategy_id, signal_timeframe)`
   que devuelve `fn(event, modules)`. **Prohibido estado global mutable**:
   el estado diario vive en el closure de la factory (reentrante y
   paralelizable). La señal usa barras cerradas y
   `time_generated = event.datetime + TF` (`delay_bars=1`).
4. **Cablear en el maestro**: `strategy.name/factory/params_class/spec/
   search_space/frozen`. El `search_space` del maestro puede **restringir**
   el canónico, nunca ampliarlo sin nuevo run.
5. **No-lookahead**: la señal nunca usa la barra en formación ni
   `get_latest_tick` para decidir (ver `example_ma_crossover.py` como
   anti-ejemplo parcial: el firewall de datos cubre el feed, no el engine —
   supuesto B2).

Diferencia clave con `example_bbands_breakout.py` (supuesto D1): el ejemplo
usa globales mutables y parámetros globales; la versión pipeline es factory
pura + añade `MaxOrdenes=1` estricto (no opera si ya hay posición abierta).

### 5.4 Primer contacto recomendado: corre solo el PREP

Antes del run grande, valida datos + manifiesto + potencia con un combo:

```powershell
poetry run python bots_dispatcher_optuna.py pipeline_master.yaml --only prep --pair AUDCAD --timeframe H1
```

Esto crea `runs/<estrategia>/<timestamp>_<githash>/` con `00_prep/`
(`spec.yaml`, `master_used.yaml`, `manifests/`, `budget.json`, `power.json`) y un backtest **nominal**
(params por defecto) que comprueba `min_trades_IS`. Lee `power.json`:
`READY` o `INCONCLUSIVE`, y `budget.json` (`range_ok` debe ser true:
50–100 trials por dimensión).

> Consejo para experimentar sin contaminar: el ledger (`research/ledger.parquet`)
> es **global y append-only** (cuenta para `S_global`). Para pruebas, copia el
> maestro a `pipeline_smoke.yaml` con ventanas cortas, `n_trials: 4` y
> `research_ledger.path` apuntando a un temporal. Cambiar el maestro = run
> nuevo, así que esto es legítimo y no rompe nada.

---

## 6. Ejecución del pipeline, etapa por etapa

Orden y qué mirar en cada una. Comandos con `poetry run` (quítalo si usas
el venv activo).

### Etapa 0 — PREP ✅ (automatizada en el dispatcher)

Ya la corriste en §5.4. Artefactos: `00_prep/`, `master_used.yaml`,
`master_hash.txt`, `process_logs.yaml`. Si `data_check` o la pureza de
ventanas da FAIL, el run se detiene aquí con diagnóstico. Bien: es lo que
tiene que hacer.

### Etapa 1 — OPTUNA (automatizada)

```powershell
poetry run python bots_dispatcher_optuna.py pipeline_master.yaml --only optimize --pair AUDCAD --timeframe H1
# Reanudar / continuar: el ProcessLog salta lo ya `done` con mismo config_hash
poetry run python bots_dispatcher_optuna.py pipeline_master.yaml --only optimize,cluster --run-dir runs/bbands_breakout/<ts>_<hash>
```

Qué hace (`pyeventbt/optimization/`): un único adaptador
`run_pyeventbt_once(params, combo, *, costs, window, cash, seed, ...)` que
valida params contra tu `params_class`, escribe OHLC a un temp dir M1-MT5 y
corre `Strategy + signal_fn + backtest()` **idéntico para todos los trials**.
Objetivo robusto `mean(SR_k) − λ·std(SR_k)` en 4 subventanas (busca mesetas,
no picos); restricciones laxas vía `constraints_func` del sampler
(`min_trades 100, maxdd 25%, PF 1.0, exposure 80%`) — los infactibles se
**conservan** (son evidencia para DSR/N_eff/PBO), no se podan.

Salida en `optimization/<SYM>_<TF>/`: `trials.parquet` (TODOS los trials),
`daily_returns.parquet` (fechas × trials), `candidatos.csv` (top-k tras
filtro duro). S y K se vuelcan al ledger global.

### Etapa 2 — CLUSTER (automatizada)

```powershell
poetry run python bots_dispatcher_optuna.py pipeline_master.yaml --only cluster --pair AUDCAD --timeframe H1 --run-dir runs/...
```

Qué hace (`cluster.py`, `neff.py`): scaler robusto (IQR), DBSCAN con `eps`
auto (k-distance p90), representante **medoide** (punto real, nunca la moda),
singletons top-decila conservados, `N_eff` por correlación de retornos
(con TPE, S=400 no son 400 hipótesis independientes). Salida en
`cluster/<SYM>_<TF>/`: CSV de representantes, `neff.json`, `diagnostics/`
(`kdistance.png`, `sizes_hist.png`, `pca2d.png`). El DSR se reporta con S
(conservador) y `N_eff` (optimista); el veto usa `N_eff`.

### Etapa 3 — SYNTH + EDGE (automatizada en el dispatcher)

Por representante (`--max-reps` limita el coste): `run_consistency()` con
`n_sim_screening` sims GJR-GARCH-t a **M1**

- **(a) Consistencia**: 1 backtest real + N sintéticos GJR-GARCH-t a **M1**
  (siempre a M1: simular a H1 no es neutro — supuesto F1), `score_percentile_rank`
  por métrica en banda [10,90]. Rank bajo = FAIL; rank alto = ÁMBAR
  (overfitting *o* edge real: lo resuelven el edge test + 3B).
- **(b) Control negativo**: la estrategia sobre generador nulo `iid_residuals`.
  Gate **unilateral hacia arriba**: lo que invalida el run es *ganar* en nulo
  (sesgo); *perder* es cost-drag esperado (supuesto F2).
- **(c) Edge**: SPA / Reality Check / Romano-Wolf con stationary bootstrap
  (≥2000 remuestreos) sobre la shortlist, familia = ledger (`S_global`).
- **(d) DSR/PSR** con skew/kurtosis/T reales. **Veto: `DSR(N_eff) < 0.20` ⇒ FAIL.**
- **(e) FDR** (Benjamini-Hochberg, q=0.10) **solo sobre p del edge test**,
  con check de resolución (`p_min ≤ q/C` o el pipeline **aborta**).
- Re-scoring estresado `spread×2, +1 tick` sin re-simular.

Veredicto por set: VERDE / ÁMBAR / ROJO. Pasan a 3B verdes + ámbares con
`edge_p_adj < alpha`.

### Etapa 3B — WF / CPCV / PBO (automatizada en el dispatcher)

- `cscv_pbo()`: PBO sobre la matriz de trials. **Gate `PBO ≤ 0.30`.**
- `cpcv_distribution()`: por superviviente. **Gate: mediana > 0 y ≥70% caminos positivos.**
- `walkforward()`: anclado, train creciente, test 12 m, mini-Optuna por fold.
  **Gate: `WFE ≥ 0.5` y ≥70% folds positivos.** Todos los trials de folds
  cuentan en el ledger. Mide el *procedimiento*, no un set: si falla, el
  problema está en el espacio o la estrategia.

### Etapa 4 — SELECT (automatizada; **orden obligatorio garantizado por construcción**)

```
selección → jitter ±10% → replay + grid estrés → PUERTAS → LOCKBOX → sizing/SL-TP
```

1. `robustness/decorrelation.py` (`greedy_select`): sobre retornos diarios,
   unilateral `rho ≤ 0.7` (el negativo se queda: diversifica), score
   `0.4·sharpe + 0.3·pf + 0.2·r2 − 0.1·turnover`, hasta `max_sets=5`.
2. `jitter_grid` + `jitter_verdict`: ≥80% vecinos con `sharpe>0, pf≥1.0`.
3. `execution/` (`stress_cells`, `cell_cost_cfg`, `stress_verdict`,
   `tick_coverage`): grid `spread×{1,1.5,2} × slippage+{0,1,2}`; pasa si PF
   peor celda ≥ 1.05 y decay de Sharpe ≤ 50%. Sin ticks del bróker, el replay
   es el feed M1 del bróker + cobertura de tickvol.
4. **PUERTAS** (`selection/gates.py`, ANTES del lockbox): `PF≥1.3`
   (1.1–1.3 solo con retest ≥1.3), `R2≥0.5`, `DD<365d`, `≤10/12` rojos,
   régimen ≥60% celdas positivas sin familia dependiente de una celda.
5. **LOCKBOX** (`selection/lockbox.py`): solo los que pasaron puertas ven
   FORWARD **una vez, sin reintentos**: `pf≥1.1 + sharpe>0`, PSR con Holm
   sobre los usos del programa, registro en `lockbox_uses.yaml`.
   `guard_gates_before_lockbox()` revienta si inviertes el orden
   (puertas-después-del-lockbox = run inválido).
6. **Sizing** (`lots.py`, `sl_tp.py`, `promote.py`): `Lots=min(...)`, SL/TP
   percentil MFE/MAE (aprox H1) o ATR — nunca min/max. `promoted.csv` con
   trazabilidad completa (params, gates, ranks, DSR, edge, FDR, PBO/WFE,
   jitter, estrés, régimen, forward+Holm, lots, SL/TP, hashes).

### Etapa 5 — PORTFOLIO (librería)

`portfolio/`: `allocate()` (HRP + Ledoit-Wolf + vol targeting + Kelly cap +
peso máximo; **guarda: `max_weight·n ≥ 1` o configuración inválida**),
`currency_exposure()` (neto por divisa), `stress_correlation()`,
`mc_dd_envelope()` (envolventes p95/p99 que usará la monitorización),
`capacity_curve()` + `marginal_benefit()` (solo entra si mejora Sharpe/DD
neto de costes).

### Etapa 6 — LIVE (librería, **nada con dinero real sin 6.1+6.4 operativos**)

`live/`: `parity.py` (mismatch ≤2%, slippage p95 ≤2 ticks, tracking error
≤0.5%), `rollout.py` (paper 60d/30 trades → micro → scaled → full, avance
solo con paridad OK + envolvente OK + cero incidentes; champion/challenger),
`monitor.py` (Sharpe rodante vs p5, DD vs envolvente, CUSUM, KS, health score
GREEN/YELLOW/ORANGE/RED, retirada pre-registrada), `risk_limits.py`
(jerarquía trade/día/estrategia/cartera + heartbeat/stale/spread, kill manual
por encima de todo, capa independiente de la estrategia), `reconcile.py`
(cada 60 s vs bróker, mismatch ⇒ halt_and_alert, órdenes idempotentes
magic+comentario con estado persistido).

### Meta-etapa M — CALIBRACIÓN (antes de fiarte de un umbral)

```powershell
# Smoke (~15 s, para CI y primer contacto)
poetry run python -m pyeventbt.calibration.run pipeline_master.yaml --smoke --pair AUDCAD
# Completa (cara: 200 nulas + edge plantado × niveles × seeds)
poetry run python -m pyeventbt.calibration.run pipeline_master.yaml --pair AUDCAD
```

Mide el **pipeline entero**: FPR con estrategias nulas por construcción
(`random_entries`, `shuffled_signals`, `indicator_on_noise` ×
`iid_synthetic`/`real_block_permuted`) y potencia con edge plantado
(`drift_rider` LONG-only sobre drift compuesto). Emite
`calibration_report.json` (FPR por etapa, curva de potencia, veredicto
VALID/INVALID vs targets `FPR ≤ 5%`, potencia a SR=1 ≥ 70%). Referencia
actual: FPR puertas 0/12; potencia 0 en niveles brutos bajos **por costes**,
no por bug (ver §9).

### Reporte agregado de un run

```powershell
poetry run python -m pyeventbt.robustness.report runs/bbands_breakout/<ts>_<hash> --full
# emite gate_report.json en el run
```

---

## 7. Cómo leer un resultado (gates de un vistazo)

| Etapa | Gate | Umbral |
|---|---|---|
| PREP | potencia | años_IS ≥ MinTRL o INCONCLUSIVE |
| OPTUNA | constraints laxas | trades≥100, maxdd≤25%, exposure≤80%, PF≥1.0 |
| SYNTH | consistencia / nulo / edge / DSR / FDR | rank [10,90], nulo unilateral, `edge_p_adj<0.10`, DSR≥0.20 |
| 3B | PBO / WFE / CPCV | ≤0.30 / ≥0.5+70% / mediana>0+70% |
| SELECT | puertas → lockbox | PF≥1.3, R2≥0.5, DD<365d, ≤10/12 rojos, régimen 60% → pf≥1.1+sharpe>0 (Holm) |
| PORTFOLIO | admisión | beneficio marginal + capacity ≤25% decay |
| LIVE | paridad/rollout | mismatch≤2%, 60d paper → micro → … |
| M | FPR/potencia | ≤5% / ≥70% a SR=1 |

`promoted.csv` + `calibration_report.json` vigente = DONE de investigación.
Menos que eso es un informe, no una selección.

---

## 8. Reglas duras (no negociar, resumen de `AGENTS.md` §5)

1. Maestro único; `costs/windows/embargo/seed/n_trials` inmutables en el run.
2. Embargo en tiempo entre IS/FORWARD y entre folds.
3. Orden Etapa 4: puertas → lockbox (una vez, Holm) → sizing. Al revés = inválido.
4. Ledger append-only; `S_global` manda sobre S del run.
5. Rank [10,90] reinterpretado; FDR solo sobre p del edge test; veto DSR.
6. Barras cerradas, `delay_bars=1`, costes congelados.
7. Reproducibilidad: hash de configs, determinismo (`deterministic_mode`, 1 worker).

---

## 9. El hallazgo que debes conocer antes de ilusionarte (sec. I)

Con spread medio 28 pts + comisión en AUDCAD, un round-trip diario cuesta
~0.7 USD a lote mínimo frente a un drift SR=3 (~0.31 USD/trade). La potencia
neta es 0 hasta edge bruto ~SR10 (medido en AUDCAD H1, spread 28).
Opciones: (a) par más barato — **ya disponible: EURUSD spread 6**,
GBPUSD/AUDUSD también baratos —, (b) holding
multi-día (el time-stop diario devora costes), (c) aceptar el rechazo honesto.
Recomendación quant del proyecto: (b)+(a).

---

## 10. Discrepancias conocidas entre docs y código (para que no te pierdas)

- `AGENTS.md` cita `python -m calibration.run`, `python -m robustness.report`
  y `live.monitor` como CLI. **Lo real**: `python -m pyeventbt.calibration.run`,
  `python -m pyeventbt.robustness.report`; **`live/monitor.py` es librería,
  no tiene CLI** (pendiente, ver sec. T).
- El dispatcher automatiza `prep, optimize, cluster, synth, wfcpcv, select,
  calibrate` con resume (por combo y por representante). El lockbox exige
  `--auto-lockbox`; sin ella los sets quedan en `AWAITING_LOCKBOX`.
  Portfolio y live siguen siendo librería (Etapas 5-6).
- `research/` y `runs/` **no existen** en un checkout limpio: se crean al
  correr (están en `.gitignore`, son estado, no código).

## 11. Troubleshooting (errores que ya nos pasaron)

| Síntoma | Causa probable | Fix |
|---|---|---|
| `FileNotFoundError: .../AUDUSD.csv` | falta par de conversión | normal: usa fallback (B4) o añade el CSV real |
| `RuntimeError: lockbox ya consumido` | segundo uso del FORWARD | correcto: es un run nuevo, no un reintento |
| `ABORT: costs/windows/... cambiaron` | editaste el maestro en un run vivo | nuevo run con `resume: true` |
| `candidatos.csv` vacío (K=0) | ventana corta: nadie llega a 100 trades | ventana mayor o smoke consciente (los infactibles se conservan igual) |
| `FDR ... aborta` | `p_min > q/C` (pocos remuestreos) | subir `n_resamples` o bajar C |
| `max_weight·n < 1` | cap infactible | subir cap o añadir estrategias |
| Tests rojos tras importar pyeventbt | floats YAML → Decimal (core) | `load_master` ya normaliza; no toques el core |
| Sharpe/PF raros en $ pero ranks OK | `aux_fallback_rates` constante | esperado (B4); tipos reales para sizing/live |

---

*Última actualización: tras commit `f900ab1` (66 tests en verde). Si algo de
esta guía contradice al código, manda el código; si contradice a la teoría,
manda `PIPELINE_OPTIMIZACION_CLUSTERING_VALIDACION_FINAL.md`.*
