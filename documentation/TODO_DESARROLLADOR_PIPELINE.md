# Pipeline — TODO list para desarrollador (por etapas)

Fuente: `PIPELINE_OPTIMIZACION_CLUSTERING_VALIDACION_FINAL.md` v3.0. Todo configurable vive en `pipeline_master.yaml`. Nada hardcodeado en `.py`.

Estructura de código esperada:
`pyeventbt/optimization|robustness|calibration|execution|analytics|selection|portfolio|live|research|runner` + `pipeline_master.yaml` + `bots_dispatcher_optuna.py`.

---

## Etapa 0 — PREP: spec, datos, presupuesto, potencia, ledger
- [ ] Crear `strategies/<est>/spec.yaml`: hipótesis, universo, sesión, entrada sobre barras cerradas, salidas (SL siempre o time-stop), filtros (`spread_max`, ATR, noticias, `MaxOrdenes=1`), criterio de invalidez.
- [ ] Crear `pipeline_master.yaml` según esquema §2 (strategy, universe, embargo en tiempo, costs congelados, seed, `n_trials`, optimization, cluster, validation, walkforward_cpcv, decorrelation, execution_realism, gates, forward, promotion, portfolio, live, calibration, runtime).
- [ ] Implementar `runner/master_config.py`: validación, hash de config, aborta si `costs/windows/embargo/seed/n_trials` cambian a mitad de run. Regla: cambiar maestro = nuevo run con `resume:true`.
- [ ] Implementar `DATA_MANIFEST.json` por símbolo: fuente, filas, gaps, spread medio/p95, hash + `broker_server_tz` y DST. `data_check` debe fallar ante huecos/duplicados de DST y sesiones que no coinciden con broker de producción.
- [ ] Implementar embargo en tiempo: `max(max_holding, warmup) + extra_bars` en unidad del TF señal. Aplicar entre IS/FORWARD y entre folds WF/CPCV. Test `test_embargo_time`.
- [ ] Implementar leakage firewall como tests bloqueantes (FAIL, no warning): look-ahead, timestamps desordenados, barras incompletas, duplicados, joins temporales mal, uso accidental de lockbox, survivorship. Tests `test_*` correspondientes.
- [ ] Implementar `budget.json` + registro en ledger: `n_trials` fijado antes (regla 50-100 por dimensión), `seed` única con derivación `seed+trial_number`. Storage: PostgreSQL o JournalStorage si `n_jobs>1` (SQLite solo 1 worker).
- [ ] Implementar `power.py` → `power.json` por combo: años efectivos, nº trades, SE Sharpe, MinTRL, potencia. Si `años_IS < MinTRL` marcar `INCONCLUSIVE` (nunca aprobar por defecto).
- [ ] Implementar `research/ledger.py`: `research/ledger.parquet` append-only (`run_id, strategy_family, combo, trial_number, params_hash, window, sharpe_net, n_trades, daily_returns_ref, timestamp`) + `lockbox_uses.yaml`. `S_global` = trials del programa sobre misma ventana. Test `test_ledger_append_only`.
- [ ] Salida: `runs/.../00_prep/{spec.yaml, master_copy.yaml, manifests/, data_check.html, budget.json, power.json}` + veredicto `READY/INCONCLUSIVE`.

## Etapa 1 — OPTUNA
- [ ] Implementar adaptador único `run_pyeventbt_once(params, combo, *, costs, window, cash, seed) -> (pnl_df, trades_df)`: valida contra `params_class` pydantic (falla ante clave desconocida/NaN), escribe OHLC real o sintético a temp dir M1-MT5, corre `Strategy()+signal_fn+backtest()`, devuelve frames + `summary()` + retornos diarios. Modos: real|sintético|nulo|tick_replay.
- [ ] Implementar `optimization/objective.py` según §1.1: screening barato sobre prefijo IS (`early_gate`), backtest completo idéntico para todos, guardar todos los trials (también podados/infactibles con `feasible/pruned_at`), constraints vía `constraints_func` (no `TrialPruned` tras backtest completo), objetivo robusto `mean(SR_k)-lambda*std(SR_k)` sobre 4 subventanas.
- [ ] Configurar sampler TPE/NSGA-II con `constraints_func`, `pruner: median` (desactivar si backtest <10s; Hyperband si >30s). `n_jobs=-1` solo a nivel trial. Verificar soporte de `constraints_func` en versión Optuna instalada.
- [ ] Guardar `trials.parquet` + `daily_returns.parquet` (fechas × trials, hash auditado). Filtro duro (`n_trades>=100, maxdd<=25%, exposure<=80%, pf>=1.0`) → Pareto o top-k (`k=3×max_clusters_kept`) → `candidatos.csv`. Volcar S y K al ledger.
- [ ] Tests: `test_optuna_resume` (20+20==40), `test_constraints_func`, `test_params_filter`, `test_regression_run` (deterministic_mode, n_trials=20).

## Etapa 2 — CLUSTER
- [ ] Implementar `optimization/cluster.py`: scaler robusto (IQR), categorical one-hot, redondeo a `step` válido con re-validación.
- [ ] DBSCAN con `eps` auto (k-distance p90, registrar `eps_auto`, `min_samples=5`, `metric=manhattan`) + opción HDBSCAN.
- [ ] Representante medoide (no moda); alternativas `best_sharpe/ensemble3`. `keep_singleton_outliers:true` (top-decila Sharpe se conserva). Opción `params_plus_behavior` (params + retornos mensuales peso 0.3).
- [ ] Implementar `optimization/neff.py` (ONC/jerárquico sobre `1-rho` de `daily_returns`) → `cluster/neff.json {S, N_eff, method}`. Reportar DSR con S y N_eff.
- [ ] Salidas + diagnósticos: `cluster/<EST>_<PAIR>_<TF>.csv`, `diagnostics/{kdistance.png, sizes_hist.png, pca2d.png, members/}`.
- [ ] Tests: `test_cluster` (medoide vs moda, eps auto), `test_neff`.

## Etapa 3 — SYNTH + EDGE
- [ ] Implementar `robustness/synthetic.py`: `SyntheticMarketGenerator` GARCH(1,1,1)-t con asimétrico (`@njit`), `generate_universe`, `build_ohlc_fast`, cache por (símbolo,TF), `to_csv()` M1-MT5. Validar que replica vol/autocorr(|r|)/kurtosis/rango diario ±20%. Spread sintético = `spread_medio_real×(1+estrés)`. Generador nulo `iid_residuals`. Soporte stationary bootstrap p=0.1. Seeds `(master.seed, cluster_id, sim_idx)`.
- [ ] Implementar `robustness/scoring.py` + `validate_synthetic.py`: 1 backtest real + N sintéticos (20 screening, 50 finalistas), `score_percentile_rank` por métrica (Sharpe, PF, #Trades), banda [10,90], quorum 0.5, bordes (NaN/inf→None, varianza-cero→50). Semántica: rank bajo=FAIL, rank alto=ÁMBAR (lo resuelve edge test + 3B).
- [ ] Implementar `robustness/null_control.py`: 200 sims sobre generador nulo; a nivel pool `|Sharpe medio|<=0.15` y ≤10% significativos, si no STOP del run; a nivel set Sharpe nulo positivo significativo=FAIL. Test `test_null_control`.
- [ ] Implementar `robustness/edge_tests.py`: shortlist top-40, White RC / Hansen SPA / Romano-Wolf stepdown con stationary bootstrap ≥2000 remuestreos (o permutación sign-flip), `edge_p_adj`, `alpha_fwer=0.10`, familia=ledger (`S_global`). Test `test_fdr_resolution`.
- [ ] Implementar `robustness/dsr.py` (PSR/DSR con skew/kurtosis/T, veto `DSR(N_eff)<0.20`, haircut Harvey-Liu) y `robustness/fdr.py` (BH q=0.10 sobre p del edge test, check `p_min<=q/C` o aborta).
- [ ] Re-scoring estresado sin re-simular: `spread×2,+1 tick`; `stress FAIL+base PASS`=ámbar mitad prioridad.
- [ ] Salidas: `synthetic_validation/<...>.csv (ranks, dsr_S, dsr_neff, edge_p_adj, fdr_adj_p, haircut, null_sharpe_mean, verdict)` + `null_control/ + edge_tests/` + plots `real_vs_synth.png, deviation_distribution.png`.
- [ ] Veredicto por set VERDE/ÁMBAR/ROJO según §3.2. Pasan a 3B verdes + ámbares con `edge_p_adj<alpha`.
- [ ] Tests: `test_scoring`, `test_synthetic_determinism`, `test_calibration_smoke`.

## Etapa 3B — WF / CPCV / PBO
- [ ] Implementar `robustness/cscv_pbo.py`: matriz daily_returns → 10 bloques → mitades train/test con purga/embargo → PBO, logits, degradación IS→OOS. Gate `PBO<=0.30`. Test `test_cscv_pbo`.
- [ ] Implementar `robustness/cpcv.py`: por superviviente, distribución Sharpe en caminos CPCV (`n_groups=10, k_test=2, purge=from_embargo`). Gate mediana>0 y ≥70% caminos positivos.
- [ ] Implementar `robustness/walkforward.py`: anclado, train creciente, test 12m, ~5 folds, mini-pipeline completo por fold (Optuna reducida `n_trials_per_fold=100` + cluster + selección) con embargo. `WFE=Sharpe_test/Sharpe_train>=0.5` y ≥70% folds positivos. Todos los trials de folds cuentan en ledger.
- [ ] Salidas: `wf_cpcv/<...>.csv` + `lambda_logits.png + oos_degradation.png`. Si PBO/WFE fuera de gate → ámbar/rojo, no forzar Etapa 4.

## Etapa 4 — SELECT: decorrelación, ejecución, puertas, lockbox, sizing
- [ ] Implementar `robustness/decorrelation.py`: curvas en real con mismos costes, `resample:D`, métricas (`n_trades, pf, r2, sharpe_net, calmar, turnover`); prefiltro `n_trades<30`/varianza-cero fuera; correlación sobre retornos (pearson|spearman), unilateral `rho<=0.7` (negativo se queda); score `0.4*sharpe+0.3*pf_winsor_p95+0.2*r2-0.1*turnover`; greedy/cluster hasta `max_sets=5`; reportar `n_efectivo=1/sum(w²)`. Test `test_decorr_unilateral`.
- [ ] Implementar jitter ±10%, `min_green=0.8` (`sharpe>0, pf>=1.0`); si falla se sustituye por siguiente del ranking.
- [ ] Implementar `execution/{tick_replay.py, cost_models.py, stress_grid.py}`: replay con ticks del broker (`intrabar_order:worst_case`, `min_coverage=0.95`), grid `spread×{1,1.5,2}×slippage+{0,1,2}×delay{1,2}`. Pasa si PF peor celda>=1.05 y decay Sharpe<=50%. Si gap barra-vs-tick>15% documentar (ticks mandan).
- [ ] Implementar `selection/gates.py` ANTES del lockbox: `PF>=1.3` (o 1.1-1.3+retest 3 salidas MA/BB/RSI>=1.3), `R2>=0.5`, `DD<365d`, `<=10/12` rojos, régimen (año, terciles vol, sesión): ≥60% celdas Sharpe>0, ninguna >50% PnL. Test `test_gate_order`.
- [ ] Implementar `selection/lockbox.py`: solo pasan-puertas ven FORWARD una vez, sin reintentos: `pf>=1.1+sharpe_net>0`, PSR corregido Holm sobre usos del programa, registro en `lockbox_uses.yaml`, test integridad CI (acceso temprano=FAIL). Test `test_lockbox_integrity`.
- [ ] Implementar `selection/{lots.py, sl_tp.py, promote.py}`: `Lots=min(Lots_dd, Lots_max_los, Lots_mean_loss)`, SL/TP percentil MFE/MAE p10/p90 o ATR (nunca min/max). Generar `promoted.csv` con `params+gates+ranks+dsr(S,N_eff)+edge_p_adj+fdr+pbo/wfe+jitter+peor celda+régimen+forward(Holm)+lots+SL/TP+cost_cfg+manifest_hashes+master_hash+ledger_snapshot_hash+commit+magic`.
- [ ] ORDEN OBLIGATORIO: selección → jitter → replay → puertas → lockbox → sizing (nunca puertas después del lockbox).

## Meta-etapa M — CALIBRACIÓN
- [ ] Implementar `calibration/{null_strategies.py, planted_edge.py, report.py}`: 200 nulas (`random_entries, shuffled_signals, indicator_on_noise`) sobre `iid_synthetic` y `real_block_permuted`; edge plantado con drift para Sharpe {0.3,0.6,1.0} ×5 seeds.
- [ ] Medir FPR (objetivo ≤5%) y potencia a SR=1 (objetivo ≥70%), FPR por etapa, curva de potencia. Congelar umbrales en `calibration_report.json` referenciado por `master_hash`. Re-ejecutar ante cambio de umbrales/generador/versión mayor. CI con calibración reducida.

## Etapa 5 — PORTFOLIO
- [ ] Implementar `portfolio/{allocation.py, exposure.py, stress_corr.py, mc_envelope.py, capacity.py}`: universo = todos los `promoted.csv`; correlación media + estrés (peor decil) + clustering jerárquico entre estrategias.
- [ ] Exposición neta por divisa (`max_currency_net_exposure`), asignación HRP/risk parity con covarianza Ledoit-Wolf, vol targeting anual, Kelly fraccional cap 0.25, peso máximo por estrategia, rebalanceo mensual con banda.
- [ ] Límites (DD cartera, apalancamiento, nº posiciones, margen) + MC bloques (5000 caminos, envolventes p95/p99: `P(DD>X)`, ES, max periodo perdedor, recovery time). Capacity ligero: grid `{10k,50k,250k,1M}`, decay Sharpe ≤25%.
- [ ] Regla de admisión: solo si mejora Sharpe/DD de cartera neto de costes (beneficio marginal).

## Etapa 6 — LIVE
- [ ] Implementar `live/parity.py`: replay diario real vs backtester mismos params; métricas ≤2% señales discordantes, slippage p95 ≤2 ticks, tracking error ≤0.5%. Chequear tz servidor, requotes, sufijos, tick size/lot step/swaps, cierres.
- [ ] Implementar `live/rollout.py`: `paper(60d,30 trades)→micro×0.1(60d)→scaled×0.5(90d)→full`, avance solo con paridad OK + Sharpe/DD en envolvente + cero incidentes. Champion/challenger para versiones. Nunca params en caliente.
- [ ] Implementar `live/monitor.py`: Sharpe rodante vs p5, DD vs envolvente MC (alerta p95, halt p99), CUSUM/SPRT (k=0.5,h=5), deriva frecuencia/duración/hit rate/slippage/spread, KS retornos. Health score GREEN/YELLOW/ORANGE/RED. Tests live incluidos.
- [ ] Implementar `live/risk_limits.py`: jerarquía por trade/día/estrategia/cartera + operacionales (heartbeat 30s, stale 120s, spread máx 30). Acciones `halt/flatten` + kill manual. Capa independiente de la estrategia. Test `test_live_risk_limits`.
- [ ] Implementar `live/reconcile.py`: cada 60s compara posiciones/órdenes/equity vs broker; mismatch→`halt_and_alert`; órdenes idempotentes (magic+comentario único). Persistir estado. Tests `test_live_reconcile`, `test_order_idempotency`.
- [ ] Política retirada pre-registrada: DD>p99 o 2 CUSUM o Sharpe<p5 30d → paper/shadow; re-opt = run nuevo completo. Infra: VPS, NTP, log inmutable, backups, verificar soporte MT5 de PyEventBT.

## Transversal — Runner, gobernanza, tests, DONE
- [ ] Implementar `runner/{stage.py, process_log.py, master_config.py, dashboard/}`: `Stage(name, discover_inputs, run, persist)`, `ProcessLog` atómico (`tmp+os.replace`), resume por `config_hash`, dashboard Rich sin `print`. Layout `runs/{est}/{ts}_{githash}/` + `research/` + `live/` según §Orquestación.
- [ ] Implementar `research/registry.py` + Approval Package (§Gobernanza): estados `RESEARCH→CANDIDATE→VALIDATED→APPROVED→PAPER→SHADOW→MICRO→LIVE (+DEGRADED/QUARANTINED/RETIRED)` con gates; `approval_package/` con hashes.
- [ ] Dependencias opt: `pip install optuna arch joblib psycopg2-binary mlflow dvc` (+ `pyeventbt[opt]`). CLI: `bots_dispatcher_optuna.py`, `--only`, `calibration.run`, `robustness.report --full`, `live.monitor`. Repro: Docker digest + lockfile, DVC, MLflow, CI regresión + behavioural regression (mejora brusca Sharpe dispara alerta).
- [ ] DONE investigación: `promoted.csv` con trazabilidad completa + `calibration_report.json` vigente. DONE producción: cartera + paridad + kills + rollout + monitorización activa.
- [ ] Prioridades: P0 adaptador+scoring+sintético+objetivo (con correcciones §3.2, FDR, orden puertas→lockbox, embargo tiempo) → P1 daily_returns+N_eff+DSR+WF/CPCV/PBO+potencia+M mínima (20-50 nulas) → P2 ejecución+régimen+portfolio → P3 live. Defer: market impact/queue, data layer 6 capas, event store, multi-estrategia distribuida.
