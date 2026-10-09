# Supuestos, decisiones y notas quant (documento vivo)

> Creado al iniciar la Etapa 0. Todo supuesto con impacto estadístico o de
> ejecución queda registrado aquí con fecha. Revisar antes de fiar de un umbral.

## A. Datos y mercado
- **A1 (2026-10-08):** `E:\Data_csv\MT5\*.csv` formato MT5 sin cabecera
  `YYYY.MM.DD,HH:MM,OHLC,tickvol,volume,spread` (hora SIN segundos). El conector
  `csv_data_connector.py` parsea con `%H:%M:%S` estricto → el adaptador
  `run_pyeventbt_once` normaliza a `HH:MM:SS` al escribir el temp dir. Si el
  broker cambia el export, `data_check` debe fallar (nuevo formato = nuevo manifiesto).
- **A2:** Solo hay `AUDCAD` (136MB, 2021-01→2026-10) y `AUDCHF` (4.5GB). No hay
  EURUSD en el disco de datos → el combo de referencia es **bbands_breakout /
  AUDCAD / H1**. EURUSD existe solo como dataset empaquetado de ejemplo.
- **A3:** `broker_server_tz=Europe/Athens` asumida (típica MT5 EET/EET+DST); NO
  verificada contra el broker de producción. Antes de LIVE exigir confirmación
  del servidor real y re-chequear DST/sesiones (21:00/08:00 son hora servidor).
- **A4:** Gaps intradiarios (ej. faltan minutos sueltos) se forward-fillean por el
  conector; `data_check` cuenta gaps>1min como WARN (fines de semana/feriados),
  FAIL solo ante duplicados/NaN/cero filas. Revisar: si un símbolo ilíquido
  tiene gaps grandes entre semana, el fill-forward subestima el riesgo de gap.
- **A5:** `file_hash_sampled` (tamaño + primer/último MB) en vez de SHA completo:
  suficiente para detectar cambios de dataset en el flujo diario; para el
  `approval_package` final usar hash completo (anotado como mejora).

## B. Costes y realismo de ejecución
- **B1:** PyEventBT simula spread vía `SymbolInfo.spread`; comisión/slippage por
  tick NO son parámetros nativos del backtester. El adaptador aplica
  `spread_mult` vía `symbols_info_override` y registra el resto de costes en
  `cost_cfg`; la Etapa 4 añade grid de estrés + haircut post-trade documentado.
  Riesgo: subestimar costes si el haircut no se aplica. TODO: verificar contra
  `trade_deal.commission/swap` reales del primer backtest y ajustar.
- **B2:** `delay_bars=1` se cumple porque la señal usa barras cerradas y
  `time_generated = event.datetime + TF`. Pendiente de verificación formal con
  test de look-ahead sobre el feed (firewall actual cubre datos, no el engine).
- **B3:** `intrabar_order=worst_case` (SL y TP en la misma barra gana el peor
  caso) es configuración del motor; el tick_replay de la Etapa 4 lo audita.
- **B4 (2026-10-08):** el simulador exige los pares de conversión de divisa
  (AUDUSD, USDCAD para cuenta USD) y no existen en `E:\Data_csv\MT5`. El
  adaptador los genera como **tasa constante** (`universe.aux_fallback_rates`;
  AUDUSD 0.66, USDCAD 1.36, medias aproximadas del periodo) y lo registra en
  `aux_origins`. Demostrablemente neutro para ranking: margen/comisión/swap por
  lote constantes ⇒ Sharpe/PF/maxdd% invariantes; solo la escala absoluta en $
  queda aproximada. Si aparece el CSV real del broker se usa automáticamente.
  Puerta: sizing de Etapa 4 y cuenta live exigen tipos reales (Etapa 5/6).

## C. Estadística
- **C1:** Parquet vía **polars** (nativo, sin pyarrow instalado). `ledger.parquet`,
  `trials.parquet`, `daily_returns.parquet` usan el mismo formato.
- **C2:** `MinTRL` con penalización simple por skew/kurtosis; es aproximación,
  no el test exacto de Bailey-Lopez de Prado. Suficiente para el veredicto
  READY/INCONCLUSIVE; el DSR formal (Etapa 3) usa el cálculo completo.
- **C3:** Ventanas: IS 2021-01-03→2023-12-31 (~3 años), FORWARD 2024-01→2025-12.
  Con target_sharpe=0.8 el MinTRL ≈ 3.8 años → el combo nace probablemente
  INCONCLUSIVE por potencia: es el veredicto honesto, no un fallo. La evidencia
  adicional debe venir de WF/CPCV + paper, nunca de bajar umbrales.

- **C4 (2026-10-08):** el firewall detectó que FORWARD 2024-01-01 violaba el
  embargo de 40h tras IS 2023-12-31 → FORWARD movido a 2024-01-03 en el maestro.
  Prueba de que el check de pureza funciona; queda registrado como ejemplo.

## D. Estrategia
- **D1:** `example_bbands_breakout.py` usaba estado global mutable
  (`orders_placed_today`, `current_trading_date`) y parámetros globales. La
  versión pipeline (`strategies/bbands_breakout/signal.py`) es factory pura con
  estado por closure: reentrante y paralelizable. Comportamiento idéntico + dos
  endurecimientos: no opera si ya hay posición abierta (MaxOrdenes=1 estricto)
  y params validados por pydantic.
- **D2:** SL/TP = 0.0 en señal (time-stop diario como salida). La spec lo declara
  explícitamente (`exit.time_stop`); el sizing/SL-TP por MFE/MAE llega en Etapa 4.
- **D3 (mejora futura):** filtro `spread_max` y filtro ATR/noticias están en spec
  pero NO implementados en la señal v1.0.0. Decisión: no añadir filtros antes de
  la primera pasada punta a punta (cada filtro es un grado de libertad).

## E. Infra (notas E1–E5 tras la sección F)
## F. Validación sintética (Etapa 3)
- **F1 (2026-10-08):** simular a H1 e inyectar al conector M1→H1 NO es neutro
  (mismos 21 trades pero PF 0.47 vs 0.57: el re-muestreo + spread medio cambian
  fills). Decisión: **simular siempre a M1** (mismo feed que el real). Wicks
  como fracción del precio (ratio vs |C-O| explotaba en dojis y duplicaba el
  rango), spreads por bootstrap real (no constante). Validación M1 + rango H1.
- **F2 (2026-10-08):** gate del control nulo hecho UNILATERAL hacia arriba: lo
  que invalida un run es GANAR en iid (sesgo), no perder (cost-drag esperado:
  nominal da media −3.3 en nulo). Un gate simétrico |media|≤0.15 pararía TODA
  estrategia con costes reales. Se reporta `cost_drag` como flag.
- **F3 (2026-10-08):** tolerancia de validación con suelo absoluto en ac1_abs
  (0.05): la desviación relativa sobre bases ~0.08 no tiene significado
  económico. E2E 1 mes: validación PASS, ranks 50/50/100 → GREEN.

## E. Infra — notas E1–E5
- **E1:** Sin `arch`, `mlflow`, `dvc`, `psycopg2` instalados. GARCH implementado
  a mano (numpy/numba, Etapa 3); tracking MLflow/DVC diferido a Transversal
  (el ProcessLog + master_hash ya dan reproducibilidad mínima).
- **E2:** Optuna 5.0 instalado. `constraints_func` existe en TPESampler v5
  (verificado en Etapa 1 antes de usarlo).
- **E3:** `deterministic_mode:true, n_jobs:1` por defecto hasta que la Etapa 1
  valide el modo paralelo con JournalStorage.
- **E4 (2026-10-08, Etapa 1):** params nominales (bb 20/2.5) en AUDCAD H1 dan
  Sharpe ≈ −3/−4 y ~20 trades/mes: la estrategia base PIERDE en 2021. Es el
  escenario honesto para calibrar el pipeline (debe rechazar, no rescatar).
  Backtest 1 mes ≈ 2-3s, 3 meses ≈ 5s, determinista bit a bit con misma seed.
  `study.best_value` revienta sin trials factibles → el runner calcula best
  sobre completados con flag `best_feasible` (los infactibles conservan valor,
  necesario para DSR/N_eff).
- **E5 (2026-10-08, Etapa 2):** importar `pyeventbt` registra un constructor
  global en `yaml.SafeLoader` que convierte floats en `Decimal` (precisión de
  precios del core; no tocar). `load_master` normaliza a nativos. Además
  Optuna guarda `suggest_float(step)` con ruido binario (2.4000000000000004):
  el objetivo redondea al step ANTES del backtest y persiste
  `params_rounded`; el cluster solo ve grid válido. E2E 8 trials reales:
  1 cluster, N_eff=2 (corr media 0.76: todos pierden igual, correcto).

## G. Etapa 3B (2026-10-08)- PBO/CPCV verificados en dos direcciones: ruido puro -> PBO aprox 0.5; edge plantado -> PBO<=0.30 PASS.
- Sobre trials reales perdedores: PBO 0.82 + CPCV FAIL (correcto: elegir el mejor entre perdedores no persiste).
- WF usa attr_sharpe_net_full (Sharpe real) para WFE, no el score robusto. Trials de folds van al ledger via optimize.

## H. Etapa 4 (2026-10-08)
- Sin ticks del broker: replay = feed M1 del broker (el motor ya corre sobre
  M1) + cobertura tickvol + grid spread/slippage. Slippage como spread
  aditivo; `delay_bars>1` N/A en senal v1 (time_generated fijo). Al no haber
  SL/TP en senal, el orden intrabarra worst-case no aplica (salida time-stop
  a cierre); se audita si la senal v2 anade SL/TP.
- Regla de concentracion de regimen corregida: |PnL| bruto sobre celdas de
  distinto tamano NO es dependencia. FAIL solo si una celda concentra >50%
  del |PnL| familiar con hermana no positiva (ano que gana vs ano que pierde).
- E2E nominal: puertas FAIL (pf 0.47, regimen) -> lockbox bloqueado, FORWARD
  intacto. Orden puertas->lockbox verificado por guarda que revienta.

## I. Viabilidad estrategia/universo (2026-10-08, hallazgo principal M)
- Con spread medio 28pts + comision, un round-trip diario en AUDCAD cuesta
  ~0.7 USD a lote minimo contra un drift SR=3 (0.00087/dia ~= 0.31 USD/trade).
  Resultado: potencia neta 0 hasta edge bruto SR~10. NO es un bug: son los
  costes congelados hablando. Opciones (decision del usuario): (a) par mas
  barato (EURUSD ~10pts, pero no esta en el disco); (b) holding multi-dia
  (el time-stop diario es el devorador de costes); (c) aceptar que el pipeline
  rechaza este combo (veredicto honesto). Recomendacion quant: (b)+(a).
- `plant_drift_m1` debio componer entre dias (ancla = close plantado previo);
  sin componer solo sobrevive el drift del ultimo dia. Bug cazado por la
  propia calibracion.
- La sonda de potencia debe ser LONG-only (`drift_rider`): tendencia + moneda
  al aire = potencia 0 por construccion. FPR puertas 0/12 en 60d.

## M. Meta-etapa M (2026-10-08)
- Nulas: random/shuffled/noise x {iid, block-permuted}. Reporte
  `calibration_report.json` con FPR por etapa + curva de potencia + veredicto
  VALID/INVALID vs targets. Smoke CI: 60d, 2 nulos, 1 nivel (~15s).

## K. Etapa 6 (2026-10-08)
- Paridad: mismatch senales + slippage p95 + tracking error con gates del
  maestro. Bug: `or 1e9` convertia TE=0.0 en FAIL (falsy); comparacion
  explicita con isfinite. CUSUM verificado con salto de nivel determinista
  (el ruido N(0,1) con seed 0 daba falsa alarma: ARL finito, test fragil).
- Kill manual > todo; heartbeat/stale/spread como halt; ordenes idempotentes
  con estado persistido (superviven reinicios).

## T. Transversal (2026-10-08) — diferidos con motivo
- Dashboard Rich: logging estructurado por ahora (sin `rich` instalado);
  el dispatcher emite `strategy/asset/step/progress` por log. Anadir Rich
  cuando haya runs largos en produccion.
- Tracking MLflow / DVC / Docker digest: el ProcessLog + master_hash +
  manifest_hash ya dan reproducibilidad minima auditable; MLflow/DVC son
  mejora P2, no bloquean seleccion.
- `live.monitor` como servicio CLI + behavioural regression en CI: pendientes
  de la primera promocion real (sin sets promovidos no hay linea base).
- Cluster con K < min_samples: todo ruido (-1), singletons top-decila
  rescatados; eps_auto con k = min(min_samples, n-1). Sin este guarda el
  smoke con 4 trials reventaba.

## J. Etapa 5 (2026-10-08)- HRP con bugs reales cazados por tests: raiz del dendrograma (2n-2),
  clip pre-normalizado que rompia el cap, Kelly en unidades anuales,
  simplex infactible (2x0.3<1) ahora revienta con mensaje, resto Kelly-0
  reparte el hueco. Guarda: max_weight*n >= 1 o configuracion invalida.
- Capacity con impacto sqrt: a 1M y 200 trades/ano el decay supera 25%
  (correcto: esa frecuencia no escala a ese capital).

## N. Multi-par / multi-TF (2026-10-09)
- Dataset refrescado: 26 pares x ~2.1M filas M1 (2021-01/02 -> 2026-10),
  manifiestos PASS los 26, sin duplicados ni NaN. Ojo: 10 pares empiezan el
  2021-02-01 (AUDJPY/AUDNZD/AUDUSD/CADCHF/EURUSD/GBPUSD/NZDUSD/USDCAD/USDCHF/
  USDJPY), un mes despues del inicio IS: su IS efectivo es mas corto.
- Pares de conversion reales en disco (AUDUSD/USDCAD/USDCHF/...): el fallback
  de tasa constante queda solo para huecos. Spreads 6.0 (EURUSD) -> 71.6
  (GBPNZD): el spread global habria falseado todos los rankings; por eso
  `universe.base_spread_points` por simbolo + `resolve_base_spread()`.
- Senal D1: barras diarias sin sesion intradia -> una oportunidad por barra +
  flat en cada barra nueva (holding 1). D1 con bb 2.5 da 0 trades/3m y con
  1.5-1.8 da 5-6: es ancho de banda, no bug (verificado). Los combos D1
  nacen con pocos trades: el filtro `min_trades` y las puertas deciden.
- Embargo prep ahora por TF (barras de 1 dia de time-stop: M30:48, H1:24,
  H4:6, D1:1). 104 combos: empezar con `--pair/--timeframe`, el barrido
  completo es `n_trials x coste x 104`.
- FORWARD movido a 2024-02-11: con D1 el embargo son 40 dias y el hueco de
  3 dias solo cubria M30/H1 (H4/D1 daban FAIL de pureza). Mismo firewall que
  en C4, aplicado por TF.
