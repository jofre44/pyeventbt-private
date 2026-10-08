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

## E. Infra- **E1:** Sin `arch`, `mlflow`, `dvc`, `psycopg2` instalados. GARCH implementado
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
