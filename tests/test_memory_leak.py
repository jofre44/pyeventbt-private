"""Regresión de memoria: la caché de TFs no debe retener instancias de conector.

Bug histórico: ``@lru_cache`` sobre un MÉTODO de instancia incluye ``self`` en la
clave, así que la caché global (maxsize=None) retenía cada ``CSVDataProvider``
—y sus DataFrames M1/TF— para siempre. En un run de Optuna (~200 trials) eso
agotaba la RAM: "memory allocation of N bytes failed" (panic de polars/Rust).
"""
import gc

from pyeventbt.data_provider.connectors.csv_data_connector import CSVDataProvider


def test_tf_cache_no_retencion_de_instancias():
    # Claves de caché constantes: sin importar cuántas instancias "distintas" llamen,
    # la caché solo crece por VALOR de tf, nunca por instancia.
    tf_dur, tf_min = "37H", "37H"  # valores no usados por otras pruebas
    antes_dur = CSVDataProvider._timeframe_to_duration.cache_info().currsize
    antes_min = CSVDataProvider._parse_timeframe_to_minutes.cache_info().currsize

    for _ in range(500):
        inst = object.__new__(CSVDataProvider)  # sin __init__ (no carga CSV)
        assert inst._timeframe_to_duration(tf_dur) == "37h"
        assert inst._parse_timeframe_to_minutes(tf_min) == 37 * 60

    tras_dur = CSVDataProvider._timeframe_to_duration.cache_info().currsize
    tras_min = CSVDataProvider._parse_timeframe_to_minutes.cache_info().currsize
    # 500 instancias -> como máximo UNA entrada nueva (por el nuevo tf), no 500.
    assert tras_dur - antes_dur <= 1, "la caché retiene instancias (self en la clave)"
    assert tras_min - antes_min <= 1, "la caché retiene instancias (self en la clave)"
    gc.collect()


def test_mapa_tf_coherente():
    assert CSVDataProvider._timeframe_to_duration("15min") == "15m"
    assert CSVDataProvider._timeframe_to_duration("4H") == "4h"
    assert CSVDataProvider._timeframe_to_duration("1D") == "1d"
    assert CSVDataProvider._parse_timeframe_to_minutes("15min") == 15
    assert CSVDataProvider._parse_timeframe_to_minutes("4H") == 240
    assert CSVDataProvider._parse_timeframe_to_minutes("1D") == 1440
