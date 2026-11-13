"""Pruebas de la limpieza de EPS reportados."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.eps_clean import clean_eps, flag_counts


def _raw():
    return {
        "A.MX": [
            {"announced": "2024-02-09", "reported": 1.0, "surprise": 0.10},   # viernes
            {"announced": "2024-02-09", "reported": 1.0, "surprise": 0.10},   # duplicado
            {"announced": "2024-05-11", "reported": 2.0, "surprise": 0.0},    # sábado, sorpresa 0
            {"announced": "2024-08-08", "reported": 3.0, "surprise": 7.5},    # extremo
            {"announced": "2024-11-07", "reported": 4.0, "surprise": None},
        ],
        "VACIO.MX": [],
    }


def test_columnas_y_orden():
    df = clean_eps(_raw())
    assert list(df.columns) == ["ticker", "announced", "reported", "surprise", "available_from", "flags"]
    assert df["announced"].is_monotonic_increasing
    assert set(df["ticker"]) == {"A.MX"}


def test_duplicados_se_quitan():
    df = clean_eps(_raw())
    assert len(df) == 4


def test_fin_de_semana_se_mueve_y_se_marca():
    df = clean_eps(_raw()).set_index("announced")
    fila = df.loc["2024-05-13"]  # lunes
    assert "weekend_moved" in fila["flags"]


def test_sorpresa_cero_es_nan():
    df = clean_eps(_raw()).set_index("announced")
    fila = df.loc["2024-05-13"]
    assert np.isnan(fila["surprise"]) and "surprise_zero_nan" in fila["flags"]


def test_extremo_se_marca_no_se_borra():
    df = clean_eps(_raw()).set_index("announced")
    assert df.loc["2024-08-08", "surprise"] == 7.5
    assert "extreme" in df.loc["2024-08-08", "flags"]


def test_available_from_es_siguiente_habil_tras_announced():
    df = clean_eps(_raw()).set_index("announced")
    assert df.loc["2024-02-09", "available_from"] == pd.Timestamp("2024-02-12")  # viernes -> lunes
    assert df.loc["2024-05-13", "available_from"] == pd.Timestamp("2024-05-14")
    assert (df["available_from"] > df.index).all()


def test_acepta_dataframe_y_cuenta_reglas():
    raw = _raw()
    filas = [dict(r, ticker=t) for t, rs in raw.items() for r in rs]
    df = clean_eps(pd.DataFrame(filas))
    c = flag_counts(df, n_input=len(filas))
    assert c["duplicates_removed"] == 1
    assert c["weekend_moved"] == 1 and c["surprise_zero_nan"] == 1 and c["extreme"] == 1
