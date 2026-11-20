"""Pruebas de la calidad de series por año."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.series_quality import low_quality_years, series_quality_by_year


def _ohlcv():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    n = len(idx)
    c = pd.Series(np.linspace(10, 20, n), index=idx)
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1000.0})
    m = idx.year == 2020
    df.loc[m, "Close"] = 10.0                     # 2020: todos los cierres repetidos
    df.loc[m, ["Open", "High", "Low"]] = [10.0, 10.1, 9.9]
    df.loc[idx[-5:], "Volume"] = 0.0              # 5 días sin volumen en 2021
    df.loc[idx[-1], "High"] = df.loc[idx[-1], "Low"] - 1  # OHLC inconsistente
    return df


def test_porcentajes_por_año():
    q = series_quality_by_year(_ohlcv()).set_index("year")
    assert q.loc[2020, "pct_repeated_close"] > 99
    assert q.loc[2021, "pct_repeated_close"] < 1
    assert 0 < q.loc[2021, "pct_zero_volume"] < 3
    assert q.loc[2021, "n_bad_ohlc"] == 1 and q.loc[2020, "n_bad_ohlc"] == 0


def test_lista_de_baja_calidad_con_umbrales():
    q = series_quality_by_year(_ohlcv())
    malos = low_quality_years({"X": q}, max_repeated=20.0, max_zero_vol=10.0)
    assert [(m["ticker"], m["year"]) for m in malos] == [("X", 2020)]
