"""Fundamentales: el día de publicación no se puede usar ese mismo día (look-ahead de 1 día)."""
import numpy as np
import pandas as pd

from src.ml.features import _FUND_QUARTERLY_COLS, add_fundamental_features


def test_fundamental_disponible_desde_la_sesion_siguiente():
    col = _FUND_QUARTERLY_COLS[0]
    idx = pd.bdate_range("2024-05-01", periods=6)  # mié 1 may ... mié 8 may
    df = pd.DataFrame({"Close": np.arange(6, dtype=float)}, index=idx)
    pub = idx[3]  # el balance sale el día idx[3]
    fund = pd.DataFrame({"available_date": [pub], col: [42.0]})

    out = add_fundamental_features(df, fund, ticker=None, symbol=None)

    assert out.loc[idx[2], col] != out.loc[idx[2], col]  # NaN antes de publicar
    assert np.isnan(out.loc[pub, col])                   # el mismo día NO se usa
    assert out.loc[idx[4], col] == 42.0                  # desde la sesión siguiente sí
    assert out.loc[idx[5], col] == 42.0


def test_publicacion_en_fin_de_semana_se_usa_el_lunes():
    col = _FUND_QUARTERLY_COLS[0]
    idx = pd.bdate_range("2024-05-06", periods=5)  # lun 6 ... vie 10 may
    df = pd.DataFrame({"Close": np.arange(5, dtype=float)}, index=idx)
    fund = pd.DataFrame({"available_date": [pd.Timestamp("2024-05-04")], col: [7.0]})  # sábado

    out = add_fundamental_features(df, fund, ticker=None, symbol=None)

    assert out.loc[idx[0], col] == 7.0
