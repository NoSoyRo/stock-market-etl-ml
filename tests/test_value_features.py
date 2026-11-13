"""Pruebas de ttm_eps_on_prices: sin look-ahead y robusta a la resolución de fechas (ns/us)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.value_features import ttm_eps_on_prices


def _referencia(close: pd.Series, events: pd.DataFrame) -> pd.Series:
    """Versión lenta: suma de los últimos 4 EPS anunciados con fecha <= t-1 día."""
    ev = events.sort_values("announced")
    out = []
    for t in close.index:
        vis = ev[ev["announced"] + pd.Timedelta(days=1) <= t]
        out.append(vis["reported"].tail(4).sum() if len(vis) >= 4 else np.nan)
    return pd.Series(out, index=close.index) / close


def _datos(unidad: str):
    idx = pd.date_range("2020-01-01", periods=400, freq="B").astype(f"datetime64[{unidad}]")
    close = pd.Series(np.linspace(10, 20, 400), index=idx)
    ann = pd.to_datetime(["2020-02-10", "2020-05-11", "2020-08-10", "2020-11-09", "2021-02-08"])
    ev = pd.DataFrame({"announced": ann.astype(f"datetime64[{unidad}]"), "reported": [1, 2, 3, 4, 5.0]})
    return close, ev


def test_coincide_con_referencia_en_ns_y_us():
    for unidad in ("ns", "us"):
        close, ev = _datos(unidad)
        pd.testing.assert_series_equal(
            ttm_eps_on_prices(close, ev), _referencia(close, ev), check_names=False, check_freq=False
        )


def test_mezcla_de_resoluciones_no_falla():
    close, _ = _datos("ns")
    _, ev = _datos("us")
    out = ttm_eps_on_prices(close, ev)
    assert out.notna().sum() > 0


def test_sin_lookahead_el_dia_del_anuncio():
    close, ev = _datos("ns")
    out = ttm_eps_on_prices(close, ev)
    # 4º anuncio: 2020-11-09 -> disponible el 2020-11-10, no antes
    assert np.isnan(out.loc["2020-11-09"])
    assert out.loc["2020-11-10"] == (1 + 2 + 3 + 4) / close.loc["2020-11-10"]
