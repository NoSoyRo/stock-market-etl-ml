"""Pruebas offline de dividendos y retorno total (M3b); la red se simula."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.total_return import (
    add_dividends_to_forward_return,
    ex_div_flag,
    fetch_dividends,
    normalize_dividends,
    total_return_series,
)

IDX = pd.bdate_range("2024-01-01", periods=20)


def _divs(*pairs):
    return pd.Series([v for _, v in pairs], index=pd.DatetimeIndex([d for d, _ in pairs]), name="dividend")


def test_normalize_quita_zona_y_hora():
    raw = pd.Series([1.0, 0.0], index=pd.DatetimeIndex(["2024-02-06 07:30", "2024-03-01 07:30"], tz="America/Mexico_City"))
    s = normalize_dividends(raw)
    assert list(s.index) == [pd.Timestamp("2024-02-06")] and s.index.tz is None


def test_total_return_en_la_fecha_ex():
    close = pd.Series(100.0, index=IDX)
    close.iloc[5:] = 99.0  # el precio cae 1 el día ex
    tr = total_return_series(pd.DataFrame({"Close": close}), _divs((IDX[5], 1.0)))
    assert tr.iloc[5] == pytest.approx(0.0)  # (99+1)/100 − 1
    assert tr.iloc[6] == pytest.approx(0.0)


def test_sin_dividendos_igual_precio():
    close = pd.Series(np.linspace(100, 110, 20), index=IDX)
    tr = total_return_series(pd.DataFrame({"Close": close}), normalize_dividends(None))
    assert np.allclose(tr.dropna(), close.pct_change().dropna())


def test_flag_dentro_y_fuera_de_ventana():
    f = ex_div_flag(IDX, _divs((IDX[10], 1.0)), window=5)
    assert f.iloc[5:10].all()             # t=5..9: ex en (t, t+5]
    assert not f.iloc[10]                 # el día ex no cuenta (ventana abierta en t)
    assert not f.iloc[:5].any() and not f.iloc[11:].any()


def test_flag_sin_dividendos():
    assert not ex_div_flag(IDX, normalize_dividends(None), 5).any()


def test_ex_fecha_fin_de_semana_va_al_siguiente_habil():
    sab = pd.Timestamp("2024-01-13")  # sábado -> lunes 15 (posición 10)
    f = ex_div_flag(IDX, _divs((sab, 1.0)), window=1)
    assert f.iloc[9] and f.sum() == 1


def test_varios_en_la_misma_ventana_se_suman():
    close = pd.Series(100.0, index=IDX)
    fwd = pd.Series(0.0, index=IDX)
    d = _divs((IDX[8], 1.0), (IDX[10], 2.0))
    out = add_dividends_to_forward_return(fwd, d, 5, close=close)
    assert out.iloc[5] == pytest.approx(3.0)  # (1+2)/100·100
    assert out.iloc[9] == pytest.approx(2.0)  # solo el de IDX[10]
    assert out.iloc[1] == pytest.approx(0.0)


def test_fetch_cache_y_reintentos(tmp_path):
    llamadas = []

    def falla_una_vez(sym):
        llamadas.append(sym)
        if len(llamadas) == 1:
            raise RuntimeError("429")
        return pd.Series([0.5], index=pd.DatetimeIndex(["2024-03-01"], tz="UTC"))

    s = fetch_dividends("X.MX", cache_dir=tmp_path, fetcher=falla_una_vez, pause=0)
    assert len(llamadas) == 2 and s.iloc[0] == 0.5
    fetch_dividends("X.MX", cache_dir=tmp_path, fetcher=lambda _s: 1 / 0)  # sale del caché
    with pytest.raises(RuntimeError):
        fetch_dividends("Y.MX", cache_dir=tmp_path, fetcher=lambda _s: (_ for _ in ()).throw(RuntimeError("x")), retries=2, pause=0)
