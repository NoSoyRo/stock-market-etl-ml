"""Tests de exdiv_capture (N3): retornos de evento ex-dividendo y predicción causal de la fecha."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.exdiv_capture import (
    causal_trades,
    event_returns,
    net_return,
    predict_exdate,
    quarter_mean_t,
    top_n_share,
)


def _flat_close(n: int = 40, level: float = 100.0) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(level, index=idx, name="Close")


def test_captura_parcial_es_40_por_ciento_del_dividendo():
    close = _flat_close()
    p = 20
    div_amt = 5.0
    close.iloc[p:] = 100.0 - 0.6 * div_amt  # cae el 60% del dividendo
    divs = pd.Series([div_amt], index=[close.index[p]], name="dividend")
    ev = event_returns(close, divs, window=5)
    assert len(ev) == 1
    # (97 + 5) / 100 - 1 = 2% bruto = 40% del dividendo (5%) = 2%
    assert ev["gross"].iloc[0] == pytest.approx(0.02)
    assert ev["yield"].iloc[0] == pytest.approx(0.05)
    # neto de 0.30%/lado sin retención: 2% - 0.6%
    assert net_return(ev["gross"], ev["yield"], tax=0.0, cost_per_side=0.003).iloc[0] == pytest.approx(0.014)
    # con retención de 10% se pierde 10% del dividendo (0.5%)
    assert net_return(ev["gross"], ev["yield"], tax=0.10, cost_per_side=0.003).iloc[0] == pytest.approx(0.009)


def test_fecha_ex_en_fin_de_semana_va_al_siguiente_dia_habil():
    close = _flat_close()
    sat = close.index[20] + pd.Timedelta(days=(5 - close.index[20].weekday()) % 7)  # sábado de esa semana
    divs = pd.Series([1.0], index=[sat])
    ev = event_returns(close, divs, window=5)
    assert ev.index[0] > sat and ev.index[0].weekday() == 0


def test_descarta_sin_historia_previa_y_yield_extraordinario():
    close = _flat_close()
    divs = pd.Series([1.0, 20.0], index=[close.index[2], close.index[25]])  # 1º sin 5 días previos; 2º yield 20%
    ev = event_returns(close, divs, window=5, max_yield=0.15)
    assert ev.empty
    ev_all = event_returns(close, divs, window=5, max_yield=None)
    assert len(ev_all) == 1  # el de yield 20% se conserva si no hay tope


def test_predict_exdate_causal_sin_lookahead():
    hist = pd.DatetimeIndex(["2019-05-14", "2020-05-12", "2021-05-11"])
    today = pd.Timestamp("2021-05-04")
    # solo ve eventos <= today: 2020-05-12 + 364 d = 2021-05-11
    pred = predict_exdate(hist, today)
    assert pred == pd.Timestamp("2021-05-11")
    # añadir eventos futuros a la historia no cambia la predicción
    hist2 = hist.append(pd.DatetimeIndex(["2022-05-10", "2023-05-09"]))
    assert predict_exdate(hist2, today) == pred


def test_predict_exdate_sin_historia_da_nat():
    assert pd.isna(predict_exdate(pd.DatetimeIndex([]), pd.Timestamp("2021-01-01")))
    # evento hace más de un año y nada más reciente: no hay candidato > today
    assert pd.isna(predict_exdate(pd.DatetimeIndex(["2015-01-05"]), pd.Timestamp("2021-01-01")))


def test_causal_trades_acierto_y_fallo():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    close = pd.Series(100.0, index=idx)
    h = pd.Timestamp("2019-06-11")  # martes
    fail_next = pd.Timestamp("2020-06-16")  # el real se mueve 3 días hábiles tarde
    divs = pd.Series([2.0, 2.0], index=[h, fail_next])
    tr = causal_trades(close, divs, window=5)
    # la operación prevista para 2020-06-09 (h+364) compra 5 días antes y NO cobra el dividendo real (cae después)
    t1 = tr[tr["pred_ex"] == pd.Timestamp("2020-06-09")].iloc[0]
    assert t1["gross"] == pytest.approx(0.0)
    assert t1["n_div_captured"] == 0
    assert t1["yield"] == pytest.approx(0.02)  # yield causal = dividendo de h / precio de compra
    # la prevista para 2021-06-15 (fail_next + 364) tampoco tiene dividendo real
    assert (tr["pred_ex"] == pd.Timestamp("2021-06-15")).any()
    # si el dividendo real cae dentro de la ventana, sí se cobra
    divs2 = pd.Series([2.0, 2.0], index=[h, pd.Timestamp("2020-06-08")])
    t2 = causal_trades(close, divs2, window=5)
    t2 = t2[t2["pred_ex"] == pd.Timestamp("2020-06-09")].iloc[0]
    assert t2["n_div_captured"] == 1
    assert t2["gross"] == pytest.approx(0.02)


def test_causal_trades_solo_usa_historia_hasta_la_compra():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    close = pd.Series(100.0, index=idx)
    base = pd.Series([2.0], index=[pd.Timestamp("2019-06-11")])
    extra = pd.Series([2.0, 9.0], index=[pd.Timestamp("2019-06-11"), pd.Timestamp("2020-06-09")])
    a = causal_trades(close, base, window=5)
    b = causal_trades(close, extra, window=5)
    # el yield causal de la operación 2020-06-09 no depende del dividendo futuro (9.0)
    ya = a[a["pred_ex"] == pd.Timestamp("2020-06-09")]["yield"].iloc[0]
    yb = b[b["pred_ex"] == pd.Timestamp("2020-06-09")]["yield"].iloc[0]
    assert ya == pytest.approx(yb)


def test_quarter_mean_t_y_top_n_share():
    idx = pd.to_datetime(["2020-01-10", "2020-02-10", "2020-04-10", "2020-05-10", "2020-07-10", "2020-08-10"])
    r = pd.Series([0.01, 0.03, 0.02, 0.02, 0.02, 0.04], index=idx)  # medias por trimestre: .02, .02, .03
    s = quarter_mean_t(r)
    assert s["n_q"] == 3
    assert s["n"] == 6
    assert s["mean_q"] == pytest.approx(0.07 / 3)
    assert np.isfinite(s["t"])
    vals = pd.Series([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = top_n_share(vals, 5)
    assert out["share"] == pytest.approx(14.0 / 16.0)
    assert out["mean_ex_top"] == pytest.approx(1.0)
