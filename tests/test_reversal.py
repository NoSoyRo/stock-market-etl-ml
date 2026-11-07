"""Pruebas de las señales de reversión (M4): causalidad, alineación y utilidades."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.reversal import (
    SIGNAL_NAMES,
    abnormal_volume,
    block_means,
    build_signals,
    forward_returns,
    tercile_by_date,
    to_panel,
    weekly_bottom_weights,
)

IDX = pd.bdate_range("2020-01-01", periods=400)


def _prices(seed: int = 0, n_tick: int = 6):
    rng = np.random.default_rng(seed)
    cols = [f"T{i}" for i in range(n_tick)]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(IDX), n_tick)), axis=0)), IDX, cols)
    o = c.shift(1).fillna(c.iloc[0]) * np.exp(rng.normal(0, 0.003, c.shape))
    return o, c


def test_causalidad_cambiar_el_futuro_no_altera_el_pasado():
    o, c = _prices()
    s1 = build_signals(o, c)
    T = 300
    o2, c2 = o.copy(), c.copy()
    o2.iloc[T + 1:] *= 3.0  # perturba todo lo posterior a T
    c2.iloc[T + 1:] *= 0.4
    s2 = build_signals(o2, c2)
    assert set(s1) == set(SIGNAL_NAMES)
    for name in SIGNAL_NAMES:
        a, b = s1[name].iloc[: T + 1], s2[name].iloc[: T + 1]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


def test_alineacion_senales():
    o, c = _prices()
    s = build_signals(o, c)
    t = 200
    assert np.isclose(s["ret_1d"].iloc[t, 0], c.iloc[t, 0] / c.iloc[t - 1, 0] - 1)
    assert np.isclose(s["ret_5d"].iloc[t, 1], c.iloc[t, 1] / c.iloc[t - 5, 1] - 1)
    assert np.isclose(s["overnight_1d"].iloc[t, 2], o.iloc[t, 2] / c.iloc[t - 1, 2] - 1)
    assert np.isclose(s["intraday_1d"].iloc[t, 2], c.iloc[t, 2] / o.iloc[t, 2] - 1)
    # overnight × intradía compuestos reconstruyen el retorno de cierre a cierre
    lhs = (1 + s["overnight_5d"]) * (1 + s["intraday_5d"])
    pd.testing.assert_frame_equal(lhs, 1 + s["ret_5d"], check_exact=False, rtol=1e-10)


def test_beta_con_shift_y_residuo_quita_el_mercado():
    # todos los tickers = mercado × 1 ⇒ el residuo es ~0 donde hay beta
    rng = np.random.default_rng(1)
    m = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(IDX)))) * 100, IDX)
    c = pd.DataFrame({"A": m, "B": m, "C": m})
    o = c.shift(1).bfill()
    r = build_signals(o, c)["resid_5d"].dropna()
    assert len(r) > 100 and r.abs().to_numpy().max() < 1e-9


def test_retornos_futuros_alineados():
    o, c = _prices()
    f = forward_returns(o, c, horizon=5)
    t = 100
    assert np.isclose(f["close_close"].iloc[t, 0], c.iloc[t + 5, 0] / c.iloc[t, 0] - 1)
    assert np.isclose(f["open_close"].iloc[t, 0], c.iloc[t + 6, 0] / o.iloc[t + 1, 0] - 1)
    assert np.isclose(f["open_open"].iloc[t, 0], o.iloc[t + 6, 0] / o.iloc[t + 1, 0] - 1)
    assert np.isclose(f["skip_close"].iloc[t, 0], c.iloc[t + 6, 0] / c.iloc[t + 1, 0] - 1)
    assert f["open_close"].iloc[-6:].isna().all().all()


def test_volumen_anormal_excluye_hoy_y_terciles():
    v = pd.DataFrame({"A": [10.0] * 25 + [40.0], "B": [10.0] * 26}, index=pd.bdate_range("2020-01-01", periods=26))
    av = abnormal_volume(v, window=20)
    assert np.isclose(av["A"].iloc[-1], 4.0) and np.isclose(av["B"].iloc[-1], 1.0)
    x = pd.DataFrame([[1.0, 2.0, 3.0, np.nan]], columns=list("abcd"))
    t = tercile_by_date(x)
    assert list(t.iloc[0, :3]) == [0, 1, 2] and np.isnan(t.iloc[0, 3])


def test_panel_bloques_y_pesos():
    o, c = _prices()
    p = to_panel({"x": c, "y": o})
    assert len(p) == c.size and set(p.columns) == {"ticker", "x", "y"}
    ic = pd.Series(np.arange(100.0), index=pd.bdate_range("2020-01-01", periods=100))
    b = block_means(ic, 5)
    assert len(b) == 5 and b[0]["n"] == 20 and b[0]["mean"] < b[4]["mean"]
    score = -build_signals(o, c)["ret_5d"].dropna()
    w = weekly_bottom_weights(score, k=2)
    assert np.allclose(w.sum(axis=1), 1.0) and ((w > 0).sum(axis=1) == 2).all()
    d = w.index[0]
    assert set(w.loc[d][w.loc[d] > 0].index) == set(score.loc[d].nlargest(2).index)
