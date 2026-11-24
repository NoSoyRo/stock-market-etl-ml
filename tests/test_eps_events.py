"""Tests de eps_events: ventana de evento causal y fecha esperada (+364 días)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.eps_events import (
    earnings_eps_features,
    expected_accuracy,
    expected_events,
    expected_flag,
    expected_next_date,
    has_recent_event,
)


def _hist(*announced: str) -> pd.DataFrame:
    a = pd.to_datetime(list(announced))
    # available_from = siguiente día lunes-viernes
    av = [x + pd.offsets.BDay(1) for x in a]
    return pd.DataFrame({"announced": a, "available_from": av})


def test_evento_en_t_no_cuenta_hasta_available_from():
    dates = pd.bdate_range("2024-03-04", periods=15)
    h = _hist("2024-03-08")  # viernes → available_from lunes 11
    r = has_recent_event(h, dates, window=5)
    assert not r[pd.Timestamp("2024-03-08")]      # el día del anuncio aún no se sabe
    assert r[pd.Timestamp("2024-03-11")]          # primer día conocido


def test_ventana_de_5_dias():
    dates = pd.bdate_range("2024-03-04", periods=15)
    h = _hist("2024-03-08")                       # conocido desde el 11
    r = has_recent_event(h, dates, window=5)
    assert r[pd.Timestamp("2024-03-15")]          # 4 días después del 11: dentro
    assert not r[pd.Timestamp("2024-03-18")]      # 5 días después: fuera


def test_sin_eventos_todo_falso():
    dates = pd.bdate_range("2024-01-01", periods=5)
    r = has_recent_event(_hist()[["announced", "available_from"]], dates)
    assert not r.any() and len(r) == 5


def test_expected_events_mas_364_y_known_from():
    h = _hist("2023-04-20")  # jueves
    ex = expected_events(h)
    assert ex.loc[0, "expected"] == pd.Timestamp("2024-04-18")   # +364 días: jueves
    assert ex.loc[0, "known_from"] == pd.Timestamp("2023-04-21")


def test_expected_next_date_causal():
    h = _hist("2023-04-20", "2023-07-20")
    # el 2023-05-01 solo se conoce el primer anuncio
    assert expected_next_date(h, pd.Timestamp("2023-05-01")) == pd.Timestamp("2024-04-18")
    # antes de cualquier anuncio conocido no hay fecha
    assert pd.isna(expected_next_date(h, pd.Timestamp("2023-04-20")))
    # con ambos conocidos, la más próxima en el futuro
    assert expected_next_date(h, pd.Timestamp("2023-08-01")) == pd.Timestamp("2024-04-18")


def test_expected_next_date_sin_look_ahead():
    """Un anuncio posterior a `today` no puede influir."""
    h = _hist("2023-04-20", "2024-04-18")
    assert expected_next_date(h, pd.Timestamp("2024-01-10")) == pd.Timestamp("2024-04-18")
    h2 = _hist("2023-04-20", "2024-04-25")
    assert expected_next_date(h2, pd.Timestamp("2024-01-10")) == pd.Timestamp("2024-04-18")


def test_expected_flag_ventana_y_cobertura():
    idx = pd.bdate_range("2024-04-01", "2024-04-30")
    h = _hist("2023-04-20", "2023-07-20")            # esperadas 2024-04-18 y 2024-07-18
    f = expected_flag(idx, h, before=2, after=8)
    assert f[pd.Timestamp("2024-04-10")] == 1.0      # 6 días hábiles antes: dentro (t+6..)
    assert f[pd.Timestamp("2024-04-19")] == 1.0      # pasada 1 día: dentro (tol)
    assert f[pd.Timestamp("2024-04-22")] == 0.0      # 2 días hábiles después: fuera
    assert f[pd.Timestamp("2024-04-01")] == 0.0      # 13 días antes: fuera
    # antes de conocer el anuncio previo: sin información
    idx2 = pd.bdate_range("2023-01-02", "2023-02-01")
    assert expected_flag(idx2, h).isna().all()


def test_expected_accuracy():
    h = _hist("2022-04-21", "2023-04-20", "2024-04-19")   # esperadas: 2023-04-20, 2024-04-18
    err = expected_accuracy(h)
    assert len(err) == 2 and list(err) == [0, 1]          # error en días hábiles (real − esperado)


def test_earnings_eps_features_columnas():
    idx = pd.bdate_range("2024-03-01", "2024-05-31")
    out = earnings_eps_features(idx, _hist("2023-04-20"))
    assert list(out.columns) == ["earn_in5", "earn_in21", "earn_days"]
    assert out.loc["2024-04-15", "earn_in5"] == 1.0
    assert np.isnan(earnings_eps_features(idx, _hist()[["announced", "available_from"]]).to_numpy()).all()


def test_sharpe_diff_ci():
    from src.ml.eps_events import sharpe_diff_ci
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0.002, 0.02, 400))
    a = base * 0.5 + 0.002           # misma media, mitad de volatilidad aprox.
    r = sharpe_diff_ci(a, base, periods_per_year=50.4)
    assert r["delta"] > 0 and r["ci_lo"] < r["delta"] < r["ci_hi"]
    # reproducible y con datos cortos no truena
    assert sharpe_diff_ci(a, base, periods_per_year=50.4) == r
    assert np.isnan(sharpe_diff_ci(a[:5], base[:5], periods_per_year=50.4)["ci_lo"])
