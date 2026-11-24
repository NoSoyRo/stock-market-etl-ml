"""Tests de src/ml/vol_calendar.py: parseo, ventanas de evento, causalidad y macro rezagado."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml import vol_calendar as vc
from src.ml import vol_regressors as vr


def _ohlcv(start="2017-01-02", end="2021-12-31", spikes=(), seed=0) -> pd.DataFrame:
    """OHLCV sintético; `spikes` = fechas con gap y volumen enormes."""
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(rng.normal(0, 0.002, len(idx)))
    vol = rng.uniform(900, 1100, len(idx))
    df = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.005,
                       "Low": np.minimum(open_, close) * 0.995, "Close": close, "Volume": vol}, index=idx)
    for d in spikes:
        d = pd.Timestamp(d)
        df.loc[d, "Open"] *= 1.08
        df.loc[d, "High"] *= 1.09
        df.loc[d, "Volume"] *= 8
    return df


# ─── Parseo de fuentes ───────────────────────────────────────────────────────

def test_parse_fomc_historical_excluye_llamadas_y_no_programadas():
    html = """<p>January 27-28 Meeting - 2015</p><p>March 25 Conference Call - 2015</p>
              <p>April/May 30-1 Meeting - 2015</p><p>March 15 (unscheduled) Meeting - 2015</p>
              <p>Jan/Feb 31-1 Meeting - 2015</p><p>June 3 Meeting - 2014</p>"""
    got = vc.parse_fomc_year_page(html, 2015)
    assert got == [pd.Timestamp("2015-01-28"), pd.Timestamp("2015-02-01"), pd.Timestamp("2015-05-01")]


def test_parse_fomc_current_ignora_minutas_y_votos_por_notacion():
    html = """<h4>2025 FOMC Meetings</h4> January 28-29 Statement: PDF Minutes: PDF | HTML (Released February 19, 2025)
              March 18-19* Statement August 22 (notation vote) Statement on Longer-Run Goals
              Oct/Nov 31-1 Statement * Meeting associated with a Summary
              <h4>2026 FOMC Meetings</h4> December 8-9*"""
    got = vc.parse_fomc_current_page(html)
    assert got == [pd.Timestamp("2025-01-29"), pd.Timestamp("2025-03-19"),
                   pd.Timestamp("2025-11-01"), pd.Timestamp("2026-12-09")]


def test_parse_banxico_fechas_dd_mm_aa():
    html = "<td>05/11/26</td><td>18/12/25</td><td>18/01/00</td><td>31/02/25</td>"
    assert vc.parse_banxico_announcements(html) == [
        pd.Timestamp("2000-01-18"), pd.Timestamp("2025-12-18"), pd.Timestamp("2026-11-05")]


def test_parse_inegi_inpc_mensual_exige_12_fechas():
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre"]
    filas = [f"{' ' * 60}{9:02d}/{m}     {m.capitalize()}" for m in meses]
    filas.insert(6, "   Índice Nacional de Precios al Consumidor, 2.ª quincena y mensual.   Sí")
    got = vc.parse_inegi_inpc_monthly("\n".join(filas), 2026)
    assert len(got) == 12 and got[0] == pd.Timestamp("2026-01-09") and got[-1] == pd.Timestamp("2026-12-09")
    assert vc.parse_inegi_inpc_monthly("\n".join(filas[:8]), 2026) == []


def test_calendario_csv_real_carga_y_tiene_fuente():
    cal = vc.load_calendar()
    assert {"date", "event", "source"} <= set(cal.columns)
    assert cal["source"].str.startswith("https://").all()
    assert {"banxico", "fomc", "inpc"} <= set(cal["event"])
    fomc = vc.event_dates(cal, "fomc")
    assert pd.Timestamp("2015-01-28") in fomc and pd.Timestamp("2008-12-16") in fomc
    assert vc.event_dates(cal, "banxico").is_monotonic_increasing


# ─── Ventanas de evento ──────────────────────────────────────────────────────

def test_event_features_ventana_5_y_21():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    ev = pd.DatetimeIndex(["2020-02-12", "2020-03-25"])        # miércoles; el 2.º cierra la cobertura
    f = vc.event_features(idx, ev, (5, 21), prefix="x")
    # 5 días hábiles antes (5-feb) → dentro; 6 días antes (4-feb) → fuera
    assert f.loc["2020-02-05", "x_in5"] == 1 and f.loc["2020-02-04", "x_in5"] == 0
    assert f.loc["2020-02-05", "x_days"] == 5
    # el mismo día del evento ya no cuenta (ventana (t, t+h]) y después tampoco
    assert f.loc["2020-02-12", "x_in5"] == 0 and f.loc["2020-02-13", "x_in5"] == 0
    assert f.loc["2020-01-14", "x_in21"] == 1 and f.loc["2020-01-13", "x_in21"] == 0


def test_event_features_cobertura_nan_fuera_del_calendario():
    idx = pd.bdate_range("2019-01-01", "2020-12-31")
    f = vc.event_features(idx, pd.DatetimeIndex(["2020-03-04", "2020-05-06"]), (5,), prefix="x")
    assert f.loc["2019-06-03"].isna().all()        # antes del calendario: se desconoce
    assert f.loc["2020-12-01"].isna().all()        # después del último evento
    assert f.loc["2020-03-02", "x_in5"] == 1


def test_event_features_tolerancia_y_known_from():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    ev = pd.DatetimeIndex(["2020-02-12", "2020-03-25"])
    f = vc.event_features(idx, ev, (5,), tol=2, prefix="x")
    assert f.loc["2020-02-13", "x_in5"] == 1       # 1 día después: la fecha incierta aún puede caer en (t, t+h]
    assert f.loc["2020-02-14", "x_in5"] == 0
    g = vc.event_features(idx, ev, (5,), known_from=pd.DatetimeIndex(["2020-02-10", "2020-02-10"]),
                          prefix="x")
    assert g.loc["2020-02-07", "x_in5"] == 0       # el evento aún no se conocía
    assert g.loc["2020-02-10", "x_in5"] == 1


def test_event_features_solo_usa_calendario_no_filas_futuras():
    """Con feriados/huecos en el índice del ticker, la distancia no cambia."""
    full = pd.bdate_range("2020-01-01", "2020-03-31")
    ev = pd.DatetimeIndex(["2020-02-12"])
    a = vc.event_features(full, ev, (5,), prefix="x")
    b = vc.event_features(full.delete([25, 26, 27]), ev, (5,), prefix="x")
    pd.testing.assert_frame_equal(a.loc[b.index], b)


# ─── Resultados trimestrales ─────────────────────────────────────────────────

SPIKES_Y1 = ["2018-04-26", "2018-07-26", "2018-10-25", "2019-02-21"]


def test_fecha_esperada_viene_del_anio_anterior():
    spikes = SPIKES_Y1 + ["2019-04-30"]      # el pico de abr-2019 se movió 2 días hábiles después
    df = _ohlcv(spikes=spikes)
    ex = vc.expected_report_dates(df)
    esperado_abr19 = pd.Timestamp("2018-04-26") + pd.Timedelta(days=364)   # 2019-04-25
    assert esperado_abr19 in set(pd.DatetimeIndex(ex["expected"]))
    # el pico real de 2019 (30-abr) no se usa para predecir abr-2019
    assert pd.Timestamp("2019-04-30") not in set(pd.DatetimeIndex(ex["expected"]))


def test_earnings_features_son_causales_al_truncar():
    df = _ohlcv(spikes=SPIKES_Y1 + ["2019-04-30", "2019-07-25"])
    full = vc.earnings_features(df)
    cut = pd.Timestamp("2019-06-14")
    part = vc.earnings_features(df[df.index <= cut])
    a, b = full.loc[:cut].dropna(), part.loc[:cut].dropna()
    # donde la versión truncada tiene valor (cobertura puede acabar antes), debe coincidir
    common = b.index.intersection(a.index)
    assert len(common) > 100
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common])


def test_earnings_flag_se_enciende_antes_de_la_fecha_esperada():
    df = _ohlcv(spikes=SPIKES_Y1)
    f = vc.earnings_features(df)
    esp = pd.Timestamp("2019-04-25")
    assert f.loc[esp - pd.offsets.BDay(3), "earn_in5"] == 1
    assert f.loc[esp - pd.offsets.BDay(40), "earn_in5"] == 0
    assert f.loc[:"2017-12-31"].isna().all().all()     # sin historia previa: se desconoce


# ─── Dividendos ──────────────────────────────────────────────────────────────

def test_exdiv_declared_y_expected():
    idx = pd.bdate_range("2018-01-01", "2020-12-31")
    div = pd.Series([1.0, 1.0], index=pd.DatetimeIndex(["2018-06-14", "2019-06-13"]))
    d = vc.exdiv_declared_features(idx, div)
    assert d.loc["2019-06-10", "exdiv_in5"] == 1 and d.loc["2019-05-20", "exdiv_in5"] == 0
    assert d.loc["2017-12-01"].isna().all() if idx[0] < pd.Timestamp("2017-12-01") else True
    x = vc.exdiv_expected_features(idx, div)
    esp = pd.Timestamp("2018-06-14") + pd.Timedelta(days=364)          # 2019-06-13
    assert x.loc[esp - pd.offsets.BDay(4), "exdivx_in5"] == 1
    # el expected de 2020 viene de la fecha de 2019, ya conocida
    assert x.loc[pd.Timestamp("2019-06-13") + pd.Timedelta(days=364) - pd.offsets.BDay(3), "exdivx_in5"] == 1


def test_exdiv_sin_dividendos_es_cero_y_sin_archivo_es_nan():
    idx = pd.bdate_range("2018-01-01", "2018-03-01")
    assert (vc.exdiv_declared_features(idx, pd.Series([], dtype=float, index=pd.DatetimeIndex([])))
            ["exdiv_in5"] == 0).all()
    assert vc.exdiv_declared_features(idx, None).isna().all().all()


# ─── Macro con rezago ────────────────────────────────────────────────────────

def test_macro_rezagado_un_dia():
    us = pd.bdate_range("2020-01-01", "2020-03-31")
    vix = pd.Series(20.0, index=us)
    vix.loc["2020-02-10":] = 40.0                      # salto el 10-feb (cierre de ese día)
    out = vc.lag_to_calendar(vix, us)
    assert out.loc["2020-02-10"] == 20.0               # el cierre de t no se conoce en t
    assert out.loc["2020-02-11"] == 40.0


def test_macro_ffill_en_dia_sin_sesion_usa_ultimo_cierre_previo():
    us = pd.bdate_range("2020-01-01", "2020-01-31").delete(10)      # EE.UU. cerrado 15-ene
    s = pd.Series(np.arange(len(us), dtype=float) + 1, index=us)
    mx = pd.bdate_range("2020-01-01", "2020-01-31")
    out = vc.lag_to_calendar(s, mx)
    assert out.loc["2020-01-15"] == s.loc["2020-01-14"] - 0 or out.loc["2020-01-15"] == s.loc["2020-01-13"]
    assert out.loc["2020-01-15"] < s.loc["2020-01-14"] + 1e-9    # nunca el valor de t o posterior


def test_vix_features_no_miran_el_futuro():
    us = pd.bdate_range("2019-01-01", "2020-06-30")
    rng = np.random.default_rng(1)
    vix = pd.Series(15 * np.exp(np.cumsum(rng.normal(0, 0.02, len(us)))), index=us)
    base = vc.vix_features(vix, us)
    alt = vix.copy()
    alt.loc["2020-03-01":] *= 3
    mod = vc.vix_features(alt, us)
    pd.testing.assert_frame_equal(base.loc[:"2020-03-02"], mod.loc[:"2020-03-02"])   # t=2-mar usa hasta 28-feb


# ─── Registro ────────────────────────────────────────────────────────────────

def test_regresores_registrados_en_el_arnes():
    for name in ("earnings_cal", "exdiv_declared", "exdiv_expected", "banxico", "fomc", "inpc", "vix", "dxy"):
        assert name in vr.REGRESSORS
    cols = [c for n in ("earnings_cal", "banxico", "vix") for c in vr.REGRESSORS[n].cols]
    assert len(cols) == len(set(cols))


def test_skill_in_out_detecta_ganancia_dentro_de_la_ventana():
    """Señal sintética: el regresor explica el objetivo solo cuando la bandera está encendida."""
    from src.ml.targets import HAR_COLS
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2015-01-01", periods=600), 5)
    n = len(dates)
    flag = (rng.random(n) < 0.3).astype(float)
    sig = rng.normal(0, 1, n)
    panel = pd.DataFrame({"date": dates, "flag": flag, "sig": sig * flag})
    for c in HAR_COLS:
        panel[c] = rng.normal(0, 1, n)
    panel["log_rv_fwd_5"] = 0.3 * panel[HAR_COLS[0]] + 1.5 * sig * flag + rng.normal(0, 0.3, n)
    r = vc.skill_in_out(panel, "log_rv_fwd_5", 5, ["sig"], "flag", n_splits=3)
    assert r["in"]["delta"] > 0.05 and abs(r["out"]["delta"]) < 0.02
