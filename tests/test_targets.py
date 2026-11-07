"""Pruebas de las etiquetas alternativas y del CV purgado por fechas."""
import numpy as np
import pandas as pd
import pytest

from src.ml.purged_cv import purged_walk_forward_date_splits
from src.ml.targets import (
    build_targets, extra_features, forward_realized_vol, forward_return,
    forward_return_open, label_span,
    is_regression, past_realized_vol, relative_forward_return, target_horizon,
)
from src.ml.vol_models import HARModel, naive_forecast, qlike, regression_metrics


def _ohlc(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.abs(rng.normal(0, 0.01, n)) + 0.001
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, n)),
        "High": close * (1 + spread),
        "Low": close * (1 - spread),
        "Close": close,
        "Volume": rng.integers(1_000, 10_000, n).astype(float),
    }, index=idx)


# ─── Sin mirar al futuro ─────────────────────────────────────────────────────

@pytest.mark.parametrize("h", [5, 21, 63])
def test_labels_do_not_use_prices_beyond_horizon(h):
    """Cambiar precios después de t+h no altera ninguna etiqueta de t."""
    df, bench = _ohlc(), _ohlc(seed=1)["Close"]
    t = 100
    base = build_targets(df, bench, horizons=(h,), vol_horizons=(5,))
    shocked = df.copy()
    shocked.iloc[t + h + 1:] *= 3.0
    after = build_targets(shocked, bench, horizons=(h,), vol_horizons=(5,))
    for col in (f"dir_{h}", f"rel_{h}"):
        assert base[col].iloc[t] == after[col].iloc[t]


def test_labels_do_not_use_past_prices():
    """Cambiar precios antes de t no altera la etiqueta de rendimiento de t."""
    df = _ohlc()
    t = 200
    shocked = df.copy()
    shocked.iloc[:t] *= 0.5
    assert forward_return(df["Close"], 5).iloc[t] == pytest.approx(
        forward_return(shocked["Close"], 5).iloc[t])


def test_features_are_causal():
    """Cambiar precios futuros no altera ninguna feature de t."""
    df = _ohlc()
    t = 300
    shocked = df.copy()
    shocked.iloc[t + 1:] *= 3.0
    pd.testing.assert_series_equal(extra_features(df).iloc[t],
                                   extra_features(shocked).iloc[t])


def test_forward_vol_uses_exactly_next_h_days():
    df = _ohlc()
    t, h = 150, 5
    base = forward_realized_vol(df, h).iloc[t]
    shocked = df.copy()
    shocked.iloc[t + h + 1:, :4] *= 2.0     # después de la ventana: no cambia
    assert forward_realized_vol(shocked, h).iloc[t] == pytest.approx(base)
    shocked = df.copy()
    shocked.iloc[t + 2, 1] *= 1.05           # High dentro de la ventana: sí cambia
    assert forward_realized_vol(shocked, h).iloc[t] != pytest.approx(base)


# ─── NaN, nunca 0, al final ──────────────────────────────────────────────────

@pytest.mark.parametrize("h", [5, 21, 63])
def test_tail_is_nan(h):
    df, bench = _ohlc(), _ohlc(seed=1)["Close"]
    out = build_targets(df, bench, horizons=(h,), vol_horizons=(h,))
    for col in (f"dir_{h}", f"rel_{h}", f"log_rv_fwd_{h}"):
        assert out[col].iloc[-h:].isna().all()
        assert out[col].iloc[:-h].notna().all()


def test_binary_labels_are_0_or_1():
    out = build_targets(_ohlc(), _ohlc(seed=1)["Close"])
    for col in ("dir_5", "rel_5", "dir_63", "rel_63"):
        assert set(out[col].dropna().unique()) <= {0.0, 1.0}


# ─── Rendimiento relativo ────────────────────────────────────────────────────

def test_relative_return_against_itself_is_zero():
    c = _ohlc()["Close"]
    rel = relative_forward_return(c, c, 21).dropna()
    assert np.allclose(rel, 0.0)


def test_relative_return_missing_benchmark_date_is_nan():
    df = _ohlc()
    bench = df["Close"].drop(df.index[50])
    rel = relative_forward_return(df["Close"], bench, 5)
    assert np.isnan(rel.iloc[50])          # falta la fecha t
    assert np.isnan(rel.iloc[45])          # falta la fecha t+h


# ─── Volatilidad ─────────────────────────────────────────────────────────────

def test_vol_is_positive_and_annualized():
    rv = past_realized_vol(_ohlc(), 22).dropna()
    assert (rv > 0).all()
    assert 0.01 < rv.median() < 2.0


def test_vol_falls_back_when_high_equals_low():
    df = _ohlc()
    df.loc[df.index[10:20], ["High", "Low"]] = df.loc[df.index[10:20], ["Close"]].values
    rv = past_realized_vol(df, 5)
    assert rv.iloc[25] > 0 and np.isfinite(rv.iloc[25])


def test_vol_is_nan_when_price_is_frozen():
    """Etiqueta sin operaciones (precio congelado): vol futura NaN, no 0 ni el piso del log."""
    df = _ohlc()
    frozen = df.index[100:130]
    df.loc[frozen, ["Open", "High", "Low", "Close"]] = df["Close"].iloc[99]
    assert np.isnan(forward_realized_vol(df, 5).iloc[100])      # t+1..t+5 congelados
    assert forward_realized_vol(df, 5).iloc[90] > 0             # ventana con movimiento
    assert np.isnan(past_realized_vol(df, 5).iloc[110])         # feature: ventana congelada
    assert past_realized_vol(df, 1).iloc[110] == 0              # 1d: día plano es un 0 legítimo


def test_oscillators_stay_in_range_with_dirty_bars():
    """Close fuera de [Low, High] con rango ~0 no debe disparar stoch/williams."""
    from src.ml.features import compute_features
    df = _ohlc()
    bad = df.index[200:230]
    df.loc[bad, ["High", "Low"]] = df.loc[bad, ["Low"]].values
    df.loc[bad, "Close"] = df["Close"].iloc[199] * 1.05
    f = compute_features(df)
    assert f["stoch_k"].dropna().between(0, 100).all()
    assert f["williams_r"].dropna().between(-100, 0).all()


def test_har_recovers_persistent_signal():
    df = _ohlc(1000)
    X = extra_features(df).dropna()
    y = X["log_rv_5d"].shift(-5).dropna()
    X = X.loc[y.index]
    pred = HARModel().fit(X, y.to_numpy()).predict(X)
    m = regression_metrics(y.to_numpy(), pred, naive_forecast(X, 5))
    assert m["skill_vs_naive"] > 0


def test_qlike_is_zero_for_perfect_forecast():
    v = np.log(np.array([0.1, 0.2, 0.3]))
    assert qlike(v, v) == pytest.approx(0.0)
    assert qlike(v, v - 0.5) > 0


def test_target_helpers():
    assert target_horizon("rel_63") == 63
    assert target_horizon("log_rv_fwd_21") == 21
    assert is_regression("log_rv_fwd_5") and not is_regression("dir_5")


# ─── CV purgado por fechas ───────────────────────────────────────────────────

@pytest.mark.parametrize("h,embargo", [(5, 10), (63, 63)])
def test_date_splits_never_share_or_touch_dates(h, embargo):
    """En un panel, train y test no comparten fechas y hay h+embargo de hueco."""
    dates = pd.bdate_range("2010-01-01", periods=1500)
    panel = pd.Series(np.repeat(dates, 30))          # 30 tickers por fecha
    unique = pd.Series(np.arange(len(dates)), index=dates)
    n = 0
    for tr, te in purged_walk_forward_date_splits(panel, n_splits=6,
                                                  horizon=h, embargo=embargo):
        n += 1
        tr_pos = unique[panel.iloc[tr]].to_numpy()
        te_pos = unique[panel.iloc[te]].to_numpy()
        assert not set(tr_pos) & set(te_pos)
        assert tr_pos.max() + h + embargo < te_pos.min()
    assert n == 6


def test_date_splits_accept_unsorted_rows():
    dates = pd.Series(np.repeat(pd.bdate_range("2015-01-01", periods=600), 5))
    shuffled = dates.sample(frac=1.0, random_state=0).reset_index(drop=True)
    for tr, te in purged_walk_forward_date_splits(shuffled, n_splits=3):
        assert shuffled.iloc[tr].max() < shuffled.iloc[te].min()


# ─── M2a: entrada al Open siguiente y horizonte en días hábiles ──────────────

def test_default_build_targets_unchanged_by_new_options():
    """Con los defaults (close, filas) la salida es idéntica a la previa."""
    df, bench = _ohlc(), _ohlc(seed=1)["Close"]
    base = build_targets(df, bench, horizons=(5,), vol_horizons=(5,))
    explicito = build_targets(df, bench, horizons=(5,), vol_horizons=(5,),
                              entry="close", horizon_mode="rows")
    pd.testing.assert_frame_equal(base, explicito)
    assert list(base.columns) == ["dir_5", "rel_5", "log_rv_fwd_5"]


def test_next_open_return_uses_open_t1_and_close_t1_plus_h():
    df = _ohlc()
    r = forward_return_open(df["Open"], df["Close"], 5)
    t = 50
    esperado = df["Close"].iloc[t + 1 + 5] / df["Open"].iloc[t + 1] - 1.0
    assert r.iloc[t] == pytest.approx(esperado)
    assert r.iloc[-6:].isna().all()          # t+1+h no existe


def test_build_targets_next_open_matches_manual_label():
    df, bench_df = _ohlc(), _ohlc(seed=1)
    out = build_targets(df, bench_df["Close"], horizons=(5,), vol_horizons=(5,),
                        entry="next_open", bench_open=bench_df["Open"])
    t = 50
    r_stock = df["Close"].iloc[t + 6] / df["Open"].iloc[t + 1] - 1
    r_bench = bench_df["Close"].iloc[t + 6] / bench_df["Open"].iloc[t + 1] - 1
    assert out["dir_5"].iloc[t] == float(r_stock > 0)
    assert out["rel_5"].iloc[t] == float(r_stock - r_bench > 0)
    assert out["dir_5"].iloc[-6:].isna().all()


def test_next_open_requires_bench_open():
    df, bench = _ohlc(), _ohlc(seed=1)["Close"]
    with pytest.raises(ValueError):
        build_targets(df, bench, horizons=(5,), vol_horizons=(5,), entry="next_open")


def test_next_open_label_never_uses_prices_beyond_horizon():
    """La etiqueta de t vive hasta t+1+h: mover t+h+2 en adelante no la altera."""
    df, bench_df = _ohlc(), _ohlc(seed=1)
    t, h = 100, 5
    kw = dict(horizons=(h,), vol_horizons=(5,), entry="next_open")
    base = build_targets(df, bench_df["Close"], bench_open=bench_df["Open"], **kw)
    shocked = df.copy()
    shocked.iloc[t + h + 2:] *= 3.0
    after = build_targets(shocked, bench_df["Close"], bench_open=bench_df["Open"], **kw)
    assert base["dir_5"].iloc[t] == after["dir_5"].iloc[t]


def test_label_span_adds_one_day_for_next_open():
    assert label_span("dir_5", "close") == 5
    assert label_span("dir_5", "next_open") == 6
    assert label_span("log_rv_fwd_5", "next_open") == 5   # la vol no cambia de ventana


def test_business_days_differs_from_rows_when_series_has_gaps():
    """Un ticker con huecos: 5 filas ≠ 5 días hábiles del calendario común."""
    df, bench = _ohlc(60), _ohlc(60, seed=1)["Close"]
    calendar = df.index
    con_huecos = df.drop(df.index[[11, 12, 13]])       # faltan 3 días del calendario
    t_date = df.index[10]
    filas = build_targets(con_huecos, bench, horizons=(5,), vol_horizons=(5,))
    habiles = build_targets(con_huecos, bench, horizons=(5,), vol_horizons=(5,),
                            horizon_mode="business_days", calendar=calendar)
    # por filas: la salida es la fila 5 del ticker (= día 18 del calendario)
    sal_filas = con_huecos["Close"].iloc[con_huecos.index.get_loc(t_date) + 5]
    assert sal_filas == df["Close"].iloc[18]
    # por días hábiles: la salida es el día 15 del calendario
    esperado = float(df["Close"].iloc[15] / df["Close"].iloc[10] - 1 > 0)
    assert habiles["dir_5"].loc[t_date] == esperado
    assert filas["dir_5"].loc[t_date] == float(df["Close"].iloc[18] / df["Close"].iloc[10] - 1 > 0)


def test_business_days_keeps_original_index_and_nan_when_exit_day_missing():
    df, bench = _ohlc(60), _ohlc(60, seed=1)["Close"]
    con_huecos = df.drop(df.index[[15]])                # el día de salida de t=10 no cotizó
    out = build_targets(con_huecos, bench, horizons=(5,), vol_horizons=(5,),
                        horizon_mode="business_days", calendar=df.index)
    assert out.index.equals(con_huecos.index)
    assert np.isnan(out["dir_5"].loc[df.index[10]])


def test_business_days_equals_rows_on_a_complete_series():
    df, bench = _ohlc(), _ohlc(seed=1)["Close"]
    a = build_targets(df, bench, horizons=(5, 21), vol_horizons=(5,))
    b = build_targets(df, bench, horizons=(5, 21), vol_horizons=(5,),
                      horizon_mode="business_days", calendar=df.index)
    pd.testing.assert_frame_equal(a, b)


def test_business_days_requires_calendar():
    df, bench = _ohlc(), _ohlc(seed=1)["Close"]
    with pytest.raises(ValueError):
        build_targets(df, bench, horizon_mode="business_days")


def test_return_cols_expose_underlying_returns():
    df, bench = _ohlc(), _ohlc(seed=1)["Close"]
    out = build_targets(df, bench, horizons=(5,), vol_horizons=(5,), return_cols=True)
    assert (out["dir_5"].dropna() == (out["ret_dir_5"].dropna() > 0).astype(float)).all()
    assert (out["rel_5"].dropna() == (out["ret_rel_5"].dropna() > 0).astype(float)).all()
