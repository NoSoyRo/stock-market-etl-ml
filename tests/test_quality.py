"""Tests de la máscara de calidad (datos sintéticos)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.quality import apply_mask, compute_quality_mask


def _ohlcv(n: int = 100, start: str = "2020-01-01", vol: float = 1e6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.005, n))
    return pd.DataFrame(
        {"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": vol}, index=idx
    )


def test_serie_limpia_es_valida():
    m = compute_quality_mask(_ohlcv())
    assert m["valid"].all()
    assert not m[["zero_volume", "flat_run", "bad_ohlc", "suspect_jump", "long_gap"]].any().any()


def test_volumen_cero():
    df = _ohlcv()
    df.iloc[10, df.columns.get_loc("Volume")] = 0
    m = compute_quality_mask(df)
    assert m["zero_volume"].iloc[10] and not m["valid"].iloc[10]
    assert m["zero_volume"].sum() == 1


def test_precio_congelado_tipo_posadas():
    df = _ohlcv()
    for col in ("Open", "High", "Low", "Close"):
        df.iloc[20:30, df.columns.get_loc(col)] = 5.2
    df.iloc[20:30, df.columns.get_loc("Volume")] = 0
    m = compute_quality_mask(df)
    assert m["flat_run"].iloc[22:30].all()      # desde la 3.ª repetición
    assert not m["flat_run"].iloc[:22].any()
    assert not m["valid"].iloc[20:30].any()


def test_cierres_iguales_con_volumen_y_rango_no_es_flat_run():
    df = _ohlcv()
    df.iloc[40:45, df.columns.get_loc("Close")] = 100.0
    df.iloc[40:45, df.columns.get_loc("Open")] = 100.0
    df.iloc[40:45, df.columns.get_loc("High")] = 101.0
    df.iloc[40:45, df.columns.get_loc("Low")] = 99.0
    assert not compute_quality_mask(df)["flat_run"].any()


def test_salto_con_rebote_marca_dia_del_rebote():
    df = _ohlcv()
    c = df.columns.get_loc("Close")
    base = df.iloc[49, c]
    for col in ("Open", "Close"):
        df.iloc[50, df.columns.get_loc(col)] = base * 1.5
    df.iloc[50, df.columns.get_loc("High")] = base * 1.51
    m = compute_quality_mask(df)
    assert not m["suspect_jump"].iloc[50]       # causal: aún no se sabe el rebote
    # el día 51 vuelve al nivel anterior
    for col in ("Open", "Close"):
        df.iloc[51, df.columns.get_loc(col)] = base
    m = compute_quality_mask(df)
    assert m["suspect_jump"].iloc[51] and not m["valid"].iloc[51]


def test_salto_con_volumen_sin_rebote_no_se_marca():
    df = _ohlcv()
    for col in ("Open", "Close"):
        df.iloc[50:, df.columns.get_loc(col)] *= 1.5
    df.iloc[50:, df.columns.get_loc("High")] *= 1.5
    df.iloc[50:, df.columns.get_loc("Low")] *= 1.5
    assert not compute_quality_mask(df)["suspect_jump"].any()


def test_salto_con_volumen_cero_se_marca():
    df = _ohlcv()
    for col in ("Open", "Close"):
        df.iloc[50:, df.columns.get_loc(col)] *= 1.5
    df.iloc[50:, df.columns.get_loc("High")] *= 1.5
    df.iloc[50:, df.columns.get_loc("Low")] *= 1.5
    df.iloc[50, df.columns.get_loc("Volume")] = 0
    assert compute_quality_mask(df)["suspect_jump"].iloc[50]


def test_hueco_largo():
    df = _ohlcv()
    df = df.drop(df.index[30:40])               # ~14 días calendario
    m = compute_quality_mask(df)
    assert m["long_gap"].sum() == 1 and m["long_gap"].iloc[30]


def test_ohlc_invalido():
    df = _ohlcv()
    df.iloc[5, df.columns.get_loc("High")] = df.iloc[5]["Low"] - 1   # High < Low
    df.iloc[6, df.columns.get_loc("Close")] = df.iloc[6]["High"] + 1  # Close > High
    df.iloc[7, df.columns.get_loc("Close")] = 0                       # precio ≤ 0
    m = compute_quality_mask(df)
    assert m["bad_ohlc"].iloc[5:8].all() and m["bad_ohlc"].sum() == 3


def test_iliquido_usa_solo_pasado():
    df = _ohlcv(n=120, vol=10.0)                # Close×Vol ≈ 1000 < umbral
    m = compute_quality_mask(df, liq_threshold=1e6)
    assert m["illiquid"].iloc[30:].all()
    # un pico de volumen hoy no cambia la bandera de hoy (ventana desplazada 1 día)
    df2 = df.copy()
    df2.iloc[100, df2.columns.get_loc("Volume")] = 1e12
    assert compute_quality_mask(df2, liq_threshold=1e6)["illiquid"].iloc[100]
    # illiquid no invalida la fila
    assert m["valid"].all()


def test_macro_sin_banderas_de_volumen():
    df = _ohlcv(vol=0)
    m = compute_quality_mask(df, kind="macro")
    assert not m[["zero_volume", "flat_run", "illiquid"]].any().any()
    assert m["valid"].all()


def test_benchmark_marca_pero_no_excluye():
    df = _ohlcv()
    df.iloc[10, df.columns.get_loc("Volume")] = 0
    m = compute_quality_mask(df, kind="benchmark")
    assert m["zero_volume"].iloc[10] and m["valid"].iloc[10]


def test_causalidad():
    df = _ohlcv(n=200)
    df.iloc[50, df.columns.get_loc("Volume")] = 0
    df = df.drop(df.index[80:90])
    full = compute_quality_mask(df)
    for cut in (60, 100, 150):
        part = compute_quality_mask(df.iloc[:cut])
        pd.testing.assert_frame_equal(part, full.iloc[:cut])


def test_vacio():
    m = compute_quality_mask(pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
    assert m.empty


def test_apply_mask():
    df = _ohlcv(n=20)
    df.iloc[3, df.columns.get_loc("Volume")] = 0
    m = compute_quality_mask(df)
    long = m.copy()
    long.insert(0, "date", df.index)
    long.insert(0, "ticker", "X")
    out = apply_mask(df, long.reset_index(drop=True), "X")
    assert len(out) == 19 and df.index[3] not in out.index
    assert len(apply_mask(df, long.reset_index(drop=True), "OTRO")) == 20


# ---------- N7: causalidad y cobertura ----------

def test_causalidad_serie_cortada_vs_completa():
    """Agregar datos futuros no cambia banderas pasadas."""
    df = _ohlcv(400, seed=3)
    df.iloc[50, df.columns.get_loc("Volume")] = 0
    df.iloc[120:123, df.columns.get_loc("Close")] = df["Close"].iloc[119]
    df.iloc[200, df.columns.get_loc("Close")] *= 1.4
    df.iloc[201, df.columns.get_loc("Close")] *= 0.7
    half = len(df) // 2
    full = compute_quality_mask(df)
    cut = compute_quality_mask(df.iloc[:half])
    pd.testing.assert_frame_equal(full.iloc[:half], cut)


def test_mask_coverage_ausentes_y_fechas_posteriores():
    from src.data.quality import mask_coverage

    idx = pd.bdate_range("2020-01-01", periods=10)
    mask = pd.DataFrame({"ticker": "AAA", "date": idx, "valid": True})
    dates = pd.bdate_range("2020-01-06", periods=10)  # algunas > idx.max()
    cov = mask_coverage(mask, ["AAA", "BBB"], dates)
    assert cov["missing_tickers"] == ["BBB"]
    expected = float((dates > idx.max()).mean())
    assert cov["frac_after_max"]["AAA"] == expected
    assert "BBB" not in cov["frac_after_max"]


def test_mask_coverage_completa():
    from src.data.quality import mask_coverage

    idx = pd.bdate_range("2020-01-01", periods=10)
    mask = pd.DataFrame({"ticker": "AAA", "date": idx, "valid": True})
    cov = mask_coverage(mask, ["AAA"], idx)
    assert cov["missing_tickers"] == [] and cov["frac_after_max"]["AAA"] == 0.0
