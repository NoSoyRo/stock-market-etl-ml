"""Tests de M10: σ̂ sin futuro, B/C/D alineados por fecha, D no recorta el test, determinismo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.cs_targets import (
    build_variant, constant_by_date_cols, daily_sigma_pct, neutral_zone_mask, pct_by_date,
    sigma_for_panel, target_a, target_b, target_c,
)
from src.ml.rank_models import RidgeArm


def _panel(n_dates: int = 60, n_names: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    d = np.repeat(dates, n_names)
    df = pd.DataFrame({"ticker": np.tile([f"T{i}" for i in range(n_names)], n_dates)}, index=d)
    df["rn_a"] = df.groupby(df.index)["ticker"].transform(lambda s: rng.permutation(len(s)) / len(s))
    df["rn_mkt"] = np.repeat(rng.uniform(size=n_dates), n_names)  # constante por fecha
    df["fwd_ret"] = 4 * (df["rn_a"] - 0.5) + rng.normal(scale=2, size=len(d))
    return df


def test_sigma_sin_futuro():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(scale=0.01, size=120))),
                      index=pd.bdate_range("2021-01-01", periods=120))
    base = daily_sigma_pct(close)
    alt = close.copy()
    alt.iloc[80:] *= 3.0                      # cambia solo el futuro de t=79
    alt_s = daily_sigma_pct(alt)
    pd.testing.assert_series_equal(base.iloc[:80], alt_s.iloc[:80])
    assert base.iloc[:19].isna().all() and base.iloc[40:].notna().all()
    assert 0.3 < base.dropna().mean() < 3.0   # ≈ 1 % diario


def test_sigma_for_panel_alinea_por_fecha_y_ticker():
    p = _panel(30, 3)
    sig = {"T0": pd.Series(1.0, index=p.index.unique()), "T1": pd.Series(2.0, index=p.index.unique())}
    out = sigma_for_panel(p, sig)
    assert (out[p["ticker"] == "T0"] == 1.0).all() and (out[p["ticker"] == "T1"] == 2.0).all()
    assert np.isnan(out[p["ticker"] == "T2"]).all()


def test_variantes_alineadas_y_en_rango_por_fecha():
    p = _panel()
    sigma = np.random.default_rng(2).uniform(0.5, 3, size=len(p))
    for t in (target_a(p["fwd_ret"].to_numpy(), p.index),
              target_b(p["fwd_ret"].to_numpy(), sigma, p.index),
              target_c(p["fwd_ret"].to_numpy(), p.index)):
        s = pd.Series(t, index=p.index)
        assert len(t) == len(p) and not np.isnan(t).any()
        assert (s.groupby(level=0).max() == 1.0).all()           # el mejor de cada fecha = 1
        assert (s.groupby(level=0).min() > 0).all()


def test_c_equivale_a_a_cuando_se_resta_una_constante_por_fecha():
    p = _panel()
    a = target_a(p["fwd_ret"].to_numpy(), p.index)
    c = target_c(p["fwd_ret"].to_numpy(), p.index)
    np.testing.assert_allclose(a, c)   # restar la media de la fecha no cambia el orden


def test_b_difiere_de_a_y_usa_sigma_de_su_fila():
    p = _panel()
    sigma = np.tile(np.linspace(0.5, 5, 12), 60)
    a = target_a(p["fwd_ret"].to_numpy(), p.index)
    b = target_b(p["fwd_ret"].to_numpy(), sigma, p.index)
    assert not np.allclose(a, b)
    # con σ̂ constante, B == A
    b1 = target_b(p["fwd_ret"].to_numpy(), np.full(len(p), 2.0), p.index)
    np.testing.assert_allclose(a, b1)


def test_b_sigma_invalida_da_nan():
    p = _panel(10, 4)
    sigma = np.ones(len(p)); sigma[0] = np.nan; sigma[1] = 0.0
    b = target_b(p["fwd_ret"].to_numpy(), sigma, p.index)
    assert np.isnan(b[:2]).all() and not np.isnan(b[2:]).any()


def test_mascara_zona_neutra():
    fwd = np.array([0.5, -3.0, 3.0, 1.0, np.nan])
    sig = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    m = neutral_zone_mask(fwd, sig, k=0.5, horizon=5)       # umbral = 0.5·√5 ≈ 1.118
    assert m.tolist() == [False, True, True, False, False]
    assert not neutral_zone_mask(np.array([9.0]), np.array([np.nan])).any()


def test_d_no_recorta_el_test():
    p = _panel()
    sigma = np.full(len(p), 2.0)
    out, cols = build_variant(p, "D", sigma, ["rn_a", "rn_mkt"])
    assert len(out) == len(p)                       # el panel conserva TODAS las filas
    assert 0 < out["train_ok"].mean() < 1           # el filtro solo vive en train_ok
    np.testing.assert_allclose(out["pct_fwd"], target_a(p["fwd_ret"].to_numpy(), p.index))
    assert cols == ["rn_a", "rn_mkt"]
    # predecir sobre el test completo da un score por fila
    tr, te = out.iloc[:400], out.iloc[400:]
    arm = RidgeArm(cols).fit(tr[tr["train_ok"]])
    assert len(arm.predict(te)) == len(te)


def test_e_quita_columnas_constantes_por_fecha():
    p = _panel()
    assert constant_by_date_cols(p, ["rn_a", "rn_mkt"]) == ["rn_mkt"]
    _, cols = build_variant(p, "E", np.ones(len(p)), ["rn_a", "rn_wti_ret5", "rn_usdmxn_ret5"])
    assert cols == ["rn_a"]
    assert build_variant(p, "A", np.ones(len(p)), ["rn_a", "rn_wti_ret5"])[1] == ["rn_a", "rn_wti_ret5"]


def test_constantes_toleran_desfases_ocasionales():
    p = _panel()
    p.iloc[3, p.columns.get_loc("rn_mkt")] += 0.1      # una fecha desfasada de 60
    assert constant_by_date_cols(p, ["rn_a", "rn_mkt"]) == ["rn_mkt"]


def test_no_modifica_entrada_y_variante_invalida():
    p = _panel()
    antes = p.copy()
    build_variant(p, "B", np.ones(len(p)), ["rn_a"])
    pd.testing.assert_frame_equal(p, antes)
    with pytest.raises(ValueError):
        build_variant(p, "Z", np.ones(len(p)), ["rn_a"])


def test_determinismo():
    p = _panel()
    sigma = np.random.default_rng(3).uniform(0.5, 3, size=len(p))
    r1 = [build_variant(p, v, sigma, ["rn_a", "rn_mkt"])[0] for v in "ABCDE"]
    r2 = [build_variant(p, v, sigma, ["rn_a", "rn_mkt"])[0] for v in "ABCDE"]
    for a, b in zip(r1, r2):
        pd.testing.assert_frame_equal(a, b)
    s = [pct_by_date(p["fwd_ret"], p.index) for _ in range(2)]
    np.testing.assert_array_equal(s[0], s[1])
