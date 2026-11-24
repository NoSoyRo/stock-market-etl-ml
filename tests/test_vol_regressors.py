"""Tests de src/ml/vol_regressors: causalidad, formulas y arnés (datos sintéticos)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import vol_regressors as vr


def make_ohlcv(n: int = 400, seed: int = 0, drift_vol: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, drift_vol, n)))
    gap = rng.normal(0, 0.005, n)
    open_ = np.r_[close[0], close[:-1]] * np.exp(gap)
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.008, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.008, n)))
    vol = rng.integers(1_000, 100_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close,
                         "Volume": vol}, index=idx)


def make_universe(n_names: int = 25, n: int = 400) -> dict[str, pd.DataFrame]:
    return {f"T{i}": make_ohlcv(n, seed=i) for i in range(n_names)}


def test_registry_contiene_los_regresores_esperados():
    esperados = {"overnight", "garman_klass", "rogers_satchell", "semivar",
                 "attention", "absorption", "dispersion"}
    assert esperados <= set(vr.REGRESSORS)
    for reg in vr.REGRESSORS.values():
        assert reg.cols and reg.kind in ("ticker", "universe")


@pytest.mark.parametrize("name", ["overnight", "garman_klass", "rogers_satchell",
                                  "semivar", "attention"])
def test_regresores_de_ticker_son_causales(name):
    """Alterar el futuro (t > T) no cambia los valores hasta T."""
    df = make_ohlcv(400)
    T = 300
    full = vr.compute_regressor(name, {"A": df})["A"]
    alt = df.copy()
    alt.iloc[T + 1:] = alt.iloc[T + 1:] * 1.7 + 3.0     # futuro distinto
    cut = vr.compute_regressor(name, {"A": alt})["A"]
    a = full.iloc[: T + 1]
    b = cut.iloc[: T + 1]
    pd.testing.assert_frame_equal(a, b)
    assert list(full.columns) == list(vr.REGRESSORS[name].cols)


@pytest.mark.parametrize("name", ["absorption", "dispersion"])
def test_regresores_de_universo_son_causales(name):
    raws = make_universe()
    T = 330
    full = vr.compute_regressor(name, raws)["T0"]
    alt = {k: v.copy() for k, v in raws.items()}
    for v in alt.values():
        v.iloc[T + 1:] = v.iloc[T + 1:] * 2.0 + 1.0
    cut = vr.compute_regressor(name, alt)["T0"]
    pd.testing.assert_frame_equal(full.iloc[: T + 1], cut.iloc[: T + 1])
    assert full.iloc[: T + 1].notna().any().any()


def test_overnight_formula():
    df = make_ohlcv(60)
    out = vr.compute_regressor("overnight", {"A": df})["A"]
    t = 30
    gap = np.log(df["Open"].iloc[t] / df["Close"].iloc[t - 1])
    assert out["log_gap2_1d"].iloc[t] == pytest.approx(np.log(gap ** 2 + vr.EPS_VAR))
    g = np.log(df["Open"] / df["Close"].shift(1))
    assert out["log_gap2_5d"].iloc[t] == pytest.approx(
        np.log((g ** 2).iloc[t - 4: t + 1].mean() + vr.EPS_VAR))


def test_garman_klass_y_rogers_satchell_recuperan_la_varianza():
    """Con precios sintéticos de vol conocida, ambos estiman ~σ² diaria."""
    rng = np.random.default_rng(1)
    n, sigma, m = 4000, 0.02, 50
    # Browniano fino: m pasos por día; O/H/L/C salen del camino intradía.
    steps = rng.normal(0, sigma / np.sqrt(m), (n, m))
    path = 100 * np.exp(np.cumsum(steps.ravel())).reshape(n, m)
    prev = np.r_[100.0, path[:-1, -1]]
    full = np.c_[prev, path]
    df = pd.DataFrame({"Open": prev, "High": full.max(1), "Low": full.min(1),
                       "Close": path[:, -1], "Volume": 1.0},
                      index=pd.bdate_range("2000-01-01", periods=n))
    for name, col in (("garman_klass", "log_gk_22d"), ("rogers_satchell", "log_rs_22d")):
        out = vr.compute_regressor(name, {"A": df})["A"][col].dropna()
        # media de la varianza estimada ≈ σ² (con sesgo discreto pequeño)
        assert np.exp(out).mean() == pytest.approx(sigma ** 2, rel=0.35)


def test_semivarianza_separa_bajas_y_alzas():
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = pd.Series(100.0, index=idx)
    close.iloc[30:] = 110.0                       # un solo salto al alza
    df = pd.DataFrame({"Open": close, "High": close, "Low": close,
                       "Close": close, "Volume": 1.0})
    out = vr.compute_regressor("semivar", {"A": df})["A"]
    t = 40                                        # ventana 22d incluye el salto (día 30)
    assert out["log_semi_dn_22d"].iloc[t] == pytest.approx(np.log(vr.EPS_VAR))
    assert out["log_semi_up_22d"].iloc[t] > np.log(vr.EPS_VAR) + 1


def test_atencion_no_usa_el_dia_actual_en_su_escala():
    """|gap|/σ y rango/ATR usan escalas hasta t-1: un shock en t se ve completo."""
    df = make_ohlcv(120)
    base = vr.compute_regressor("attention", {"A": df})["A"]
    shock = df.copy()
    t = 100
    shock.iloc[t, shock.columns.get_loc("High")] *= 1.5
    out = vr.compute_regressor("attention", {"A": shock})["A"]
    assert out["range_atr"].iloc[t] > base["range_atr"].iloc[t]
    # el día t+1 sí puede cambiar (el ATR incluye t), pero antes de t no
    pd.testing.assert_frame_equal(base.iloc[:t], out.iloc[:t])


def test_absorcion_sube_con_un_factor_comun():
    n = 500
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=n)
    market = rng.normal(0, 0.01, n)

    def uni(loading: float) -> dict[str, pd.DataFrame]:
        out = {}
        for i in range(25):
            r = loading * market + np.sqrt(1 - loading ** 2) * rng.normal(0, 0.01, n)
            c = 100 * np.exp(np.cumsum(r))
            out[f"T{i}"] = pd.DataFrame({"Open": c, "High": c * 1.001, "Low": c * 0.999,
                                         "Close": c, "Volume": 1.0}, index=idx)
        return out

    raw_lo = vr.absorption_raw(uni(0.1))
    raw_hi = vr.absorption_raw(uni(0.9))
    assert raw_hi.dropna().mean() > raw_lo.dropna().mean() + 0.2


def test_evaluate_regressor_detecta_senal_incremental_y_ruido():
    """Un regresor que explica el residuo del HAR suma skill; el ruido no."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2010-01-01", periods=1500)
    rows = []
    for tk in range(8):
        log5 = rng.normal(-1.5, 0.4, len(dates))
        log22 = log5 * 0.6 + rng.normal(-0.6, 0.2, len(dates))
        log1 = log5 + rng.normal(0, 0.3, len(dates))
        sig = rng.normal(0, 1, len(dates))
        y = 0.5 * log5 + 0.3 * log22 + 0.4 * sig + rng.normal(0, 0.2, len(dates))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"T{tk}", "log_rv_1d": log1, "log_rv_5d": log5,
            "log_rv_22d": log22, "sig": sig, "ruido": rng.normal(0, 1, len(dates)),
            "log_rv_fwd_5": y}))
    panel = pd.concat(rows, ignore_index=True).sort_values("date", kind="stable")
    good = vr.evaluate_regressor(panel, "log_rv_fwd_5", 5, ["sig"], n_splits=3,
                                 n_boot=200, seed=0)
    bad = vr.evaluate_regressor(panel, "log_rv_fwd_5", 5, ["ruido"], n_splits=3,
                                n_boot=200, seed=0)
    assert good["delta_skill_fold_mean"] > 0.05
    assert good["delta_ci_lo"] > 0 and good["dm_p"] < 0.01
    assert abs(bad["delta_skill_fold_mean"]) < 0.01
    assert bad["n_obs"] == good["n_obs"]


def test_veredicto_usa_el_criterio_declarado():
    assert vr.verdict(0.015, 0.002) == "conserva (robusto)"
    assert vr.verdict(0.015, -0.003) == "conserva (no robusto)"
    assert vr.verdict(0.004, 0.001) == "descarta"
