"""Pruebas de cross_section: IC por fecha, baselines y bootstrap (datos sintéticos)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.shared.evaluators.cross_section import (
    auc_by_date,
    equal_weight_returns,
    ic_by_date,
    ic_summary,
    percentile_in,
    random_topk_returns,
    topk_returns,
)


def _panel(n_dates: int = 600, n_names: int = 30, rho: float = 0.0, seed: int = 1) -> pd.DataFrame:
    """Panel con score y retorno; correlación objetivo `rho` por fecha."""
    rng = np.random.default_rng(seed)
    fechas = pd.bdate_range("2020-01-01", periods=n_dates)
    s = rng.standard_normal((n_dates, n_names))
    e = rng.standard_normal((n_dates, n_names))
    r = rho * s + np.sqrt(1 - rho**2) * e
    idx = np.repeat(fechas, n_names)
    return pd.DataFrame({
        "ticker": np.tile([f"T{i}" for i in range(n_names)], n_dates),
        "score": s.ravel(),
        "ret": r.ravel() * 0.02,
    }, index=idx)


def test_ic_recupera_senal_plantada():
    ic = ic_by_date(_panel(rho=0.10), "score", "ret")
    assert abs(ic.mean() - 0.10) < 0.02


def test_ic_nulo_no_es_significativo():
    res = ic_summary(ic_by_date(_panel(rho=0.0), "score", "ret"))
    assert abs(res["mean"]) < 0.02 and abs(res["t"]) < 3


def test_ic_pearson_y_spearman_coinciden_en_monotona():
    p = _panel(rho=0.3)
    a = ic_by_date(p, "score", "ret")
    b = ic_by_date(p, "score", "ret", method="pearson")
    assert abs(a.mean() - b.mean()) < 0.02


def test_omite_fechas_con_pocos_nombres():
    p = _panel(n_dates=50, n_names=30)
    p = p[~((p.index == p.index[-1]) & (p["ticker"].str[1:].astype(int) >= 5))]  # última fecha: 5 nombres
    ic = ic_by_date(p, "score", "ret", min_names=10)
    assert len(ic) == 49 and p.index[-1] not in ic.index


def test_ic_ignora_nan_y_orden():
    p = _panel(n_dates=40)
    p.iloc[3, p.columns.get_loc("ret")] = np.nan
    barajado = p.sample(frac=1, random_state=0)
    assert np.allclose(ic_by_date(p, "score", "ret"), ic_by_date(barajado, "score", "ret"))


def test_summary_reproducible_y_ci_contiene_media():
    ic = ic_by_date(_panel(rho=0.1), "score", "ret")
    a = ic_summary(ic, step=5, seed=3)
    b = ic_summary(ic, step=5, seed=3)
    assert a == b
    assert a["n"] == len(ic[::5]) and a["ci_lo"] < a["mean"] < a["ci_hi"]
    assert ic_summary(ic, step=5, seed=4)["ci_lo"] != a["ci_lo"]


def test_summary_con_pocos_datos():
    assert np.isnan(ic_summary(pd.Series([0.1]))["t"])


def test_auc_perfecto_e_invertido():
    fechas = pd.bdate_range("2021-01-01", periods=5)
    idx = np.repeat(fechas, 12)
    sc = np.tile(np.arange(12.0), 5)
    y = (sc >= 6).astype(int)
    p = pd.DataFrame({"y": y, "score": sc}, index=idx)
    assert np.allclose(auc_by_date(p, "y", "score"), 1.0)
    p["score"] = -sc
    assert np.allclose(auc_by_date(p, "y", "score"), 0.0)


def test_topk_elige_los_nombres_correctos():
    fechas = pd.bdate_range("2021-01-01", periods=3)
    sc = np.tile(np.arange(12.0), 3)
    p = pd.DataFrame({"score": sc, "ret": sc / 100}, index=np.repeat(fechas, 12))
    top = topk_returns(p, "score", "ret", 3)
    assert np.allclose(top, (0.11 + 0.10 + 0.09) / 3)
    bajo = topk_returns(p, "score", "ret", 3, ascending=True)
    assert np.allclose(bajo, (0.0 + 0.01 + 0.02) / 3)
    assert np.allclose(equal_weight_returns(p, "ret"), 0.055)


def test_random_topk_reproducible_y_centrado():
    p = _panel(n_dates=60)
    a = random_topk_returns(p, "ret", 5, n_draws=200, seed=7)
    b = random_topk_returns(p, "ret", 5, n_draws=200, seed=7)
    assert a.shape == (60, 200) and a.equals(b)
    assert not a.equals(random_topk_returns(p, "ret", 5, n_draws=200, seed=8))
    ew = equal_weight_returns(p, "ret")
    assert abs((a.mean(axis=1) - ew).abs().mean()) < 0.005  # el azar promedia al universo


def test_percentile_in():
    assert percentile_in(100, [1, 2, 3]) == 100.0
    assert percentile_in(-1, [1, 2, 3]) == 0.0
    assert percentile_in(2, [1, 2, 3]) == 50.0
