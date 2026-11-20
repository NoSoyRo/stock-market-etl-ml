"""Pruebas del vol targeting: sin mirar al futuro, tope, costo y sin apalancamiento."""
import numpy as np
import pandas as pd
import pytest

from src.ml.vol_target import ewma_vol, target_exposure, vol_targeted_returns


def _series(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=idx)


def test_exposure_is_target_over_sigma_with_cap():
    sig = pd.Series([0.05, 0.10, 0.20, 0.40])
    w = target_exposure(sig, target_vol=0.10, cap=1.0)
    assert w.tolist() == pytest.approx([1.0, 1.0, 0.5, 0.25])


def test_exposure_never_exceeds_cap_and_no_leverage():
    sig = pd.Series(np.linspace(0.01, 1.0, 100))
    assert target_exposure(sig, 0.10, cap=1.0).max() <= 1.0
    assert target_exposure(sig, 0.10, cap=0.6).max() <= 0.6
    with pytest.raises(ValueError):
        target_exposure(sig, 0.10, cap=1.5)


def test_exposure_nan_sigma_stays_nan():
    w = target_exposure(pd.Series([np.nan, 0.2]), 0.1)
    assert np.isnan(w.iloc[0]) and w.iloc[1] == pytest.approx(0.5)


def test_weight_uses_lagged_sigma_not_future():
    """Cambiar σ̂ en t+1 no puede alterar el retorno de t."""
    r = _series()
    sig = pd.Series(0.2, index=r.index)
    base = vol_targeted_returns(r, sig, 0.10, cost_per_unit_change=0.0)
    sig2 = sig.copy()
    sig2.iloc[150:] = 0.8  # choque desde la posición 150
    alt = vol_targeted_returns(r, sig2, 0.10, cost_per_unit_change=0.0)
    # La salida omite el día 0 (sin σ̂ previa). El día de posición 150 usa σ̂ hasta
    # 149 (igual); el de posición 151 usa σ̂[150] (distinta).
    day150, day151 = r.index[150], r.index[151]
    assert alt.loc[:day150].equals(base.loc[:day150])
    assert alt.loc[day151] != pytest.approx(base.loc[day151])


def test_weight_applies_to_next_day():
    r = pd.Series([0.01, 0.02, 0.03], index=pd.bdate_range("2021-01-01", periods=3))
    sig = pd.Series([0.20, 0.40, 0.10], index=r.index)
    out = vol_targeted_returns(r, sig, 0.10, lag=1, cost_per_unit_change=0.0)
    # Día 0 sin σ̂ previa → no hay peso. Día 1: 0.1/0.2=0.5; día 2: 0.1/0.4=0.25.
    assert out.index.tolist() == r.index[1:].tolist()
    assert out.tolist() == pytest.approx([0.5 * 0.02, 0.25 * 0.03])


def test_cost_charged_per_weight_change():
    r = pd.Series([0.0, 0.0, 0.0, 0.0], index=pd.bdate_range("2021-01-01", periods=4))
    sig = pd.Series([0.2, 0.2, 0.4, 0.4], index=r.index)
    out = vol_targeted_returns(r, sig, 0.10, lag=1, cost_per_unit_change=0.001)
    # Pesos: [nan, 0.5, 0.5, 0.25]. Entrada 0→0.5 cuesta 0.0005; luego 0; luego 0.25·0.001.
    assert out.tolist() == pytest.approx([-0.0005, 0.0, -0.00025])


def test_constant_sigma_costs_only_entry():
    r = _series()
    sig = pd.Series(0.3, index=r.index)
    with_cost = vol_targeted_returns(r, sig, 0.10, cost_per_unit_change=0.0005)
    no_cost = vol_targeted_returns(r, sig, 0.10, cost_per_unit_change=0.0)
    diff = (no_cost - with_cost)
    assert diff.iloc[0] == pytest.approx(0.0005 * (0.10 / 0.3))
    assert diff.iloc[1:].abs().max() == pytest.approx(0.0)


def test_returns_never_levered():
    """Con tope 1, el retorno escalado nunca supera en magnitud al original (sin costo)."""
    r = _series()
    sig = pd.Series(0.01, index=r.index)  # σ̂ baja: sin tope habría apalancamiento
    out = vol_targeted_returns(r, sig, 0.10, cost_per_unit_change=0.0)
    assert (out.abs() <= r.loc[out.index].abs() + 1e-12).all()


def test_ewma_vol_is_causal_and_annualized():
    r = _series(400)
    v = ewma_vol(r, span=10)
    r2 = r.copy()
    r2.iloc[300:] = 0.5
    v2 = ewma_vol(r2, span=10)
    pd.testing.assert_series_equal(v.iloc[:300], v2.iloc[:300])
    assert 0.05 < v.dropna().median() < 0.4  # ≈ 0.01·√252 ≈ 0.16


def test_har_oos_is_causal_and_leaves_early_nan():
    from src.ml.vol_target import har_oos_vol
    r = _series(1500, seed=3)
    p = har_oos_vol(r)
    assert p.iloc[:300].isna().all()
    assert p.notna().sum() > 300 and (p.dropna() > 0).all()
    # Perturbar solo el último bloque no cambia predicciones de bloques previos.
    r2 = r.copy()
    r2.iloc[-50:] = 0.2
    p2 = har_oos_vol(r2)
    first_blocks = p.dropna().index[:200]
    pd.testing.assert_series_equal(p.loc[first_blocks], p2.loc[first_blocks])
