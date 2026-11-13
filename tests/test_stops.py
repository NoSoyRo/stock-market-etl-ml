"""M8 · Tests de stops por σ̂, Kelly estimado, etiqueta de cola y golden de risk_engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import stops
from src.ml.risk_engine import build_signal, kelly_fraction


def _close(n: int = 120, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


# ── σ̂ causal ─────────────────────────────────────────────────────────────────
def test_sigma_no_mira_al_futuro():
    """Cambiar precios posteriores a t no altera σ̂ en t."""
    c = _close()
    s1 = stops.daily_sigma(c, span=21)
    c2 = c.copy()
    c2.iloc[80:] *= 1.5  # shock futuro
    s2 = stops.daily_sigma(c2, span=21)
    pd.testing.assert_series_equal(s1.iloc[:80], s2.iloc[:80])


def test_sigma_lag_rezaga_un_periodo():
    c = _close()
    s0 = stops.daily_sigma(c, span=21, lag=0)
    s1 = stops.daily_sigma(c, span=21, lag=1)
    pd.testing.assert_series_equal(s1.iloc[1:], s0.shift(1).iloc[1:])


def test_sigma_es_diaria_no_anualizada():
    c = _close(400)
    s = stops.daily_sigma(c, span=21).dropna()
    assert 0.005 < s.mean() < 0.02  # ≈ 1% diario de los datos sintéticos


def test_stop_levels_usan_sigma_horizonte():
    stop, tgt = stops.sigma_stop_levels(100.0, sigma_daily=0.02, horizon=5,
                                        mult_stop=1.0, mult_target=1.5)
    sh = 0.02 * np.sqrt(5)
    assert stop == pytest.approx(100 * (1 - sh))
    assert tgt == pytest.approx(100 * (1 + 1.5 * sh))


def test_stop_mas_ancho_con_mas_vol():
    s_lo, _ = stops.sigma_stop_levels(100.0, 0.01, 5)
    s_hi, _ = stops.sigma_stop_levels(100.0, 0.03, 5)
    assert s_hi < s_lo


# ── tamaño inverso a σ̂ ───────────────────────────────────────────────────────
def test_tamano_inverso_a_sigma_y_con_tope():
    f1 = stops.inverse_vol_size(0.02, risk_budget=0.005, horizon=5, stop_mult=1.0, max_f=1.0)
    f2 = stops.inverse_vol_size(0.04, risk_budget=0.005, horizon=5, stop_mult=1.0, max_f=1.0)
    assert f1 == pytest.approx(2 * f2)
    assert stops.inverse_vol_size(1e-6, max_f=0.10) == 0.10
    assert stops.inverse_vol_size(0.0, max_f=0.10) == 0.0  # σ̂ inválida → no apostar
    assert stops.inverse_vol_size(float("nan")) == 0.0


# ── Kelly estimado ────────────────────────────────────────────────────────────
def test_kelly_estimado_cero_si_edge_no_positivo():
    pay = stops.Payoffs(avg_win=2.0, avg_loss=2.0)
    assert stops.kelly_fraction_estimated(0.50, pay) == 0.0   # edge = 0
    assert stops.kelly_fraction_estimated(0.40, pay) == 0.0   # edge < 0
    assert stops.kelly_fraction_estimated(0.60, pay) > 0.0


def test_kelly_estimado_degenerado():
    assert stops.kelly_fraction_estimated(0.9, stops.Payoffs(0.0, 2.0)) == 0.0
    assert stops.kelly_fraction_estimated(0.9, stops.Payoffs(2.0, 0.0)) == 0.0
    assert stops.kelly_fraction_estimated(float("nan"), stops.Payoffs(2.0, 2.0)) == 0.0


def test_kelly_estimado_respeta_tope_y_fraccion():
    pay = stops.Payoffs(avg_win=5.0, avg_loss=1.0)
    assert stops.kelly_fraction_estimated(0.99, pay, kelly_frac=1.0, max_f=0.07) == 0.07


def test_estimate_payoffs_solo_train_y_neto_de_costo():
    train = pd.Series([4.0, 2.0, -1.0, -3.0])  # %
    pay = stops.estimate_payoffs(train, cost_pct=0.6)
    assert pay.avg_win == pytest.approx(3.0 - 0.6)
    assert pay.avg_loss == pytest.approx(2.0 + 0.6)
    # sin perdedores no hay pérdida estimable → 0 (Kelly lo trata como degenerado → 0)
    assert stops.estimate_payoffs(pd.Series([1.0, 2.0])).avg_loss == 0.0


def test_edge_negativo_con_costo_da_kelly_cero_con_p_050():
    """Con payoffs simétricos y costo, p=0.50 ya no apuesta (el fijo sí apostaba 3.3%)."""
    pay = stops.estimate_payoffs(pd.Series([3.0, -3.0] * 50), cost_pct=0.6)
    assert stops.kelly_fraction_estimated(0.50, pay) == 0.0
    assert kelly_fraction(0.50) > 0.03


# ── etiqueta de cola ──────────────────────────────────────────────────────────
def test_tail_label_alineada_con_retorno_correcto():
    idx = pd.bdate_range("2024-01-01", periods=5)
    fwd = pd.Series([-0.10, -0.02, 0.05, np.nan, -0.04], index=idx)   # fracciones
    sig_h = pd.Series([0.05, 0.05, 0.05, 0.05, 0.02], index=idx)
    t = stops.tail_label(fwd, sig_h, k=1.0)
    assert t.iloc[0] == 1.0   # -10% < -5%
    assert t.iloc[1] == 0.0   # -2% no
    assert t.iloc[2] == 0.0
    assert np.isnan(t.iloc[3])  # sin retorno futuro → NaN, no 0
    assert t.iloc[4] == 1.0   # -4% < -2%
    # k mayor → cola más exigente
    t15 = stops.tail_label(fwd, sig_h, k=1.5)
    assert t15.iloc[0] == 1.0 and t15.iloc[4] == 1.0 and t15.iloc[1] == 0.0
    t_big = stops.tail_label(fwd, sig_h, k=3.0)   # umbrales −15% y −6%
    assert t_big.iloc[0] == 0.0 and t_big.iloc[4] == 0.0


def test_tail_label_sigma_invalida_es_nan():
    idx = pd.bdate_range("2024-01-01", periods=2)
    t = stops.tail_label(pd.Series([-0.5, -0.5], index=idx),
                         pd.Series([np.nan, 0.0], index=idx), k=1.0)
    assert t.isna().all()


# ── utilidades de evaluación ──────────────────────────────────────────────────
def test_max_drawdown_compuesto():
    r = pd.Series([0.10, -0.20, 0.05])
    # 1.10 → 0.88 (dd −20%) → 0.924
    assert stops.max_drawdown(r) == pytest.approx(-0.20)
    assert stops.max_drawdown(pd.Series([0.01, 0.02])) == 0.0


def test_bootstrap_pareado_reproducible_y_signo():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0.01, 0.02, 100))
    b = a - 0.005
    r1 = stops.paired_block_bootstrap(a, b, lambda x: float(x.mean()), seed=3)
    r2 = stops.paired_block_bootstrap(a, b, lambda x: float(x.mean()), seed=3)
    assert r1 == r2
    assert r1["diff"] == pytest.approx(0.005)
    assert r1["ci_lo"] > 0  # diferencia constante → IC por encima de 0


# ── golden: risk_engine con defaults no cambia ────────────────────────────────
def test_golden_risk_engine_defaults():
    assert kelly_fraction(0.5) == pytest.approx(0.03313252950047177)
    assert kelly_fraction(0.6) == pytest.approx(0.07650602323290752)
    assert kelly_fraction(0.4) == 0.0
    s = build_signal("X.MX", 100.0, 2.0, 0.6, features={"williams_r": -90})
    assert (s.entry, s.stop, s.target_1, s.target_2) == (100.25, 97.25, 103.25, 106.25)
    assert (s.kelly_f, s.pct_capital, s.shares, s.capital_mxn) == (0.0765, 7.65, 76, 7600.0)
    assert (s.risk_pct, s.rr_ratio, s.edge) == (0.229, 1.0, 1.016)
    assert build_signal("X.MX", 100.0, 2.0, 0.49) is None


def test_build_signal_con_sigma_y_payoffs_opcionales():
    pay = stops.Payoffs(avg_win=2.0, avg_loss=2.0)
    s = build_signal("X.MX", 100.0, 2.0, 0.6, sigma_daily=0.02, payoffs=pay)
    sh = 0.02 * np.sqrt(5)
    assert s.stop == pytest.approx(round(100.25 * (1 - sh), 2))
    assert s.kelly_f == pytest.approx(round(stops.kelly_fraction_estimated(0.6, pay), 4))
    # edge estimado ≤ 0 → sin señal
    assert build_signal("X.MX", 100.0, 2.0, 0.5, payoffs=pay) is None
