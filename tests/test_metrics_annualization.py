"""Tests de la anualización de Sharpe/Sortino/Calmar según periods_per_year."""
from __future__ import annotations

import numpy as np

from src.ml.shared.evaluators.metrics import Metrics


def _rets(n: int = 300, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.001, 0.01, n)


def test_default_252_conserva_formula_diaria():
    r = _rets()
    esperado = np.mean(r) / np.std(r) * np.sqrt(252)
    assert np.isclose(Metrics.sharpe_ratio(r), esperado)
    assert np.isclose(Metrics.sharpe_ratio(r, periods_per_year=252.0), esperado)
    wealth = np.cumprod(1 + r)
    mdd = abs(Metrics.max_drawdown(wealth))
    assert np.isclose(Metrics.calmar_ratio(r, wealth), np.mean(r) * 252 / mdd)


def test_sharpe_y_sortino_escalan_con_raiz_de_ppy():
    r = _rets()
    for fn in (Metrics.sharpe_ratio, Metrics.sortino_ratio):
        assert np.isclose(fn(r, periods_per_year=50.4), fn(r, periods_per_year=252.0) / np.sqrt(5))


def test_calmar_escala_lineal_con_ppy():
    r = _rets()
    w = np.cumprod(1 + r)
    assert np.isclose(Metrics.calmar_ratio(r, w, periods_per_year=50.4),
                      Metrics.calmar_ratio(r, w, periods_per_year=252.0) / 5)


def test_rf_se_convierte_por_periodo():
    r = _rets()
    rf = 0.10
    ppy = 50.4
    rf_p = (1 + rf) ** (1 / ppy) - 1
    esperado = (np.mean(r) - rf_p) / np.std(r) * np.sqrt(ppy)
    assert np.isclose(Metrics.sharpe_ratio(r, rf, periods_per_year=ppy), esperado)


def test_portfolio_metrics_propaga_ppy():
    r = _rets()
    a = Metrics.portfolio_metrics_from_returns(r, periods_per_year=252.0)
    b = Metrics.portfolio_metrics_from_returns(r, periods_per_year=50.4)
    assert np.isclose(b["sharpe_ratio"], a["sharpe_ratio"] / np.sqrt(5))
    assert np.isclose(b["calmar_ratio"], a["calmar_ratio"] / 5)


def test_dual_metrics_respeta_ppy():
    r = _rets()
    a = Metrics.calculate_dual_metrics(r, r - 0.001, periods_per_year=252.0)
    b = Metrics.calculate_dual_metrics(r, r - 0.001, periods_per_year=50.4)
    assert np.isclose(b["sharpe_gross"], a["sharpe_gross"] / np.sqrt(5))
    assert np.isclose(b["sharpe_net"], a["sharpe_net"] / np.sqrt(5))
    assert np.isclose(b["calmar_ratio"], a["calmar_ratio"] / 5)
