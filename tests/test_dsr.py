from __future__ import annotations

import pytest
from scipy import stats

from src.ml.purged_cv import (deflated_sharpe_ratio, expected_max_sharpe,
                              probabilistic_sharpe_ratio)


def test_sr0_100_ensayos_sigma1_es_2_53():
    # Máximo esperado de 100 normales estándar ≈ 2.53 (aprox. del paper)
    assert expected_max_sharpe(100, 1.0) == pytest.approx(2.532, abs=0.01)


def test_psr_valor_a_mano():
    # SR=0.1, T=101, normal: z = 0.1*10/sqrt(1+0.5*0.01)
    z = 1.0 / (1.005 ** 0.5)
    assert probabilistic_sharpe_ratio(0.1, 101) == pytest.approx(stats.norm.cdf(z))


def test_n1_es_psr_contra_cero_no_cdf_del_sharpe():
    d = deflated_sharpe_ratio(0.1, 1, 101)
    assert d == pytest.approx(probabilistic_sharpe_ratio(0.1, 101, 0.0))
    assert d != pytest.approx(float(stats.norm.cdf(0.1)))


def test_propiedades():
    f = lambda sr=0.15, n=20, t=100: deflated_sharpe_ratio(sr, n, t)
    assert f(sr=0.2) > f(sr=0.1)
    assert f(t=200) > f(t=100)
    assert f(n=50) < f(n=5)
    assert f(n=5) < probabilistic_sharpe_ratio(0.15, 100)


def test_sr_std_escala_el_umbral():
    assert deflated_sharpe_ratio(0.15, 20, 100, sr_std=0.2) < deflated_sharpe_ratio(0.15, 20, 100, sr_std=0.05)
