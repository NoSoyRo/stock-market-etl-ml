from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src.ml.purged_cv import deflated_sharpe_ratio, expected_max_sharpe

_spec = importlib.util.spec_from_file_location(
    "run_purged_cv", Path(__file__).resolve().parents[1] / "scripts" / "run_purged_cv.py")
rpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rpc)


def test_dsr_a_mano_desanualiza_y_usa_sr_std_real():
    h = 5
    per = np.array([0.10, 0.14, 0.06, 0.12, 0.08, 0.10])
    annual = per * np.sqrt(252 / h)
    n_trials, T = 20, 300
    sr0 = expected_max_sharpe(n_trials, per.std(ddof=1))
    z = (per.mean() - sr0) * np.sqrt(T - 1) / np.sqrt(1 + 0.5 * per.mean() ** 2)
    esperado = stats.norm.cdf(z)
    assert rpc.dsr_from_fold_sharpes(annual, h, n_trials, T) == pytest.approx(esperado)


def test_dsr_no_depende_de_la_anualizacion():
    per = np.array([0.1, 0.12, 0.08])
    a5 = rpc.dsr_from_fold_sharpes(per * np.sqrt(252 / 5), 5, 10, 200)
    a21 = rpc.dsr_from_fold_sharpes(per * np.sqrt(252 / 21), 21, 10, 200)
    assert a5 == pytest.approx(a21)


def test_psr_valor_de_la_formula_del_paper_con_asimetria():
    # Fórmula Bailey–López de Prado (2014), SR=0.1, T=101, skew=-1, kurt=5:
    # z = 0.1*10 / sqrt(1 + 0.1 + 1*0.01) ; (verificado contra fuentes secundarias)
    z = 1.0 / np.sqrt(1 + 0.1 + (5 - 1) / 4 * 0.01)
    d = deflated_sharpe_ratio(0.1, 1, 101, skewness=-1.0, kurtosis=5.0)
    assert d == pytest.approx(stats.norm.cdf(z))
