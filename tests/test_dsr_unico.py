"""Un solo DSR: `purged_cv` y `trial_stats` comparten núcleo y bitácora."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.ml import trial_stats as ts
from src.ml.purged_cv import deflated_sharpe_ratio, expected_max_sharpe, probabilistic_sharpe_ratio
from src.ml.trials import count_trials, log_trial


def test_mismo_dsr_con_los_mismos_datos():
    rng = np.random.default_rng(0)
    r = rng.standard_t(5, 600) * 0.01 + 0.0006          # colas gordas
    t = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    theirs = ts.deflated_sharpe(r, 20)["dsr"]
    mine = deflated_sharpe_ratio(sr, 20, t, skewness=skew, kurtosis=kurt, sr_std=1 / np.sqrt(t))
    assert mine == pytest.approx(theirs, abs=1e-3)       # trial_stats redondea a 3 decimales


def test_purged_cv_delega_en_el_nucleo():
    assert probabilistic_sharpe_ratio(0.1, 200, 0.02, -0.3, 5.0) == pytest.approx(
        ts.probabilistic_sharpe_ratio(0.1, 200, 0.02, -0.3, 5.0))
    assert expected_max_sharpe(30, 0.05) == pytest.approx(ts.sr0_from_std(30, 0.05))
    # la versión anual de trial_stats es el mismo SR0 con sd = 1/√años
    assert ts.expected_max_sharpe(30, 4.0) == pytest.approx(ts.sr0_from_std(30, 0.5))


def test_normalidad_vs_colas_gordas_castiga_mas():
    sr, T = 0.25, 300   # SR por encima del azar esperado (SR0 ~0.09)
    normal = deflated_sharpe_ratio(sr, 10, T)
    gordas = deflated_sharpe_ratio(sr, 10, T, skewness=-1.0, kurtosis=9.0)
    assert gordas < normal


def test_una_sola_bitacora_n_por_familia_y_global(tmp_path):
    p = tmp_path / "trial_log.jsonl"
    log_trial("m9_bakeoff:ridge", {"a": 1}, {"ic": 0.03}, 400, path=str(p))
    log_trial("m9_bakeoff:lgbm", {"a": 2}, {"ic": 0.02}, 400, path=str(p))
    log_trial("m10_target:A", {"a": 1}, {"ic": 0.03}, 400, path=str(p))
    assert not log_trial("m10_target:A", {"a": 1}, {"ic": 0.03}, 400, path=str(p))  # idempotente
    assert ts.n_trials("m9_bakeoff", p) == 2             # N por familia (API de trial_stats)
    assert ts.n_trials("m10_target", p) == 1
    assert count_trials(path=str(p)) == 3                # N global
    # lo que escribe `trial_stats` también entra al conteo global
    ts.log_trial("core_satellite", {"k": 1}, {"sharpe": 0.9}, p)
    assert count_trials(path=str(p)) == 4
