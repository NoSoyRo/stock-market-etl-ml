from __future__ import annotations

import numpy as np

from src.ml.ensemble import break_ties


def test_desempata_por_score_crudo_sin_cambiar_orden_entre_mesetas():
    cal = np.array([0.5, 0.5, 0.5, 0.6, 0.4])
    raw = np.array([0.30, 0.10, 0.20, 0.05, 0.9])
    out = break_ties(cal, raw)
    assert len(np.unique(out)) == 5
    assert np.argsort(-out).tolist() == [3, 0, 2, 1, 4]


def test_no_depende_del_orden_de_las_filas():
    rng = np.random.default_rng(0)
    cal = rng.choice([0.45, 0.5, 0.55], 40)
    raw = rng.random(40)
    perm = rng.permutation(40)
    a = break_ties(cal, raw)
    b = break_ties(cal[perm], raw[perm])
    np.testing.assert_allclose(a[perm], b)


def test_sin_empates_no_cambia_el_orden():
    cal = np.array([0.1, 0.3, 0.2]); raw = np.array([0.9, 0.1, 0.5])
    assert np.argsort(break_ties(cal, raw)).tolist() == np.argsort(cal).tolist()


def test_predict_proba_por_defecto_sin_cambios_golden():
    import inspect
    from src.ml.ensemble import EnsembleModel
    src = inspect.getsource(EnsembleModel.predict_proba)
    assert "break_ties" not in src and "_calibrator.predict" in src
