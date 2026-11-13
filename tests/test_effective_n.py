"""Pruebas del N efectivo (razón de participación) y la estabilidad por bloques."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.shared.evaluators.effective_n import (
    block_stability, pc1_share, participation_ratio, rolling_effective_n,
)


def _wide(k=5, n=1500, rho=0.0, seed=0):
    rng = np.random.default_rng(seed)
    comun = rng.standard_normal((n, 1))
    x = np.sqrt(rho) * comun + np.sqrt(1 - rho) * rng.standard_normal((n, k))
    return pd.DataFrame(x, index=pd.bdate_range("2010-01-01", periods=n), columns=list("ABCDEFGHIJ")[:k])


def test_independientes_dan_n_cercano_a_k():
    assert 4.5 < participation_ratio(_wide(5, rho=0.0)) <= 5.0


def test_casi_identicas_dan_n_cercano_a_1():
    assert participation_ratio(_wide(5, rho=0.99)) < 1.2


def test_pc1_share():
    assert pc1_share(_wide(5, rho=0.99)) > 0.9
    assert pc1_share(_wide(5, rho=0.0)) < 0.3


def test_rolling_devuelve_un_valor_por_bloque():
    r = rolling_effective_n(_wide(5, n=1500), window="3Y")
    assert len(r) >= 1 and (r["n_eff"] <= 5.0 + 1e-9).all() and {"n_eff", "pc1_share"} <= set(r.columns)


def test_ignora_columnas_sin_datos():
    w = _wide(5)
    w["Z"] = np.nan
    assert participation_ratio(w) == participation_ratio(w.drop(columns="Z"))


def test_block_stability_detecta_feature_con_senal_y_ruido():
    rng = np.random.default_rng(1)
    fechas = pd.bdate_range("2010-01-01", "2018-12-31")
    filas = []
    for t in ["a", "b", "c", "d", "e", "f"]:
        sen = rng.standard_normal(len(fechas))
        filas.append(pd.DataFrame({"ticker": t, "sen": sen, "ruido": rng.standard_normal(len(fechas)),
                                   "ret": 0.5 * sen + rng.standard_normal(len(fechas))}, index=fechas))
    panel = pd.concat(filas)
    out = block_stability(panel, ["sen", "ruido"], "ret", block_years=3, min_names=5)
    assert set(out.columns) >= {"feature", "block", "ic_mean", "t", "n"}
    s = out[out.feature == "sen"]
    assert (s.ic_mean > 0.1).all() and len(s) == 3
    assert (out[out.feature == "ruido"].t.abs() < 3).all()
