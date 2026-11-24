"""Cuántos activos independientes hay de verdad, y qué tan estable es cada feature en el tiempo.

`participation_ratio` = (Σλ)² / Σλ² de la matriz de correlación: vale k si los k activos son
independientes y 1 si son copias. Es el "n honesto" para juzgar significancia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.shared.evaluators.cross_section import ic_by_date, ic_summary


def _eigs(returns_wide: pd.DataFrame, min_obs: int = 60) -> np.ndarray:
    """Valores propios de la correlación (columnas sin datos o casi constantes se descartan)."""
    x = returns_wide.dropna(axis=1, thresh=min_obs)
    x = x.loc[:, x.std() > 0]
    if x.shape[1] == 0:
        return np.array([])
    corr = x.corr(min_periods=min_obs).fillna(0.0).to_numpy()
    corr = corr.copy()
    np.fill_diagonal(corr, 1.0)
    return np.clip(np.linalg.eigvalsh(corr), 0.0, None)


def participation_ratio(returns_wide: pd.DataFrame) -> float:
    """N efectivo = (Σλ)² / Σλ² de la matriz de correlación."""
    lam = _eigs(returns_wide)
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.size else float("nan")


def pc1_share(returns_wide: pd.DataFrame) -> float:
    """Fracción de la varianza (estandarizada) que explica el primer componente."""
    lam = _eigs(returns_wide)
    return float(lam.max() / lam.sum()) if lam.size else float("nan")


def rolling_effective_n(returns_wide: pd.DataFrame, window: str = "3Y") -> pd.DataFrame:
    """N efectivo por bloques consecutivos sin traslape de `window` ("3Y", "6M"...).

    Índice = inicio del bloque; columnas `n_eff`, `pc1_share`, `n_assets`, `n_obs`.
    """
    años = int(window[:-1]) * (12 if window[-1].upper() == "Y" else 1)
    off = pd.DateOffset(months=años)
    idx = pd.DatetimeIndex(returns_wide.index)
    filas, ini = [], idx.min()
    while ini <= idx.max():
        fin = ini + off
        bloque = returns_wide[(idx >= ini) & (idx < fin)]
        if len(bloque) >= 60:
            filas.append({"start": ini, "n_eff": participation_ratio(bloque), "pc1_share": pc1_share(bloque),
                          "n_assets": int((bloque.count() >= 60).sum()), "n_obs": len(bloque)})
        ini = fin
    return pd.DataFrame(filas).set_index("start") if filas else pd.DataFrame(
        columns=["n_eff", "pc1_share", "n_assets", "n_obs"])


def block_stability(
    panel: pd.DataFrame, feature_cols: list[str], ret_col: str, *, block_years: int = 3,
    min_names: int = 10, step: int = 1,
) -> pd.DataFrame:
    """IC por fecha medio y t por bloque de `block_years` años, para cada feature.

    Salida larga: feature, block (año inicial), ic_mean, t, n (fechas). `step` se pasa a
    `ic_summary` (usa 5 con horizonte 5d para que el t cuente fechas casi independientes).
    """
    años = pd.DatetimeIndex(panel.index).year
    y0 = int(años.min())
    filas = []
    for f in feature_cols:
        ic = ic_by_date(panel, f, ret_col, min_names=min_names)
        if ic.empty:
            continue
        bloque = y0 + ((ic.index.year - y0) // block_years) * block_years
        for b, g in ic.groupby(bloque):
            if len(g) < 20:
                continue
            s = ic_summary(g, step=step, n_boot=2)  # el bootstrap no se usa aquí: solo media y t
            filas.append({"feature": f, "block": int(b), "ic_mean": s["mean"], "t": s["t"], "n": int(s["n"])})
    return pd.DataFrame(filas, columns=["feature", "block", "ic_mean", "t", "n"])
