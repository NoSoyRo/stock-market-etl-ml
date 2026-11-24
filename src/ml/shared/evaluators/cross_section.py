"""Evaluación de señales "por fecha" (corte transversal) y baselines.

El panel es un DataFrame con índice de fechas (repetidas, una fila por ticker)
y columnas de score y retorno. Todas las funciones son puras y vectorizadas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _limpiar(panel: pd.DataFrame, cols: list[str], min_names: int) -> pd.DataFrame:
    """Devuelve las columnas pedidas + `_d` (fecha), sin NaN y solo fechas con >= min_names."""
    df = panel[cols].copy()
    df["_d"] = panel.index
    df = df.dropna()
    n = df.groupby("_d")["_d"].transform("size")
    return df[n >= min_names]


def ic_by_date(
    panel: pd.DataFrame,
    score_col: str,
    ret_col: str,
    *,
    method: str = "spearman",
    min_names: int = 10,
) -> pd.Series:
    """IC por fecha: correlación (Spearman por defecto) entre score y retorno posterior.

    Mide si, cada día, las acciones mejor rankeadas rinden más; fechas con < min_names se omiten.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("method debe ser 'spearman' o 'pearson'")
    df = _limpiar(panel, [score_col, ret_col], min_names)
    g = df.groupby("_d")
    if method == "spearman":
        a = g[score_col].rank()
        b = g[ret_col].rank()
    else:
        a, b = df[score_col], df[ret_col]
    aux = pd.DataFrame({"_d": df["_d"], "a": a, "b": b})
    aux["a"] -= aux.groupby("_d")["a"].transform("mean")
    aux["b"] -= aux.groupby("_d")["b"].transform("mean")
    aux["ab"], aux["aa"], aux["bb"] = aux["a"] * aux["b"], aux["a"] ** 2, aux["b"] ** 2
    s = aux.groupby("_d")[["ab", "aa", "bb"]].sum()
    den = np.sqrt(s["aa"] * s["bb"])
    ic = (s["ab"] / den).where(den > 0)
    ic.index.name = None
    return ic.dropna().sort_index().rename("ic")


def auc_by_date(
    panel: pd.DataFrame,
    y_col: str,
    score_col: str,
    *,
    min_names: int = 10,
) -> pd.Series:
    """AUC por fecha (Mann-Whitney con rangos): prob. de que un positivo tenga más score que un negativo.

    Fechas sin positivos o sin negativos se omiten.
    """
    df = _limpiar(panel, [y_col, score_col], min_names)
    df["_r"] = df.groupby("_d")[score_col].rank()
    df["_y"] = (df[y_col] > 0).astype(float)
    df["_ry"] = df["_r"] * df["_y"]
    s = df.groupby("_d").agg(n1=("_y", "sum"), n=("_y", "size"), sr=("_ry", "sum"))
    n0 = s["n"] - s["n1"]
    auc = (s["sr"] - s["n1"] * (s["n1"] + 1) / 2) / (s["n1"] * n0)
    auc = auc.where((s["n1"] > 0) & (n0 > 0))
    auc.index.name = None
    return auc.dropna().sort_index().rename("auc")


def ic_summary(
    ic: pd.Series,
    *,
    step: int = 1,
    block: int = 4,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Resumen del IC: media, std, t, n, error estándar e IC 95% por bootstrap de bloques móviles.

    Con horizonte h los IC de días vecinos se solapan; `step=h` toma una fecha de cada h
    para tener una muestra (casi) independiente sobre la que corre el bootstrap.
    """
    x = ic.dropna().to_numpy(dtype=float)[:: max(int(step), 1)]
    n = len(x)
    nan = float("nan")
    if n < 2:
        return {"mean": float(x.mean()) if n else nan, "std": nan, "t": nan, "n": n,
                "se": nan, "ci_lo": nan, "ci_hi": nan}
    mean, std = float(x.mean()), float(x.std(ddof=1))
    t = mean / (std / np.sqrt(n)) if std > 0 else nan
    b = min(max(int(block), 1), n)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": mean, "std": std, "t": float(t), "n": n,
            "se": float(means.std(ddof=1)), "ci_lo": float(lo), "ci_hi": float(hi)}


def topk_returns(
    panel: pd.DataFrame,
    score_col: str,
    ret_col: str,
    k: int,
    *,
    ascending: bool = False,
    min_names: int = 10,
) -> pd.Series:
    """Retorno equiponderado de los k nombres con mayor score (menor si ascending) en cada fecha.

    Empates se rompen por orden de aparición.
    """
    df = _limpiar(panel, [score_col, ret_col], min_names)
    rk = df.groupby("_d")[score_col].rank(method="first", ascending=ascending)
    out = df[rk <= k].groupby("_d")[ret_col].mean()
    out.index.name = None
    return out.sort_index().rename("topk")


def equal_weight_returns(panel: pd.DataFrame, ret_col: str, *, min_names: int = 10) -> pd.Series:
    """Retorno del universo equiponderado en cada fecha (baseline "comprar todo parejo")."""
    df = _limpiar(panel, [ret_col], min_names)
    out = df.groupby("_d")[ret_col].mean()
    out.index.name = None
    return out.sort_index().rename("ew")


def random_topk_returns(
    panel: pd.DataFrame,
    ret_col: str,
    k: int,
    *,
    n_draws: int = 1000,
    seed: int = 0,
    min_names: int = 10,
) -> pd.DataFrame:
    """Retornos de elegir k nombres al azar en cada fecha: DataFrame fecha × sorteo (reproducible)."""
    df = _limpiar(panel, [ret_col], min_names).sort_values("_d", kind="stable")
    rng = np.random.default_rng(seed)
    fechas, filas = [], []
    for d, v in df.groupby("_d")[ret_col]:
        v = v.to_numpy()
        kk = min(k, len(v))
        idx = rng.random((n_draws, len(v))).argpartition(kk - 1, axis=1)[:, :kk]
        fechas.append(d)
        filas.append(v[idx].mean(axis=1))
    if not filas:
        return pd.DataFrame(columns=range(n_draws), dtype=float)
    return pd.DataFrame(np.vstack(filas), index=pd.Index(fechas), columns=range(n_draws))


def percentile_in(value: float, draws) -> float:
    """Percentil (0–100) de `value` dentro de `draws`; los empates cuentan la mitad."""
    d = np.asarray(draws, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan")
    return float(100.0 * ((d < value).mean() + 0.5 * (d == value).mean()))
