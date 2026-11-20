"""M5 · Reglas de portafolio que rotan menos: histéresis y rebalanceo lento.

Trabaja con matrices anchas (filas = periodos de 5 días hábiles, columnas = tickers).
Un score NaN significa "no elegible ese día". Los pesos devueltos son los **objetivo**
de cada periodo (sin deriva), listos para ``costs.turnover``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIOD_DAYS = 5


def rebalance_step(rebalance_every: int, period_days: int = PERIOD_DAYS) -> int:
    """Días hábiles → nº de periodos entre rebalanceos: ``round(días/5)``, mínimo 1 (21 → 4)."""
    return max(1, round(rebalance_every / period_days))


def select_with_hysteresis(
    scores_by_date: pd.DataFrame,
    k: int,
    exit_rank: int,
    rebalance_every: int,
) -> pd.DataFrame:
    """Pesos iguales (1/k) con histéresis.

    En cada fecha de rebalanceo: se conservan las posiciones previas cuyo puesto
    (1 = mejor score) sea ≤ ``exit_rank``; se completa hasta ``k`` con los mejores
    puestos no tenidos. Entre rebalanceos los pesos no cambian. ``exit_rank=k`` equivale
    a reseleccionar de cero. Empates: para salir cuenta el mejor puesto empatado (no se
    expulsa por empate); para entrar manda el orden de columnas (determinista).
    """
    if k < 1 or exit_rank < k:
        raise ValueError("se requiere k >= 1 y exit_rank >= k")
    step = rebalance_step(rebalance_every)
    cols = list(scores_by_date.columns)
    w = pd.DataFrame(0.0, index=scores_by_date.index, columns=cols)
    held: list[str] = []
    prev = np.zeros(len(cols))
    for i, (_, row) in enumerate(scores_by_date.iterrows()):
        if i % step == 0:
            elig = row.dropna()
            if len(elig):
                rank_min = elig.rank(ascending=False, method="min")
                rank_first = elig.rank(ascending=False, method="first")
                keep = [t for t in held if t in elig.index and rank_min[t] <= exit_rank]
                keep = keep[:k]
                rest = rank_first.drop(index=keep).sort_values().index.tolist()
                held = keep + rest[: max(0, k - len(keep))]
                prev = np.zeros(len(cols))
                if held:
                    prev[[cols.index(t) for t in held]] = 1.0 / len(held)
        w.iloc[i] = prev
    return w


def equal_weight_weights(scores_by_date: pd.DataFrame, rebalance_every: int) -> pd.DataFrame:
    """Baseline: universo elegible equiponderado, rebalanceado con la misma frecuencia."""
    step = rebalance_step(rebalance_every)
    elig = scores_by_date.notna().astype(float)
    w = pd.DataFrame(0.0, index=scores_by_date.index, columns=scores_by_date.columns)
    prev = np.zeros(scores_by_date.shape[1])
    for i in range(len(w)):
        if i % step == 0:
            n = elig.iloc[i].sum()
            prev = (elig.iloc[i] / n).to_numpy() if n else np.zeros_like(prev)
        w.iloc[i] = prev
    return w


def held_returns(weights: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """Retorno bruto por periodo: ``Σ w·ret`` (ret NaN cuenta como 0)."""
    r = ret.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    return (weights * r).sum(axis=1)
