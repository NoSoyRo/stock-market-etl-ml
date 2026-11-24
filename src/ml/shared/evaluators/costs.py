"""Rotación (turnover) y costos de transacción proporcionales a lo que cambia.

Convención: costo del periodo = ``cost_per_side × Σ|Δw|`` = ``2 × cost_per_side × turnover``,
con ``turnover = 0.5·Σ|Δw|``. Reemplazar todo el portafolio da turnover 1.0 (costo de
ida y vuelta completo, 0.6% con 0.30% por lado). El primer periodo parte de pesos 0, así
que solo paga la entrada (compras): turnover 0.5.

Simplificación: sin deriva de pesos. Los pesos objetivo de cada periodo se comparan con
los objetivo del anterior, aunque entre rebalanceos los precios los hayan movido.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Rotación por periodo: ``0.5·Σ|Δw|`` (filas = periodos, columnas = tickers).

    Los NaN cuentan como peso 0 y el peso previo al primer periodo es 0.
    """
    w = weights.fillna(0.0)
    delta = w.diff()
    delta.iloc[0] = w.iloc[0]  # primer periodo: compra desde cero
    return 0.5 * delta.abs().sum(axis=1)


def net_returns(gross: pd.Series, turnover: pd.Series, cost_per_side: float) -> pd.Series:
    """Retorno neto: ``gross − 2·cost_per_side·turnover`` (fracciones)."""
    return gross - 2.0 * cost_per_side * turnover


def breakeven_cost_per_side(gross: pd.Series, turnover: pd.Series) -> float:
    """Costo por lado con el que el neto promedio es 0: ``mean(gross)/(2·mean(turnover))``."""
    t = float(np.mean(turnover))
    if t <= 0:
        return float("inf")
    return float(np.mean(gross)) / (2.0 * t)


def cost_sensitivity(
    gross: pd.Series,
    turnover: pd.Series,
    costs: tuple[float, ...] = (0.001, 0.0015, 0.003, 0.005),
    periods_per_year: float = 50.4,
) -> pd.DataFrame:
    """Neto medio, Sharpe anualizado y retorno total para varios costos por lado."""
    rows = []
    for c in costs:
        net = net_returns(gross, turnover, c)
        sd = float(net.std(ddof=1)) if len(net) > 1 else 0.0
        sharpe = float(net.mean()) / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0
        rows.append(
            {
                "cost_per_side": c,
                "mean_net": float(net.mean()),
                "sharpe_net": sharpe,
                "total_return_net": float((1.0 + net).prod() - 1.0),
            }
        )
    return pd.DataFrame(rows)
