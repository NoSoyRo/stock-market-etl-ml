"""Tests de rotación y costos por turnover."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.shared.evaluators.costs import (
    breakeven_cost_per_side,
    cost_sensitivity,
    net_returns,
    turnover,
)
from src.ml.shared.evaluators.metrics import Metrics


def _w(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).fillna(0.0)


def test_primer_periodo_solo_paga_entrada():
    t = turnover(_w([{"A": 0.5, "B": 0.5}]))
    assert t.iloc[0] == pytest.approx(0.5)


def test_sin_cambios_rotacion_cero():
    t = turnover(_w([{"A": 0.5, "B": 0.5}] * 3))
    assert t.iloc[1:].tolist() == [0.0, 0.0]


def test_reemplazo_total_es_uno_y_coincide_con_fijo():
    t = turnover(_w([{"A": 0.5, "B": 0.5}, {"C": 0.5, "D": 0.5}]))
    assert t.iloc[1] == pytest.approx(1.0)
    gross = pd.Series([0.01, 0.02])
    net = net_returns(gross, t, 0.003)
    assert net.iloc[1] == pytest.approx(Metrics.net_return(0.02, 0.0025, 0.0005))


def test_reemplazo_parcial():
    # sale B, entra C (la mitad del portafolio): rotación 0.5
    t = turnover(_w([{"A": 0.5, "B": 0.5}, {"A": 0.5, "C": 0.5}]))
    assert t.iloc[1] == pytest.approx(0.5)


def test_breakeven_deja_neto_medio_en_cero():
    gross = pd.Series([0.01, 0.03, -0.005, 0.02])
    t = pd.Series([0.5, 1.0, 0.4, 0.6])
    c = breakeven_cost_per_side(gross, t)
    assert net_returns(gross, t, c).mean() == pytest.approx(0.0, abs=1e-12)


def test_breakeven_sin_rotacion_es_infinito():
    assert breakeven_cost_per_side(pd.Series([0.01]), pd.Series([0.0])) == float("inf")


def test_cost_sensitivity_monotona():
    gross = pd.Series(np.linspace(-0.01, 0.03, 20))
    t = pd.Series(np.full(20, 0.6))
    df = cost_sensitivity(gross, t)
    assert len(df) == 4
    assert df["mean_net"].is_monotonic_decreasing
