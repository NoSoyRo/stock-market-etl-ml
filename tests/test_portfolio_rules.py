"""Tests de M5 · reglas de portafolio que rotan menos (datos sintéticos)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.shared.evaluators.costs import turnover
from src.ml.shared.evaluators.portfolio_rules import (
    equal_weight_weights,
    held_returns,
    rebalance_step,
    select_with_hysteresis,
)


def _scores(n_dates: int = 8, n_names: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-05", periods=n_dates, freq="7D")
    return pd.DataFrame(rng.normal(size=(n_dates, n_names)), index=idx,
                        columns=[f"T{i}" for i in range(n_names)])


def test_ranking_constante_rotacion_cero():
    base = _scores(1)
    s = pd.concat([base] * 6, ignore_index=True)
    s.index = pd.date_range("2024-01-05", periods=6, freq="7D")
    w = select_with_hysteresis(s, k=3, exit_rank=3, rebalance_every=5)
    t = turnover(w)
    assert t.iloc[0] == 0.5          # primera compra desde cero
    assert (t.iloc[1:] == 0).all()   # sin cambios de ranking ⇒ rotación 0


def test_k_constante_y_pesos_suman_uno():
    s = _scores()
    for exit_rank in (3, 4, 6):
        w = select_with_hysteresis(s, k=3, exit_rank=exit_rank, rebalance_every=5)
        assert ((w > 0).sum(axis=1) == 3).all()
        assert np.allclose(w.sum(axis=1), 1.0)


def test_salida_forzada_al_pasar_exit_rank():
    cols = list("ABCDEF")
    idx = pd.date_range("2024-01-05", periods=2, freq="7D")
    s = pd.DataFrame(
        [[6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6]], index=idx, columns=cols, dtype=float
    )
    # t0 elige A,B (k=2). En t1 A es último (rank 6) y B rank 5.
    w = select_with_hysteresis(s, k=2, exit_rank=4, rebalance_every=5)
    assert set(w.columns[w.iloc[1] > 0]) == {"F", "E"}      # ambas salen (rank > 4)
    # con exit_rank amplio se quedan los que ya estaban si siguen dentro
    w2 = select_with_hysteresis(s, k=2, exit_rank=6, rebalance_every=5)
    assert set(w2.columns[w2.iloc[1] > 0]) == {"A", "B"}


def test_histeresis_mantiene_quien_cae_poco():
    cols = list("ABCDE")
    idx = pd.date_range("2024-01-05", periods=2, freq="7D")
    s = pd.DataFrame([[5, 4, 3, 2, 1], [4, 3, 5, 2, 1]], index=idx, columns=cols, dtype=float)
    # t1: ranking C,A,B,D,E. k=2, exit 3: A(rank2) y B(rank3) se quedan ⇒ rotación 0.
    w = select_with_hysteresis(s, k=2, exit_rank=3, rebalance_every=5)
    assert turnover(w).iloc[1] == 0
    # sin histéresis (exit=k) B sale y entra C
    w0 = select_with_hysteresis(s, k=2, exit_rank=2, rebalance_every=5)
    assert turnover(w0).iloc[1] == 0.5


def test_rebalance_every_mantiene_pesos_entre_rebalanceos():
    s = _scores(9)
    w = select_with_hysteresis(s, k=3, exit_rank=3, rebalance_every=10)  # cada 2 periodos
    for i in (1, 3, 5, 7):
        pd.testing.assert_series_equal(w.iloc[i], w.iloc[i - 1], check_names=False)
    assert rebalance_step(5) == 1 and rebalance_step(10) == 2 and rebalance_step(21) == 4


def test_nan_no_elegible_no_entra():
    s = _scores(3, 6)
    s.iloc[1, 0] = np.nan
    s.iloc[:, 1] = np.nan
    w = select_with_hysteresis(s, k=2, exit_rank=3, rebalance_every=5)
    assert (w["T1"] == 0).all()


def test_equal_weight_y_held_returns():
    s = _scores(5, 4)
    ew = equal_weight_weights(s, rebalance_every=5)
    assert np.allclose(ew.sum(axis=1), 1.0) and np.allclose(ew.iloc[0], 0.25)
    ret = pd.DataFrame(0.01, index=s.index, columns=s.columns)
    assert np.allclose(held_returns(ew, ret), 0.01)
