"""Tests del desempate de `p_up` con `p_rank` en `scripts/backtest_topk.py` (Handoff 02)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.backtest_topk as bt


def _tied_panel() -> pd.DataFrame:
    """Panel sintético de 12 tickers en una semana donde todos empatan en p_up=0.60,
    salvo que tienen distinto p_rank y distinto fwd_ret (T0=0% .. T11=11%)."""
    dates = pd.bdate_range("2025-01-06", periods=5)
    rows = []
    for d in dates:
        for i in range(12):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{i:02d}",
                    "p_up": 0.60,
                    "p_rel": 0.60,
                    # p_rank invierte el orden: T11 y T10 son los mejores según el score crudo
                    "p_rank": 0.60 + (i - 5.5) * 1e-4,
                    "fwd_ret": float(i),  # T11 rinde 11%, T10 rinde 10%, T00 rinde 0%
                    "y": 1 if i >= 6 else 0,
                }
            )
    return pd.DataFrame(rows).set_index("date")


def test_topk_elige_por_p_rank_cuando_p_up_empata(monkeypatch):
    monkeypatch.setattr(bt, "_naftrac_return", lambda d0, h: 0.0)
    panel = _tied_panel()
    res = bt.run_topk_backtest(panel, top_k=2, min_p_up=0.0, tiebreak=True)
    # Con p_rank, el top-2 debe ser T11 (11%) y T10 (10%) → retorno bruto medio = 10.5%
    assert res["strategy"]["avg_period_gross_pct"] == pytest.approx(10.5)


def test_topk_invariante_al_orden_de_las_filas(monkeypatch):
    monkeypatch.setattr(bt, "_naftrac_return", lambda d0, h: 0.0)
    panel = _tied_panel()
    base = bt.run_topk_backtest(panel, top_k=5, min_p_up=0.0, tiebreak=True)

    rng = np.random.default_rng(123)
    for _ in range(10):
        perm = rng.permutation(len(panel))
        shuffled = panel.iloc[perm].copy()
        out = bt.run_topk_backtest(shuffled, top_k=5, min_p_up=0.0, tiebreak=True)
        assert out["period_returns"]["gross"] == pytest.approx(base["period_returns"]["gross"])
        assert out["period_returns"]["net"] == pytest.approx(base["period_returns"]["net"])
        assert out["turnover_series"] == pytest.approx(base["turnover_series"])


def test_min_p_up_filtra_por_p_up_no_por_p_rank(monkeypatch):
    monkeypatch.setattr(bt, "_naftrac_return", lambda d0, h: 0.0)
    panel = _tied_panel()
    # T11 tiene el p_rank más alto (0.99), pero su p_up calibrado (0.50) está bajo min_p_up=0.56
    panel.loc[panel["ticker"] == "T11", "p_up"] = 0.50
    panel.loc[panel["ticker"] == "T11", "p_rank"] = 0.99

    res_on = bt.run_topk_backtest(panel, top_k=2, min_p_up=0.56, tiebreak=True)
    res_off = bt.run_topk_backtest(panel, top_k=2, min_p_up=0.56, tiebreak=False)
    # T11 queda excluido por min_p_up; entran T10 (10%) y T09 (9%) → media bruta = 9.5%
    assert res_on["strategy"]["avg_period_gross_pct"] == pytest.approx(9.5)
    # El AUC usa p_up (no p_rank), así que coincide con y sin tiebreak
    assert res_on["strategy"]["auc"] == pytest.approx(res_off["strategy"]["auc"])


def test_tiebreak_false_reproduce_orden_anterior(monkeypatch):
    monkeypatch.setattr(bt, "_naftrac_return", lambda d0, h: 0.0)
    panel = _tied_panel()
    sin_col = bt.run_topk_backtest(panel.drop(columns=["p_rank"]), top_k=2, min_p_up=0.0)
    con_flag_off = bt.run_topk_backtest(panel, top_k=2, min_p_up=0.0, tiebreak=False)
    assert con_flag_off["period_returns"]["gross"] == pytest.approx(sin_col["period_returns"]["gross"])
    assert con_flag_off["strategy"]["avg_period_gross_pct"] != pytest.approx(10.5)
