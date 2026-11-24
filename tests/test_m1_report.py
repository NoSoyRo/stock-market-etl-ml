"""Tests ligeros de las funciones puras de scripts/run_m1_report.py (datos sintéticos)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_m1_report import (
    ols,
    rebalance_panel,
    render_md,
    sharpe,
    vol_matched_random,
    yearly_table,
)


def _panel(n_dates=40, n_tickers=12, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fechas = pd.bdate_range("2024-01-01", periods=n_dates)
    filas = []
    for d in fechas:
        for i in range(n_tickers):
            filas.append({"ticker": f"T{i}", "fwd_ret": rng.normal(0, 3), "p_up": rng.random(),
                          "ret_5d": rng.normal(0, 3), "atr_pct": 1 + i * 0.3}, )
    return pd.DataFrame(filas, index=np.repeat(fechas, n_tickers))


def test_ols_recupera_alfa_y_beta():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 500)
    y = 0.5 + 2.0 * x + rng.normal(0, 0.1, 500)
    r = ols(y, x)
    assert abs(r["alpha"] - 0.5) < 0.05 and abs(r["betas"][0] - 2.0) < 0.05
    assert r["r2"] > 0.99 and r["n"] == 500


def test_ols_sin_regresores_da_media_y_t():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    r = ols(y, np.zeros((4, 0)))
    assert abs(r["alpha"] - 2.5) < 1e-12 and r["betas"] == []


def test_sharpe_anualiza_con_raiz_ppy():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    assert abs(sharpe(r, 50.4) - r.mean() / r.std(ddof=1) * np.sqrt(50.4)) < 1e-12
    assert np.isnan(sharpe([0.01, 0.01]))


def test_rebalance_panel_una_fila_por_ticker_y_fraccion():
    rp = rebalance_panel(_panel(), horizon=5, min_names=10)
    assert (rp.groupby(level=0)["ticker"].nunique() == 12).all()
    assert abs(rp["ret"].iloc[0] - rp["fwd_ret"].iloc[0] / 100) < 1e-12
    assert rp.index.is_monotonic_increasing


def test_vol_matched_acerca_la_vol_del_canasto():
    rp = rebalance_panel(_panel(), horizon=5, min_names=10)
    vm = vol_matched_random(rp.assign(is_pick=rp["p_up"] >= rp.groupby(level=0)["p_up"].transform(lambda s: s.nlargest(5).iloc[-1])), "is_pick", 5, n_draws=300, n_keep=20, seed=0)
    assert vm.shape == (rp.index.nunique(), 20)
    assert np.isfinite(vm.to_numpy()).all()


def test_yearly_table_compone_y_resta():
    idx = pd.to_datetime(["2023-01-05", "2023-06-01", "2024-02-01"])
    t = yearly_table(pd.Series([0.1, 0.1, 0.05], idx), pd.Series([0.0, 0.1, 0.05], idx))
    a23 = next(r for r in t if r["year"] == 2023)
    assert abs(a23["ew"] - 0.21) < 1e-12 and abs(a23["diff"] - 0.11) < 1e-12


def test_render_md_corto_y_con_numeros_del_dict():
    z = {"mean": 0.003, "sharpe": 0.7, "total": 0.5, "n": 10}
    ic = {"mean": 0.01, "t": 1.0, "n": 10, "ci_lo": -0.01, "ci_hi": 0.03}
    blk = {"ic": ic, "auc_mean": 0.51}
    cb = {"avg_period_net_pct": -0.2, "sharpe_net": -0.5, "total_return_net": -0.3,
          "turnover_mean": 0.8, "breakeven_cost_per_side": 0.002}
    rep = {
        "fast": True,
        "config": {"seed": 42, "cost_per_side": 0.003},
        "datos": {"modelo_p_up": {"test_start": "a", "test_end": "b", "auc": 0.51}, "rebalanceos": 10, "n_tickers": 12},
        "ic_por_fecha": {"p_up": blk, "p_rel": blk},
        "baselines": {
            "top5_modelo": z, "equiponderado": z, "cinco_peores_ret5d": z, "naftrac": z,
            "top5_aleatorio": {**z, "n_draws": 5, "sharpe_mean": 0.6, "model_percentile_sharpe": 60.0},
            "ic_neg_ret5d": ic,
        },
        "costos": {"fixed": cb, "turnover": cb},
        "atribucion": {"atr_ratio": 1.2, "vol_anual_top5": 0.3, "vol_anual_ew": 0.2, "beta_top5_vs_ew": 1.1,
                       "alfa_anual_vs_ew": 0.01, "reg_sobre_ew": {"alpha_t": 0.5},
                       "vol_matched": {"sharpe_mean": 0.6, "model_percentile_sharpe": 55.0}},
        "supervivencia": {"diff_media_anual": 0.02},
        "ensayos": {"modelos_entrenados": 2, "corridas_backtest": 2, "configuraciones": 1},
    }
    md = render_md(rep)
    assert len(md.splitlines()) <= 60 and "+0.7" in md and "MODO FAST" in md
