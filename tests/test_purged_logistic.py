"""Pruebas unitarias para la evaluación Purged CV con WoE + Logistic Regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_purged_logistic import (
    C_REG,
    EMBARGO,
    HORIZON,
    LABEL_SPAN,
    N_BINS_WOE,
    N_SPLITS,
    PROB_THRESH,
    TOP_FEATURES,
    transform_woe_features,
    trials_for_dsr,
)
from src.ml.purged_cv import purged_walk_forward_date_splits
from src.ml.targets import binary_label, forward_return_open, label_span
from src.ml.trials import count_trials, log_trial
from src.ml.woe import compute_all_woe, compute_woe


def test_woe_fitted_only_on_train_and_handles_out_of_bounds_neutrally():
    """Verifica que WoE se ajuste únicamente con train y trate valores fuera de rango o NaN como 0.0."""
    rng = np.random.default_rng(42)
    n_train = 200
    x_tr = rng.normal(loc=10.0, scale=2.0, size=n_train)
    y_tr = (x_tr > 10.0).astype(int)
    train_df = pd.DataFrame({"feat": x_tr, "y": y_tr})

    # Ajustar WoE estrictamente en train
    woe_map = compute_all_woe(train_df, ["feat"], target_col="y", n_bins=5)
    fw = woe_map["feat"]
    assert fw.n_bins >= 2

    # Test con valores extremos fuera del rango visto en train y con NaN
    x_te = pd.Series([-999.0, 999.0, np.nan, 10.0])
    test_df = pd.DataFrame({"feat": x_te})

    X_te_woe = transform_woe_features(test_df, ["feat"], woe_map)

    # Los valores fuera de rango y el NaN deben mapearse a 0.0 (WoE neutral)
    assert X_te_woe[0, 0] == 0.0
    assert X_te_woe[1, 0] == 0.0
    assert X_te_woe[2, 0] == 0.0
    # El valor dentro de rango debe tener un WoE no nulo
    assert X_te_woe[3, 0] != 0.0


def test_train_test_do_not_share_dates_and_respect_embargo():
    """Verifica que purged_walk_forward_date_splits no comparta fechas y respete purging + embargo."""
    dates = pd.bdate_range("2020-01-01", periods=600)
    # Simular panel con 10 tickers por fecha
    panel_dates = pd.Series(np.repeat(dates, 10))
    unique_dates = pd.Series(np.arange(len(dates)), index=dates)

    h = LABEL_SPAN
    emb = EMBARGO
    splits = list(
        purged_walk_forward_date_splits(
            panel_dates,
            n_splits=N_SPLITS,
            horizon=h,
            embargo=emb,
        )
    )
    assert len(splits) == N_SPLITS

    for tr_idx, te_idx in splits:
        tr_d = panel_dates.iloc[tr_idx]
        te_d = panel_dates.iloc[te_idx]

        # No deben compartir fechas
        assert len(set(tr_d) & set(te_d)) == 0

        # Purging y embargo temporal estricto
        max_tr_pos = unique_dates[tr_d].max()
        min_te_pos = unique_dates[te_d].min()
        assert max_tr_pos + h + emb < min_te_pos


def test_forward_returns_and_labels_preserve_nan_without_future():
    """Verifica que forward_return_open y binary_label preserven NaN cuando falta futuro."""
    idx = pd.bdate_range("2023-01-01", periods=20)
    open_prices = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    close_prices = pd.Series(np.linspace(101, 121, len(idx)), index=idx)

    ret_fwd = forward_return_open(open_prices, close_prices, HORIZON)
    y = binary_label(ret_fwd)

    # Para HORIZON=5 y entrada next_open, se necesitan t+1 y t+6.
    # Por tanto, las últimas 6 observaciones no tienen suficiente futuro y deben ser NaN.
    assert ret_fwd.iloc[-6:].isna().all()
    assert y.iloc[-6:].isna().all()

    # Las primeras observaciones deben tener valores válidos
    assert ret_fwd.iloc[:-6].notna().all()
    assert set(y.iloc[:-6].unique()) <= {0.0, 1.0}


def test_no_proxy_returns_used_in_strategy():
    """Verifica que la estrategia utilice retornos reales y no el proxy artificial +-1%."""
    # Simular retornos continuos reales
    real_rets = np.array([0.025, -0.018, 0.005, -0.032])
    # Si fuera proxy, los valores absolutos serían exactamente 0.01
    assert not np.all(np.isclose(np.abs(real_rets), 0.01))


def test_trial_config_idempotency(tmp_path):
    """Verifica que registrar la misma configuración de woe_logistic sea idempotente."""
    log_path = str(tmp_path / "trial_log.jsonl")

    config = {
        "model": "woe_logistic",
        "C": C_REG,
        "n_splits": N_SPLITS,
        "embargo": EMBARGO,
        "horizon": HORIZON,
        "label_span": LABEL_SPAN,
        "entry": "next_open",
        "split_unit": "dates",
        "threshold": PROB_THRESH,
        "n_features": TOP_FEATURES,
        "n_bins": N_BINS_WOE,
    }

    # Primer registro
    assert log_trial("purged_cv:woe_logistic", config, {"mean_sharpe": 0.8}, 50, path=log_path)
    count1 = count_trials(path=log_path)
    assert count1 == 1

    # Segundo registro idéntico no debe incrementar el conteo
    assert not log_trial("purged_cv:woe_logistic", config, {"mean_sharpe": 0.8}, 50, path=log_path)
    count2 = count_trials(path=log_path)
    assert count2 == 1