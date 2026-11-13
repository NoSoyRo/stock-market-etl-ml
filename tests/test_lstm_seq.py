"""Tests para LSTMSeqPredictor (KB-013)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.lstm_seq import (
    LSTMSeqPredictor,
    add_target_up,
    build_sliding_sequences,
)
from src.ml.shared.models.lstm_model import TF_AVAILABLE


def _synthetic_panel(n_days: int = 120, n_tickers: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for t in range(n_tickers):
        ticker = f"TK{t}"
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        close = 100 + np.cumsum(rng.normal(0, 1, n_days))
        df = pd.DataFrame(
            {
                "Open": close * 0.99,
                "Close": close,
                "feat_a": rng.normal(0, 1, n_days),
                "feat_b": rng.normal(0, 1, n_days),
                "ticker": ticker,
            },
            index=dates,
        )
        rows.append(df)
    return pd.concat(rows).sort_index()


class TestLSTMSeqInterface:
    def test_imports(self):
        from src.ml.lstm_seq import LSTMSeqPredictor as P
        from src.ml.shared.models.lstm_model import LSTMModel

        assert P is not None
        assert LSTMModel is not None

    def test_predictor_defaults(self):
        pred = LSTMSeqPredictor()
        assert pred.sequence_length == 30
        assert pred.num_layers == 2
        assert pred.hidden_units == 64
        assert pred.is_fitted is False

    def test_add_target_up_single_ticker(self):
        df = _synthetic_panel(n_days=80, n_tickers=1).drop(columns=["ticker"])
        out = add_target_up(df, use_cross_section=False)
        assert "target_up" in out.columns
        assert set(out["target_up"].dropna().unique()).issubset({0, 1})

    def test_build_sliding_sequences(self):
        df = _synthetic_panel()
        df = add_target_up(df, use_cross_section=True)
        X, y, idx, _ = build_sliding_sequences(
            df, ["feat_a", "feat_b"], "target_up", sequence_length=10
        )
        assert X.ndim == 3
        assert X.shape[1] == 10
        assert X.shape[2] == 2
        assert len(y) == len(idx)
        assert len(y) > 0

    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow no instalado")
    def test_fit_and_predict_proba(self):
        df = _synthetic_panel(n_days=200)
        df = add_target_up(df, use_cross_section=True)
        pred = LSTMSeqPredictor(sequence_length=15, epochs=3, hidden_units=16)
        metrics = pred.fit(df, feature_cols=["feat_a", "feat_b"], target_col="target_up")
        assert pred.is_fitted
        assert metrics["n_samples"] > 0
        probs = pred.predict_proba(df, ["feat_a", "feat_b"])
        assert probs.ndim == 1
        # Las primeras `sequence_length` filas de cada ticker no tienen secuencia completa
        # y salen NaN por diseño (para fusionar en el SuperModel): se validan las demás.
        valid = probs[~np.isnan(probs)]
        assert len(valid) > 0
        assert np.all((valid >= 0) & (valid <= 1))

    def test_graceful_fallback_without_tf(self, monkeypatch):
        monkeypatch.setattr("src.ml.lstm_seq.TF_AVAILABLE", False)
        pred = LSTMSeqPredictor()
        df = _synthetic_panel()
        metrics = pred.fit(df, ["feat_a", "feat_b"])
        assert pred.is_fitted is False
        assert metrics.get("error") == "tensorflow_not_available"
        probs = pred.predict_proba(df, ["feat_a", "feat_b"])
        assert len(probs) >= 1
