"""Estrategias tradicionales de inversion."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_rsi(
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calcula el RSI mediante suavizado de Wilder."""

    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0), 100
    )

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain == 0), 50
    )

    return rsi


def sma_strategy(
    close: pd.Series,
    fast: int = 50,
    slow: int = 200
) -> pd.DataFrame:
    """Seguimiento de tendencia mediante medias moviles."""

    if not 0 < fast < slow:
        raise ValueError("Periodos SMA invalidos")

    result = pd.DataFrame({"close": close})

    result["sma_fast"] = close.rolling(fast).mean()
    result["sma_slow"] = close.rolling(slow).mean()

    valid = (
        result["sma_fast"].notna()
        & result["sma_slow"].notna()
    )

    result["position"] = np.where(
        valid,
        (result["sma_fast"] > result["sma_slow"]).astype(int),
        0
    )

    result["signal"] = (
        result["position"].diff().fillna(0).astype(int)
    )

    return result


def rsi_strategy(
    close: pd.Series,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70
) -> pd.DataFrame:
    """Reversion a la media mediante RSI."""

    if not 0 < oversold < overbought < 100:
        raise ValueError("Umbrales RSI invalidos")

    result = pd.DataFrame({"close": close})

    result["rsi"] = calculate_rsi(close, period)

    position = 0
    positions = []

    for value in result["rsi"]:

        if pd.notna(value):

            if position == 0 and value < oversold:
                position = 1

            elif position == 1 and value > overbought:
                position = 0

        positions.append(position)

    result["position"] = positions

    result["signal"] = (
        result["position"].diff().fillna(0).astype(int)
    )

    return result
