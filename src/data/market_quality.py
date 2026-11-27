"""Control de calidad de datos historicos del mercado."""

from __future__ import annotations

import numpy as np
import pandas as pd


OHLC = ["Open", "High", "Low", "Close"]
REQUIRED_COLUMNS = OHLC + ["Volume"]


def validate_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica problemas de calidad sin modificar los precios."""

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Columnas faltantes: {sorted(missing)}"
        )

    prices = df[OHLC].apply(
        pd.to_numeric, errors="coerce"
    )

    volume = pd.to_numeric(
        df["Volume"], errors="coerce"
    )

    quality = pd.DataFrame(index=df.index)

    quality["invalid_price"] = (
        (~np.isfinite(prices)).any(axis=1)
        | (prices <= 0).any(axis=1)
    )

    quality["invalid_ohlc"] = (
        (prices["High"] < prices.max(axis=1))
        | (prices["Low"] > prices.min(axis=1))
    )

    quality["invalid_volume"] = (
        (~np.isfinite(volume))
        | (volume < 0)
    )

    quality["zero_volume"] = volume == 0

    quality["duplicate_date"] = (
        df.index.duplicated(keep=False)
    )

    quality["valid_ohlcv"] = ~quality[
        [
            "invalid_price",
            "invalid_ohlc",
            "invalid_volume",
            "duplicate_date"
        ]
    ].any(axis=1)

    quality["tradable"] = (
        quality["valid_ohlcv"]
        & (volume > 0)
    )

    return quality


def quality_summary(
    df: pd.DataFrame,
    calendar=None
) -> dict:
    """Genera un resumen cuantitativo de calidad."""

    quality = validate_market_data(df)

    total = len(df)

    valid = int(quality["valid_ohlcv"].sum())
    tradable = int(quality["tradable"].sum())

    result = {
        "total_records": total,
        "valid_records": valid,
        "tradable_records": tradable,
        "invalid_records": total - valid,
        "invalid_prices": int(
            quality["invalid_price"].sum()
        ),
        "invalid_ohlc": int(
            quality["invalid_ohlc"].sum()
        ),
        "invalid_volume": int(
            quality["invalid_volume"].sum()
        ),
        "zero_volume": int(
            quality["zero_volume"].sum()
        ),
        "duplicate_dates": int(
            quality["duplicate_date"].sum()
        ),
        "quality_pct": round(
            100 * valid / total, 4
        ) if total else 0.0
    }

    if calendar is not None:

        calendar = pd.DatetimeIndex(calendar)

        missing_dates = calendar.difference(df.index)

        result["expected_sessions"] = len(calendar)
        result["missing_sessions"] = len(missing_dates)

        result["missing_dates"] = [
            str(date.date())
            for date in missing_dates
        ]

    return result


def prepare_market_data(
    df: pd.DataFrame,
    calendar=None
) -> pd.DataFrame:
    """Prepara datos sin inventar precios ni eliminar sesiones."""

    if not df.index.is_unique:
        raise ValueError("Existen fechas duplicadas")

    result = df.sort_index().copy()

    if calendar is not None:
        result = result.reindex(
            pd.DatetimeIndex(calendar)
        )

    quality = validate_market_data(result)

    result.loc[
        ~quality["valid_ohlcv"],
        REQUIRED_COLUMNS
    ] = np.nan

    result["tradable"] = quality["tradable"]

    return result
