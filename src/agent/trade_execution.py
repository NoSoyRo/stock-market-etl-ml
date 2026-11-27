"""Simulacion de operaciones sobre precios historicos."""

import numpy as np
import pandas as pd

from src.data.market_quality import prepare_market_data


def execute_trade(
    market: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    entry_date,
    exit_date,
    commission_pct: float = 0.0025,
    slippage_pct: float = 0.0005
) -> dict:

    entry_date = pd.Timestamp(entry_date)
    exit_date = pd.Timestamp(exit_date)

    if entry_date > exit_date:
        raise ValueError("Fechas de operacion invalidas")

    if commission_pct < 0 or slippage_pct < 0:
        raise ValueError("Costos invalidos")

    data = prepare_market_data(
        market, calendar=calendar
    )

    result = {
        "planned_entry": str(entry_date.date()),
        "planned_exit": str(exit_date.date()),
        "actual_entry": None,
        "actual_exit": None,
        "entry_price": None,
        "exit_price": None,
        "gross_return": None,
        "net_return": None,
        "status": None,
        "delayed_exit": False
    }

    if (
        entry_date not in data.index
        or not bool(data.loc[entry_date, "tradable"])
    ):
        result["status"] = "cancelled_entry"
        result["gross_return"] = 0.0
        result["net_return"] = 0.0
        return result

    entry_price = float(
        data.loc[entry_date, "Open"]
    )

    result["actual_entry"] = str(entry_date.date())
    result["entry_price"] = entry_price

    available = data.loc[exit_date:]
    available = available.loc[available["tradable"]]

    if available.empty:
        result["status"] = "open_position"
        return result

    actual_exit = available.index[0]
    exit_price = float(
        available.iloc[0]["Close"]
    )

    gross = exit_price / entry_price - 1

    cost = commission_pct + slippage_pct

    net = gross - 2 * cost

    result.update({
        "actual_exit": str(actual_exit.date()),
        "exit_price": exit_price,
        "gross_return": float(gross),
        "net_return": float(net),
        "status": "closed",
        "delayed_exit": bool(actual_exit > exit_date)
    })

    return result
