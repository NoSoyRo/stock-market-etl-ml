"""Motor diario para estrategias tradicionales."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agent.traditional_strategies import (
    sma_strategy,
    rsi_strategy
)

from src.data.market_quality import prepare_market_data

from src.ml.shared.evaluators.metrics import Metrics


def backtest_asset(
    market: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    start,
    end,
    strategy: str,
    capital: float = 100000,
    commission_pct: float = 0.0025,
    slippage_pct: float = 0.0005
) -> dict:

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    cost = commission_pct + slippage_pct

    if capital <= 0:
        raise ValueError("Capital invalido")

    if (
        commission_pct < 0
        or slippage_pct < 0
        or 2 * cost >= 1
    ):
        raise ValueError("Costos invalidos")

    if start > end:
        raise ValueError("Periodo invalido")

    calendar = pd.DatetimeIndex(
        calendar
    ).unique().sort_values()

    calendar = calendar[calendar <= end]

    data = prepare_market_data(
        market,
        calendar=calendar
    )

    # Solo precios validos y negociables para indicadores
    close = data.loc[data["tradable"], "Close"]

    if strategy == "sma":
        signals = sma_strategy(
            close,
            fast=50,
            slow=200
        )
        warmup = 200

    elif strategy == "rsi":
        signals = rsi_strategy(
            close,
            period=14,
            oversold=30,
            overbought=70
        )
        warmup = 15

    else:
        raise ValueError("Estrategia desconocida")

    if len(close.loc[close.index < start]) < warmup:
        raise ValueError(
            "Historial previo insuficiente"
        )

    # La decision del cierre se ejecuta como
    # minimo en la siguiente sesion.
    desired = (
        signals["position"]
        .reindex(calendar)
        .ffill()
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    sessions = calendar[
        (calendar >= start)
        & (calendar <= end)
    ]

    if sessions.empty:
        raise ValueError("Periodo sin sesiones")

    cash_gross = float(capital)
    cash_net = float(capital)

    invested_gross = 0.0
    invested_net = 0.0

    entry_price = None
    entry_date = None
    last_price = None

    position = False

    curve_gross = [float(capital)]
    curve_net = [float(capital)]

    trades = []

    active_days = 0
    blocked_entries = 0
    blocked_exits = 0
    stale_valuation_days = 0

    for date in sessions:

        row = data.loc[date]

        tradable = bool(row["tradable"])
        target = int(desired.loc[date])

        if tradable:

            opening = float(row["Open"])
            closing = float(row["Close"])

            # Comprar al precio de apertura
            if not position and target == 1:

                invested_gross = cash_gross
                invested_net = cash_net

                cash_gross = 0.0
                cash_net = 0.0

                entry_price = opening
                entry_date = date
                position = True

            # Vender al precio de apertura
            elif position and target == 0:

                gross_return = (
                    opening / entry_price - 1
                )

                net_return = (
                    gross_return - 2 * cost
                )

                cash_gross = (
                    invested_gross
                    * (1 + gross_return)
                )

                cash_net = (
                    invested_net
                    * (1 + net_return)
                )

                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(date.date()),
                    "entry_price": entry_price,
                    "exit_price": opening,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "status": "closed"
                })

                position = False

                invested_gross = 0.0
                invested_net = 0.0

            last_price = closing

        else:

            if not position and target == 1:
                blocked_entries += 1

            elif position and target == 0:
                blocked_exits += 1

            if position:
                stale_valuation_days += 1

        if position:

            active_days += 1

            ratio = last_price / entry_price

            equity_gross = invested_gross * ratio

            # Reservar costo de entrada y salida.
            # Misma aproximacion aditiva que Top-K.
            equity_net = (
                invested_net * (ratio - 2 * cost)
            )

        else:

            equity_gross = cash_gross
            equity_net = cash_net

        if (
            not np.isfinite(equity_gross)
            or not np.isfinite(equity_net)
            or equity_gross <= 0
            or equity_net <= 0
        ):
            raise RuntimeError(
                "Valor de cartera invalido"
            )

        curve_gross.append(float(equity_gross))
        curve_net.append(float(equity_net))

    gross = np.asarray(curve_gross)
    net = np.asarray(curve_net)

    returns_gross = gross[1:] / gross[:-1] - 1
    returns_net = net[1:] / net[:-1] - 1

    metrics = Metrics.calculate_dual_metrics(
        returns_gross,
        returns_net,
        periods_per_year=252
    )

    metrics["auc"] = None

    metrics["final_equity_gross"] = float(gross[-1])
    metrics["final_equity_net"] = float(net[-1])

    metrics["n_periods"] = len(sessions)

    return {
        "strategy_name": strategy,
        "strategy": metrics,

        "config": {
            "capital": capital,
            "commission_pct": commission_pct,
            "slippage_pct": slippage_pct,
            "round_trip_cost": 2 * cost
        },

        "evaluation_period": {
            "start": str(sessions[0].date()),
            "end": str(sessions[-1].date()),
            "trading_sessions": len(sessions),
            "periods_per_year": 252,
            "active_days": active_days,
            "blocked_entries": blocked_entries,
            "blocked_exits": blocked_exits,
            "stale_valuation_days": stale_valuation_days
        },

        "curves": {
            "gross": curve_gross,
            "net": curve_net
        },

        "daily_returns": {
            "gross": returns_gross.tolist(),
            "net": returns_net.tolist()
        },

        "dates": [
            str(date.date()) for date in sessions
        ],

        "operations": trades,

        "open_position": (
            {
                "entry_date": str(entry_date.date()),
                "entry_price": entry_price,
                "last_price": last_price,
                "status": "open_position"
            }
            if position else None
        )
    }
