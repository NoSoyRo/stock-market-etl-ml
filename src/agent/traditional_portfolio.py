"""Consolidacion de estrategias tradicionales."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agent.traditional_backtest import backtest_asset
from src.ml.shared.evaluators.metrics import Metrics


def backtest_portfolio(
    markets: dict,
    calendar: pd.DatetimeIndex,
    start,
    end,
    strategy: str,
    capital: float = 100000,
    commission_pct: float = 0.0025,
    slippage_pct: float = 0.0005
) -> dict:

    if not markets:
        raise ValueError("Universo vacio")

    if capital <= 0:
        raise ValueError("Capital invalido")

    tickers = sorted(markets)
    allocation = capital / len(tickers)

    results = {}
    operations = []
    open_positions = []
    asset_metrics = []

    dates = None
    portfolio_gross = None
    portfolio_net = None

    for ticker in tickers:

        print(
            f"Ejecutando {strategy.upper()}: {ticker}",
            flush=True
        )

        result = backtest_asset(
            market=markets[ticker],
            calendar=calendar,
            start=start,
            end=end,
            strategy=strategy,
            capital=allocation,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct
        )

        results[ticker] = result

        gross = np.asarray(
            result["curves"]["gross"],
            dtype=float
        )

        net = np.asarray(
            result["curves"]["net"],
            dtype=float
        )

        if dates is None:

            dates = result["dates"]

            portfolio_gross = np.zeros_like(gross)
            portfolio_net = np.zeros_like(net)

        if result["dates"] != dates:
            raise RuntimeError(
                f"Calendario inconsistente: {ticker}"
            )

        portfolio_gross += gross
        portfolio_net += net

        for trade in result["operations"]:

            operations.append({
                "ticker": ticker,
                **trade
            })

        if result["open_position"] is not None:

            open_positions.append({
                "ticker": ticker,
                **result["open_position"]
            })

        asset_metrics.append({
            "ticker": ticker,
            "initial_capital": allocation,
            **result["strategy"],
            "closed_trades": len(
                result["operations"]
            ),
            "active_days": result[
                "evaluation_period"
            ]["active_days"]
        })

    if not np.isclose(
        portfolio_gross[0], capital
    ):
        raise RuntimeError(
            "Capital bruto inicial inconsistente"
        )

    if not np.isclose(
        portfolio_net[0], capital
    ):
        raise RuntimeError(
            "Capital neto inicial inconsistente"
        )

    returns_gross = (
        portfolio_gross[1:]
        / portfolio_gross[:-1] - 1
    )

    returns_net = (
        portfolio_net[1:]
        / portfolio_net[:-1] - 1
    )

    metrics = Metrics.calculate_dual_metrics(
        returns_gross,
        returns_net,
        periods_per_year=252
    )

    metrics["auc"] = None

    metrics["final_equity_gross"] = float(
        portfolio_gross[-1]
    )

    metrics["final_equity_net"] = float(
        portfolio_net[-1]
    )

    metrics["n_periods"] = len(dates)

    periods = [
        r["evaluation_period"]
        for r in results.values()
    ]

    operations.sort(
        key=lambda op: (
            op["entry_date"],
            op["ticker"]
        )
    )

    return {
        "strategy_name": strategy,

        "universe": "bmv",

        "strategy": metrics,

        "config": {
            "capital": capital,
            "capital_per_asset": allocation,
            "commission_pct": commission_pct,
            "slippage_pct": slippage_pct,
            "round_trip_cost": (
                2 * (commission_pct + slippage_pct)
            ),
            "allocation": "equal_initial_weight",
            "rebalance": False,
            "n_assets": len(tickers),
            "tickers": tickers
        },

        "evaluation_period": {
            "start": dates[0],
            "end": dates[-1],
            "trading_sessions": len(dates),
            "periods_per_year": 252,

            "closed_trades": len(operations),

            "open_positions": len(
                open_positions
            ),

            "active_asset_days": sum(
                p["active_days"]
                for p in periods
            ),

            "blocked_entries": sum(
                p["blocked_entries"]
                for p in periods
            ),

            "blocked_exits": sum(
                p["blocked_exits"]
                for p in periods
            ),

            "stale_valuation_asset_days": sum(
                p["stale_valuation_days"]
                for p in periods
            )
        },

        "dates": dates,

        "curves": {
            "gross": portfolio_gross.tolist(),
            "net": portfolio_net.tolist()
        },

        "daily_returns": {
            "gross": returns_gross.tolist(),
            "net": returns_net.tolist()
        },

        "operations": operations,

        "open_positions": open_positions,

        "asset_metrics": asset_metrics
    }
