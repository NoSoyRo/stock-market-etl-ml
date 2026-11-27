"""Benchmark diario Buy and Hold NAFTRAC."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest_topk import _load_naftrac
from src.data.market_quality import validate_market_data
from src.ml.shared.evaluators.metrics import Metrics


DESTINATION = Path("data/estrategias_homologadas")


def main():

    start = pd.Timestamp("2023-05-10")
    end = pd.Timestamp("2026-10-30")

    capital = 100000.0

    commission = 0.0025
    slippage = 0.0005

    cost = Metrics.round_trip_cost(
        commission,
        slippage
    )

    market = _load_naftrac().loc[start:end].copy()

    assert len(market) == 876
    assert market.index[0] == start
    assert market.index[-1] == end

    quality = validate_market_data(market)

    # Para ejecutar compra y venta se requiere
    # que ambas sesiones sean negociables.
    assert bool(quality.iloc[0]["tradable"])
    assert bool(quality.iloc[-1]["tradable"])

    entry_price = float(
        market.iloc[0]["Open"]
    )

    exit_price = float(
        market.iloc[-1]["Close"]
    )

    close = market["Close"].astype(float)

    assert np.isfinite(close).all()
    assert (close > 0).all()
    assert entry_price > 0

    ratios = close.to_numpy() / entry_price

    curve_gross = np.concatenate((
        [capital],
        capital * ratios
    ))

    curve_net = np.concatenate((
        [capital],
        capital * (ratios - cost)
    ))

    assert (curve_net > 0).all()

    returns_gross = (
        curve_gross[1:] / curve_gross[:-1] - 1
    )

    returns_net = (
        curve_net[1:] / curve_net[:-1] - 1
    )

    metrics = Metrics.calculate_dual_metrics(
        returns_gross,
        returns_net,
        periods_per_year=252
    )

    metrics["auc"] = None

    metrics["final_equity_gross"] = float(
        curve_gross[-1]
    )

    metrics["final_equity_net"] = float(
        curve_net[-1]
    )

    metrics["n_periods"] = len(market)

    gross_return = exit_price / entry_price - 1
    net_return = gross_return - cost

    assert np.isclose(
        metrics["total_return_gross"],
        gross_return
    )

    assert np.isclose(
        metrics["total_return_net"],
        net_return
    )

    assert np.isclose(
        curve_gross[-1] - curve_net[-1],
        capital * cost
    )

    result = {
        "strategy_name": "buy_hold_naftrac",

        "strategy": metrics,

        "config": {
            "capital": capital,
            "commission_pct": commission,
            "slippage_pct": slippage,
            "round_trip_cost": cost,
            "symbol": "NAFTRACISHRS.MX"
        },

        "evaluation_period": {
            "start": str(start.date()),
            "end": str(end.date()),
            "trading_sessions": len(market),
            "periods_per_year": 252,
            "metric_frequency": "daily",
            "closed_trades": 1,
            "open_positions": 0
        },

        "data_quality": {
            "invalid_ohlcv_sessions": int(
                (~quality["valid_ohlcv"]).sum()
            ),
            "valuation_policy": "positive_finite_close",
            "execution_policy": "tradable_entry_and_exit"
        },

        "data_sources": {
            "market": "Yahoo Finance",
            "symbol": "NAFTRACISHRS.MX"
        },

        "dates": [
            str(date.date())
            for date in market.index
        ],

        "curves": {
            "gross": curve_gross.tolist(),
            "net": curve_net.tolist()
        },

        "daily_returns": {
            "gross": returns_gross.tolist(),
            "net": returns_net.tolist()
        },

        "operations": [
            {
                "ticker": "NAFTRACISHRS.MX",
                "entry_date": str(start.date()),
                "exit_date": str(end.date()),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "status": "closed"
            }
        ]
    }

    DESTINATION.mkdir(
        parents=True,
        exist_ok=True
    )

    archivo = DESTINATION / "buy_hold_naftrac.json"

    archivo.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    print("\nBENCHMARK NAFTRAC")

    print("\nMETRICAS")
    print(json.dumps(metrics, indent=2))

    print("\nOPERACION")
    print(json.dumps(
        result["operations"][0],
        indent=2
    ))

    print("\nCALIDAD DE DATOS")
    print(json.dumps(
        result["data_quality"],
        indent=2
    ))

    print("\nCosto total MXN:", capital * cost)

    print("Consistencia de rendimientos: OK")
    print("Consistencia de costos: OK")
    print("Sesiones evaluadas:", len(market))
    print("Puntos de valoracion:", len(curve_net))

    print("\nGuardado:", archivo)
    print("BENCHMARK COMPLETADO")


if __name__ == "__main__":
    main()
