"""Reconstruccion diaria de los backtests Top-K."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.store import get
from src.data.market_quality import prepare_market_data
from src.ml.shared.evaluators.metrics import Metrics
from scripts.backtest_topk import _load_naftrac


SOURCE = Path("data/estrategias_corregidas")
DESTINATION = Path("data/estrategias_homologadas")


def reconstruct_daily(result, calendar, market_cache):

    config = result["config"]
    period = result["evaluation_period"]

    capital = float(config["capital"])
    cost = float(config["round_trip_cost"])

    start = pd.Timestamp(period["start"])
    end = pd.Timestamp(period["end"])

    dates = calendar[
        (calendar >= start)
        & (calendar <= end)
    ]

    gross = pd.Series(capital, index=dates)
    net = pd.Series(capital, index=dates)

    equity_gross = capital
    equity_net = capital

    previous_exit = None
    stale_days = 0
    closed_trades = 0

    for i, operation in enumerate(result["operations"]):

        entry = pd.Timestamp(operation["entry_date"])
        exit_date = pd.Timestamp(operation["exit_date"])

        if previous_exit is not None:
            assert entry > previous_exit

        assert entry in dates
        assert exit_date in dates
        assert entry <= exit_date

        cycle_dates = dates[
            (dates >= entry)
            & (dates <= exit_date)
        ]

        gross_factor = pd.Series(
            1.0, index=cycle_dates
        )

        net_factor = pd.Series(
            1.0, index=cycle_dates
        )

        trades = operation["trades"]

        total_weight = sum(
            float(t["weight"])
            for t in trades
        )

        assert total_weight <= 1.00000001

        for trade in trades:

            weight = float(trade["weight"])

            if trade["status"] == "cancelled_entry":
                continue

            if trade["status"] != "closed":
                raise RuntimeError(
                    "Operacion sin liquidar"
                )

            ticker = trade["ticker"]

            if ticker not in market_cache:

                market = get(ticker)

                if market is None or market.empty:
                    raise RuntimeError(
                        f"Sin datos para {ticker}"
                    )

                market_cache[ticker] = (
                    prepare_market_data(
                        market,
                        calendar=dates
                    )
                )

            market = market_cache[ticker]

            actual_entry = pd.Timestamp(
                trade["actual_entry"]
            )

            actual_exit = pd.Timestamp(
                trade["actual_exit"]
            )

            assert entry <= actual_entry
            assert actual_entry <= actual_exit
            assert actual_exit <= exit_date

            entry_price = float(
                trade["entry_price"]
            )

            exit_price = float(
                trade["exit_price"]
            )

            assert entry_price > 0
            assert exit_price > 0

            holding = market.loc[
                actual_entry:actual_exit
            ]

            assert bool(
                holding.iloc[0]["tradable"]
            )

            assert bool(
                holding.iloc[-1]["tradable"]
            )

            assert np.isclose(
                holding.iloc[0]["Open"],
                entry_price
            )

            assert np.isclose(
                holding.iloc[-1]["Close"],
                exit_price
            )

            prices = holding["Close"].where(
                holding["tradable"]
            )

            stale_days += int(
                prices.isna().sum()
            )

            # Precio anterior solo para valoracion.
            prices = prices.ffill()

            assert prices.notna().all()

            ratios = prices / entry_price

            gross_factor.loc[ratios.index] += (
                weight * (ratios - 1)
            )

            # Reservar costos completos, igual que
            # el motor tradicional.
            net_factor.loc[ratios.index] += (
                weight * (ratios - 1 - cost)
            )

            # Despues de vender, conservar efectivo.
            after_exit = cycle_dates[
                cycle_dates > actual_exit
            ]

            gross_factor.loc[after_exit] += (
                weight * trade["gross_return"]
            )

            net_factor.loc[after_exit] += (
                weight * trade["net_return"]
            )

            assert np.isclose(
                ratios.iloc[-1] - 1,
                trade["gross_return"]
            )

            assert np.isclose(
                ratios.iloc[-1] - 1 - cost,
                trade["net_return"]
            )

            closed_trades += 1

        # Verificar que cada ciclo conserva
        # los rendimientos originales.
        assert np.isclose(
            gross_factor.iloc[-1] - 1,
            operation["gross_return"],
            atol=1e-10
        )

        assert np.isclose(
            net_factor.iloc[-1] - 1,
            operation["net_return"],
            atol=1e-10
        )

        gross.loc[cycle_dates] = (
            equity_gross * gross_factor
        )

        net.loc[cycle_dates] = (
            equity_net * net_factor
        )

        equity_gross = float(
            gross.loc[exit_date]
        )

        equity_net = float(
            net.loc[exit_date]
        )

        # Comparar contra la curva original.
        assert np.isclose(
            equity_gross,
            result["curves"]["gross"][i + 1]
        )

        assert np.isclose(
            equity_net,
            result["curves"]["net"][i + 1]
        )

        # Mantener el capital durante los dias
        # posteriores hasta el siguiente ciclo.
        gross.loc[gross.index > exit_date] = (
            equity_gross
        )

        net.loc[net.index > exit_date] = (
            equity_net
        )

        previous_exit = exit_date

    curve_gross = np.concatenate((
        [capital],
        gross.to_numpy()
    ))

    curve_net = np.concatenate((
        [capital],
        net.to_numpy()
    ))

    assert np.isfinite(curve_gross).all()
    assert np.isfinite(curve_net).all()

    assert (curve_gross > 0).all()
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

    metrics["auc"] = result["strategy"].get("auc")

    metrics["final_equity_gross"] = float(
        curve_gross[-1]
    )

    metrics["final_equity_net"] = float(
        curve_net[-1]
    )

    metrics["n_periods"] = len(dates)

    assert len(dates) == period["trading_sessions"]

    assert np.isclose(
        metrics["final_equity_net"],
        result["strategy"]["final_equity_net"],
        atol=0.01
    )

    assert np.isclose(
        metrics["final_equity_gross"],
        result["strategy"]["final_equity_gross"],
        atol=0.01
    )

    assert np.isclose(
        metrics["total_return_net"],
        result["strategy"]["total_return_net"]
    )

    output = dict(result)

    # Conservar metricas y curvas anteriores.
    output["original_cycle_metrics"] = (
        result["strategy"]
    )

    output["original_cycle_curves"] = (
        result["curves"]
    )

    output["strategy"] = metrics

    # El benchmark diario se incorporara
    # en el siguiente paso.
    output["original_benchmark_naftrac"] = (
        output.pop("benchmark_naftrac")
    )

    output["original_vs_benchmark"] = (
        output.pop("vs_benchmark")
    )

    output["curves"] = {
        "gross": curve_gross.tolist(),
        "net": curve_net.tolist()
    }

    output["daily_returns"] = {
        "gross": returns_gross.tolist(),
        "net": returns_net.tolist()
    }

    output["dates"] = [
        str(date.date()) for date in dates
    ]

    output["evaluation_period"] = {
        **period,
        "original_periods_per_year": (
            period["periods_per_year"]
        ),
        "periods_per_year": 252,
        "metric_frequency": "daily",
        "closed_trades": closed_trades,
        "open_positions": 0,
        "stale_valuation_asset_days": stale_days
    }

    output["homologation"] = {
        "method": "daily_mark_to_market",
        "cost_policy": "round_trip_cost_reserved",
        "original_returns_preserved": True,
        "benchmark_pending": True
    }

    return output


def main():

    DESTINATION.mkdir(
        parents=True,
        exist_ok=True
    )

    calendar = _load_naftrac().index

    market_cache = {}

    names = [
        "topk_2",
        "topk_5",
        "topk_10",
        "topk_5_ranking"
    ]

    for name in names:

        print("\n" + "=" * 60)
        print("HOMOLOGANDO:", name)
        print("=" * 60)

        original = json.loads(
            (SOURCE / f"{name}.json")
            .read_text(encoding="utf-8")
        )

        result = reconstruct_daily(
            original,
            calendar,
            market_cache
        )

        result["strategy_name"] = name

        print("\nMETRICAS ORIGINALES")

        print(json.dumps(
            original["strategy"],
            indent=2
        ))

        print("\nMETRICAS DIARIAS")

        print(json.dumps(
            result["strategy"],
            indent=2
        ))

        print("\nPERIODO HOMOLOGADO")

        print(json.dumps(
            result["evaluation_period"],
            indent=2
        ))

        print("\nPuntos de la curva:",
              len(result["curves"]["net"]))

        print("Retorno original conservado: OK")
        print("Capital final conservado: OK")
        print("Consistencia por ciclo: OK")

        archivo = DESTINATION / f"{name}.json"

        archivo.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                allow_nan=False
            ),
            encoding="utf-8"
        )

        print("\nGuardado:", archivo)

    print("\nHOMOLOGACION TOP-K COMPLETADA")


if __name__ == "__main__":
    main()
