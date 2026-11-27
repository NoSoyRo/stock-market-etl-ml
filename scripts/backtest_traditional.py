"""Ejecuta las carteras tradicionales sobre datos de la BMV."""

import json
from pathlib import Path
from datetime import datetime

import numpy as np

from src.data.store import get

from scripts.backtest_topk import _load_naftrac

from src.agent.traditional_portfolio import (
    backtest_portfolio
)


def main():

    folder = Path("data/estrategias_corregidas")

    universe = json.loads(
        (folder / "universo_tradicional.json")
        .read_text(encoding="utf-8")
    )

    tickers = universe["eligible_tickers"]

    start = universe["evaluation_start"]
    end = universe["evaluation_end"]

    capital = float(universe["capital"])

    calendar = _load_naftrac().index

    markets = {}

    print("CARGANDO DATOS DE MERCADO")

    for ticker in tickers:

        market = get(ticker)

        if market is None or market.empty:
            raise RuntimeError(
                f"Sin datos para {ticker}"
            )

        markets[ticker] = market

    print("Emisoras cargadas:", len(markets))
    print("Capital inicial:", capital)
    print("Capital por emisora:", capital / len(markets))
    print("Periodo:", start, "->", end)

    for strategy in ["sma", "rsi"]:

        print("\n" + "=" * 65)
        print("CARTERA", strategy.upper())
        print("=" * 65)

        result = backtest_portfolio(
            markets=markets,
            calendar=calendar,
            start=start,
            end=end,
            strategy=strategy,
            capital=capital,
            commission_pct=0.0025,
            slippage_pct=0.0005
        )

        result["generated_at"] = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )

        result["data_sources"] = {
            "market": "Yahoo Finance",
            "storage": "data/market",
            "calendar": "NAFTRACISHRS.MX"
        }

        result["config"]["parameters"] = (
            {"fast": 50, "slow": 200}
            if strategy == "sma"
            else {
                "period": 14,
                "oversold": 30,
                "overbought": 70
            }
        )

        assert result["config"]["n_assets"] == len(tickers)

        assert result["evaluation_period"]["start"] == start
        assert result["evaluation_period"]["end"] == end

        assert np.isclose(
            result["curves"]["net"][0],
            capital
        )

        assert np.isclose(
            result["strategy"]["final_equity_net"],
            sum(
                asset["final_equity_net"]
                for asset in result["asset_metrics"]
            )
        )

        print("\nMETRICAS CONSOLIDADAS")

        print(json.dumps(
            result["strategy"],
            indent=2
        ))

        print("\nPERIODO DE EVALUACION")

        print(json.dumps(
            result["evaluation_period"],
            indent=2
        ))

        assets = sorted(
            result["asset_metrics"],
            key=lambda a: a["total_return_net"],
            reverse=True
        )

        print("\nCINCO MEJORES EMISORAS")

        for asset in assets[:5]:
            print(
                asset["ticker"],
                f'{asset["total_return_net"]:.2%}'
            )

        print("\nCINCO PEORES EMISORAS")

        for asset in assets[-5:]:
            print(
                asset["ticker"],
                f'{asset["total_return_net"]:.2%}'
            )

        positive = sum(
            asset["total_return_net"] > 0
            for asset in assets
        )

        negative = sum(
            asset["total_return_net"] < 0
            for asset in assets
        )

        neutral = len(assets) - positive - negative

        print("\nBALANCE POR EMISORA")
        print("Positivas:", positive)
        print("Negativas:", negative)
        print("Neutrales:", neutral)

        archivo = folder / f"{strategy}_portfolio.json"

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

    print("\nCARTERAS TRADICIONALES COMPLETADAS")


if __name__ == "__main__":
    main()
