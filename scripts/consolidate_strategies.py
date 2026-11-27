"""Consolida estrategias con un benchmark comun."""

import json
from pathlib import Path

import numpy as np


SOURCE = Path("data/estrategias_corregidas")
DESTINATION = Path("data/estrategias_homologadas")


def load_json(path):

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path, data):

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        ),
        encoding="utf-8"
    )


def main():

    benchmark = load_json(
        DESTINATION / "buy_hold_naftrac.json"
    )

    bm = benchmark["strategy"]

    sources = {
        "topk_2": DESTINATION / "topk_2.json",
        "topk_5": DESTINATION / "topk_5.json",
        "topk_10": DESTINATION / "topk_10.json",

        "topk_5_ranking":
            DESTINATION / "topk_5_ranking.json",

        "sma_50_200":
            SOURCE / "sma_portfolio.json",

        "rsi_14":
            SOURCE / "rsi_portfolio.json"
    }

    catalog = {}
    summary = []

    for name, path in sources.items():

        result = load_json(path)

        metrics = result["strategy"]
        config = result["config"]
        period = result["evaluation_period"]

        # Comprobar condiciones comunes.
        assert result["dates"] == benchmark["dates"]

        assert period["start"] == (
            benchmark["evaluation_period"]["start"]
        )

        assert period["end"] == (
            benchmark["evaluation_period"]["end"]
        )

        assert period["trading_sessions"] == 876
        assert period["periods_per_year"] == 252

        for key in [
            "capital",
            "commission_pct",
            "slippage_pct",
            "round_trip_cost"
        ]:

            assert np.isclose(
                config[key],
                benchmark["config"][key]
            ), f"{name}: configuracion distinta: {key}"

        for key in ["gross", "net"]:

            curve = np.asarray(
                result["curves"][key],
                dtype=float
            )

            returns = np.asarray(
                result["daily_returns"][key],
                dtype=float
            )

            assert len(curve) == 877
            assert len(returns) == 876

            assert np.isfinite(curve).all()
            assert np.isfinite(returns).all()

            assert (curve > 0).all()

            assert np.isclose(
                curve[0],
                config["capital"]
            )

            assert np.allclose(
                curve[1:] / curve[:-1] - 1,
                returns
            )

            assert np.isclose(
                curve[-1],
                metrics[f"final_equity_{key}"]
            )

            assert np.isclose(
                curve[-1] / curve[0] - 1,
                metrics[f"total_return_{key}"]
            )

        result["strategy_name"] = name

        result["benchmark_naftrac"] = {
            "symbol": "NAFTRACISHRS.MX",
            "method": "buy_and_hold",
            "total_return": bm["total_return_net"],
            "total_return_gross": bm["total_return_gross"],
            "total_return_net": bm["total_return_net"],
            "sharpe_ratio": bm["sharpe_net"],
            "sharpe_gross": bm["sharpe_gross"],
            "sharpe_net": bm["sharpe_net"],
            "max_drawdown": bm["max_drawdown"],
            "win_rate": bm["win_rate_net"],
            "final_equity": bm["final_equity_net"],
            "final_equity_gross": bm["final_equity_gross"],
            "final_equity_net": bm["final_equity_net"],
            "n_periods": bm["n_periods"],
            "cost_basis": "net"
        }

        result["vs_benchmark"] = {
            "excess_return_net": (
                metrics["total_return_net"]
                - bm["total_return_net"]
            ),

            "sharpe_delta_net": (
                metrics["sharpe_net"]
                - bm["sharpe_net"]
            ),

            "capital_difference_net": (
                metrics["final_equity_net"]
                - bm["final_equity_net"]
            )
        }

        result["curves"]["naftrac"] = (
            benchmark["curves"]["net"]
        )

        result["curves"]["naftrac_gross"] = (
            benchmark["curves"]["gross"]
        )

        result["daily_returns"]["naftrac"] = (
            benchmark["daily_returns"]["net"]
        )

        result["evaluation_period"].update({
            "metric_frequency": "daily",
            "benchmark_symbol": "NAFTRACISHRS.MX",
            "benchmark_method": "continuous_buy_and_hold",
            "benchmark_cost_basis": "net"
        })

        is_ml = name.startswith("topk")

        result["strategy_type"] = (
            "machine_learning"
            if is_ml
            else "rule_based"
        )

        result["evaluation_context"] = {
            "auc_applicable": is_ml,

            "temporal_split_applicable": is_ml,

            "exploratory": (
                name == "topk_5_ranking"
            ),

            "no_trades": (
                period["closed_trades"] == 0
                and period["open_positions"] == 0
            )
        }

        result.setdefault(
            "homologation", {}
        ).update({
            "benchmark_pending": False,
            "benchmark_cost_basis": "net",
            "common_calendar_verified": True,
            "common_costs_verified": True
        })

        output = DESTINATION / f"{name}.json"

        save_json(output, result)

        catalog[name] = result

        summary.append({
            "name": name,
            "type": result["strategy_type"],

            "return_net": metrics["total_return_net"],

            "sharpe_net": metrics["sharpe_net"],

            "max_drawdown": metrics["max_drawdown"],

            "final_equity_net":
                metrics["final_equity_net"],

            "auc": metrics.get("auc"),

            "closed_trades": period["closed_trades"],

            "open_positions": period["open_positions"],

            "excess_return_net":
                result["vs_benchmark"]["excess_return_net"],

            "file": str(output)
        })

        print(
            f"{name}: calendario, capital, "
            "costos y curvas verificados"
        )

    save_json(
        DESTINATION / "strategies_catalog.json",
        catalog
    )

    summary.sort(
        key=lambda x: x["return_net"],
        reverse=True
    )

    save_json(
        DESTINATION / "strategies_summary.json",
        {
            "evaluation_period":
                benchmark["evaluation_period"],

            "benchmark": benchmark["strategy"],

            "n_strategies": len(summary),

            "strategies": summary
        }
    )

    print("\nRESUMEN CONSOLIDADO\n")

    for row in summary:

        print(
            f'{row["name"]:20s}',
            f'Retorno: {row["return_net"]:8.2%}',
            f'Exceso: {row["excess_return_net"]:8.2%}',
            f'Sharpe: {row["sharpe_net"]:7.3f}',
            f'Capital: ${row["final_equity_net"]:,.2f}'
        )

    print("\nBENCHMARK")
    print("Retorno neto:", f'{bm["total_return_net"]:.2%}')
    print("Capital final:", bm["final_equity_net"])

    print("\nEstrategias consolidadas:", len(catalog))

    print("CONSISTENCIA FINANCIERA: OK")
    print("CONSOLIDACION COMPLETADA")


if __name__ == "__main__":
    main()
