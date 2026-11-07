"""
Exportador del producto de datos Evaluacion/Riesgo.

Transforma los resultados de backtesting existentes
en registros compatibles con el esquema eval.
"""

import argparse
import json
import math
from pathlib import Path


def validar_proporcion(valor, nombre):
    """Valida metricas que deben estar entre 0 y 1."""
    if valor is None:
        return None

    valor = float(valor)

    if not math.isfinite(valor) or not 0 <= valor <= 1:
        raise ValueError(f"{nombre} debe estar entre 0 y 1")

    return valor


def exportar_evaluacion(entrada):
    """Lee un backtest y genera registros de evaluacion."""

    with open(entrada, encoding="utf-8") as archivo:
        datos = json.load(archivo)

    estrategia = datos["strategy"]
    configuracion = datos["config"]
    modelo = datos["model"]

    ejecucion = {
        "universe": json.dumps(datos["universe"], ensure_ascii=False),
        "generated_at": datos["generated_at"],
        "horizon": configuracion.get("horizon"),
        "top_k": configuracion.get("top_k"),
        "score_mode": configuracion.get("score_mode"),
        "weight_mode": configuracion.get("weight_mode"),
        "train_start": modelo.get("train_start"),
        "train_end": modelo.get("train_end"),
        "test_start": modelo.get("test_start"),
        "test_end": modelo.get("test_end"),
        "n_train": modelo.get("n_train"),
        "n_test": modelo.get("n_test"),
        "n_periods": estrategia.get("n_periods"),
    }

    metricas = {
        "sharpe_gross": estrategia.get("sharpe_gross"),
        "sharpe_net": estrategia.get("sharpe_net"),
        "max_drawdown": estrategia.get("max_drawdown"),
        "calmar_ratio": estrategia.get("calmar_ratio"),
        "win_rate_net": validar_proporcion(
            estrategia.get("win_rate_net"), "win_rate_net"
        ),
        "total_return_gross": estrategia.get("total_return_gross"),
        "total_return_net": estrategia.get("total_return_net"),
        "auc": validar_proporcion(
            estrategia.get("auc"), "auc"
        ),
        "commission_pct": configuracion.get("commission_pct"),
        "slippage_pct": configuracion.get("slippage_pct"),
        "round_trip_cost": configuracion.get("round_trip_cost"),
        "deflated_sharpe": None,
    }

    if ejecucion["top_k"] is not None:
        if ejecucion["top_k"] <= 0:
            raise ValueError("top_k debe ser mayor que cero")

    if ejecucion["n_periods"] is not None:
        if ejecucion["n_periods"] < 0:
            raise ValueError("n_periods no puede ser negativo")

    return {
        "product": "evaluacion_riesgo",
        "version": "1.0.0",
        "backtest_runs": ejecucion,
        "risk_metrics": metricas,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exportar resultados de Evaluacion/Riesgo"
    )

    parser.add_argument(
        "--input",
        default="data/eval/backtest_topk.json",
        help="Archivo JSON de entrada",
    )

    parser.add_argument(
        "--output",
        default="data/eval/evaluacion_export.json",
        help="Archivo JSON de salida",
    )

    args = parser.parse_args()

    resultado = exportar_evaluacion(args.input)

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)

    with open(salida, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, indent=2, ensure_ascii=False)

    print(f"Exportacion completada: {salida}")
    print("Producto:", resultado["product"])
    print("Sharpe neto:", resultado["risk_metrics"]["sharpe_net"])
    print("Max Drawdown:", resultado["risk_metrics"]["max_drawdown"])


if __name__ == "__main__":
    main()
