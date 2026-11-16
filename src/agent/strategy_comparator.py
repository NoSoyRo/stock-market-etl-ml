"""Evaluacion y comparacion de multiples estrategias."""

from __future__ import annotations

from typing import Any

from src.agent.strategy_validator import StrategyValidator


class StrategyComparator:
    """Evalua varias estrategias mediante el validador existente."""

    def __init__(self, max_drawdown_limit: float = 0.30):
        self.validator = StrategyValidator(
            max_drawdown_limit=max_drawdown_limit
        )

    def evaluate(self, strategies: dict[str, dict[str, Any]]) -> dict:
        if not isinstance(strategies, dict) or not strategies:
            raise ValueError("Se requiere al menos una estrategia")

        resultados = {}

        for nombre, datos in strategies.items():
            if not isinstance(nombre, str) or not nombre.strip():
                raise ValueError("Cada estrategia necesita un nombre valido")

            if not isinstance(datos, dict):
                raise ValueError(
                    f"La estrategia {nombre} debe contener un objeto JSON"
                )

            evaluacion = self.validator.act(datos)
            evaluacion["nombre"] = nombre
            resultados[nombre] = evaluacion

        comparacion = sorted(
            [
                {
                    "estrategia": nombre,
                    "cumple": resultado["resumen"]["cumple"],
                    "no_cumple": resultado["resumen"]["no_cumple"],
                    "pendiente": resultado["resumen"]["pendiente"],
                    "aprobada": resultado["aprobada"],
                }
                for nombre, resultado in resultados.items()
            ],
            key=lambda item: (
                -int(item["aprobada"]),
                -item["cumple"],
                item["no_cumple"],
                item["pendiente"],
                item["estrategia"],
            ),
        )

        return {
            "total_estrategias": len(resultados),
            "evaluaciones": resultados,
            "comparacion": comparacion,
            "aprobadas": sum(
                resultado["aprobada"]
                for resultado in resultados.values()
            ),
        }
