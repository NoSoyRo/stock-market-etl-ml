"""Agente de validacion de estrategias sin inteligencia artificial."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from src.agent.base import Agent


class StrategyValidator(Agent):
    name = "strategy_validator"
    description = "Valida estrategias mediante preguntas y reglas verificables."

    def __init__(self, max_drawdown_limit: float = 0.30):
        if not 0 < max_drawdown_limit < 1:
            raise ValueError("max_drawdown_limit debe estar entre 0 y 1")
        self.max_drawdown_limit = max_drawdown_limit

    def observe(self, **params: Any) -> dict:
        return {
            "preguntas": [
                "¿El Sharpe neto es positivo?",
                "¿El retorno neto es positivo?",
                "¿La perdida maxima esta dentro del limite?",
                "¿Se consideran comisiones y deslizamiento?",
                "¿La estrategia supera a NAFTRAC?",
                "¿Existe un periodo de evaluacion fuera del entrenamiento?",
                "¿Se evaluaron periodos suficientes?",
                "¿Se cuenta con una metrica AUC valida?",
            ],
            "limites": {
                "max_drawdown": self.max_drawdown_limit,
                "min_periodos": 20,
            },
        }

    @staticmethod
    def _numero(valor: Any) -> float | None:
        if isinstance(valor, bool):
            return None
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            return None
        return numero if math.isfinite(numero) else None

    @staticmethod
    def _resultado(
        pregunta: str,
        cumple: bool | None,
        evidencia: Any,
        explicacion: str,
    ) -> dict:
        estado = (
            "pendiente" if cumple is None
            else "cumple" if cumple
            else "no_cumple"
        )
        return {
            "pregunta": pregunta,
            "estado": estado,
            "evidencia": evidencia,
            "explicacion": explicacion,
        }

    def act(self, observation: dict | None = None) -> dict:
        if not isinstance(observation, dict):
            raise ValueError("Se requiere un diccionario con resultados del backtest")

        estrategia = observation.get("strategy") or {}
        comparacion = observation.get("vs_benchmark") or {}
        config = observation.get("config") or {}
        modelo = observation.get("model") or {}

        contexto = observation.get("evaluation_context") or {}
        tipo = observation.get("strategy_type")

        tradicional = tipo == "rule_based"

        resultados = []

        def agregar(pregunta, valor, condicion, explicacion):
            numero = self._numero(valor)
            resultados.append(self._resultado(
                pregunta,
                None if numero is None else condicion(numero),
                valor,
                explicacion,
            ))

        agregar(
            "¿El Sharpe neto es positivo?",
            estrategia.get("sharpe_net"),
            lambda x: x > 0,
            "El Sharpe neto debe ser mayor que cero.",
        )

        agregar(
            "¿El retorno neto es positivo?",
            estrategia.get("total_return_net"),
            lambda x: x > 0,
            "El retorno despues de costos debe ser positivo.",
        )

        agregar(
            "¿La perdida maxima esta dentro del limite?",
            estrategia.get("max_drawdown"),
            lambda x: abs(x) <= self.max_drawdown_limit,
            f"El drawdown absoluto no debe superar {self.max_drawdown_limit:.0%}.",
        )

        comision = self._numero(config.get("commission_pct"))
        deslizamiento = self._numero(config.get("slippage_pct"))
        costos_disponibles = comision is not None and deslizamiento is not None
        resultados.append(self._resultado(
            "¿Se consideran comisiones y deslizamiento?",
            (comision >= 0 and deslizamiento >= 0 and
             (comision + deslizamiento) > 0)
            if costos_disponibles else None,
            {
                "commission_pct": config.get("commission_pct"),
                "slippage_pct": config.get("slippage_pct"),
            },
            "Se verifica que existan costos de transaccion no negativos y no nulos.",
        ))

        agregar(
            "¿La estrategia supera a NAFTRAC?",
            comparacion.get("excess_return_net"),
            lambda x: x > 0,
            "El exceso de retorno neto frente al benchmark debe ser positivo.",
        )

        inicio_entrenamiento = modelo.get("train_start")
        fin_entrenamiento = modelo.get("train_end")
        inicio_prueba = modelo.get("test_start")
        fin_prueba = modelo.get("test_end")

        fechas = [
            inicio_entrenamiento, fin_entrenamiento,
            inicio_prueba, fin_prueba,
        ]
        fechas_disponibles = all(
            isinstance(fecha, str) and fecha.strip()
            for fecha in fechas
        )

        separacion_temporal = None

        if fechas_disponibles:
            try:
                inicio, fin, prueba_inicio, prueba_fin = [
                    date.fromisoformat(fecha)
                    for fecha in fechas
                ]

                separacion_temporal = (
                    inicio <= fin < prueba_inicio <= prueba_fin
                )

                ultima_salida = modelo.get("train_last_exit")

                if ultima_salida is not None:
                    salida = date.fromisoformat(ultima_salida)

                    separacion_temporal = (
                        separacion_temporal
                        and fin <= salida < prueba_inicio
                    )

            except (TypeError, ValueError):
                separacion_temporal = False

        resultados.append(self._resultado(
            "¿Existe un periodo de evaluacion fuera del entrenamiento?",
            separacion_temporal,
            {
                "train_start": inicio_entrenamiento,
                "train_end": fin_entrenamiento,
                "test_start": inicio_prueba,
                "test_end": fin_prueba,
            },
            "Se comprueba la separacion cronologica declarada entre entrenamiento y prueba.",
        ))

        agregar(
            "¿Se evaluaron periodos suficientes?",
            estrategia.get("n_periods"),
            lambda x: x >= 20 and x.is_integer(),
            "Se utiliza un minimo configurable posteriormente de 20 periodos.",
        )

        auc = estrategia.get("auc")
        if auc is None:
            auc = modelo.get("auc")

        agregar(
            "¿Se cuenta con una metrica AUC valida?",
            auc,
            lambda x: 0.5 < x <= 1,
            "El AUC debe ser superior a 0.5 y no superar uno.",
        )

        if tradicional:
            for indice in (5, 7):
                resultados[indice]["estado"] = "no_aplica"
                resultados[indice]["explicacion"] = (
                    "La estrategia utiliza reglas deterministas "
                    "y no requiere entrenamiento supervisado."
                )

        if contexto.get("no_trades") is True:
            resultados[0]["estado"] = "no_cumple"
            resultados[0]["explicacion"] = (
                "No hubo operaciones. El Sharpe cero es una "
                "convencion y no demuestra rendimiento ajustado al riesgo."
            )

        resumen = {
            "cumple": sum(r["estado"] == "cumple" for r in resultados),
            "no_cumple": sum(r["estado"] == "no_cumple" for r in resultados),
            "pendiente": sum(r["estado"] == "pendiente" for r in resultados),
            "no_aplica": sum(r["estado"] == "no_aplica" for r in resultados),
        }

        aplicables = len(resultados) - resumen["no_aplica"]

        resumen["aplicables"] = aplicables

        resumen["proporcion_cumplimiento"] = (
            resumen["cumple"] / aplicables
            if aplicables else 0.0
        )

        return {
            "agente": self.name,
            "estrategia": observation.get(
                "strategy_name",
                observation.get("universe", "desconocida")
            ),
            "tipo_estrategia": tipo or "no_especificado",
            "sin_operaciones": contexto.get("no_trades"),
            "exploratoria": contexto.get("exploratory", False),
            "preguntas": resultados,
            "resumen": resumen,
            "aprobada": (
                aplicables > 0
                and resumen["no_cumple"] == 0
                and resumen["pendiente"] == 0
                and contexto.get("no_trades") is not True
                and not contexto.get("exploratory", False)
            ),
        }
