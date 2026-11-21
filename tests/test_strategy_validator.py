"""Pruebas del agente de validacion de estrategias."""

import copy
import unittest

from src.agent.strategy_validator import StrategyValidator


class TestStrategyValidator(unittest.TestCase):

    def setUp(self):
        self.agente = StrategyValidator(max_drawdown_limit=0.30)

        self.datos = {
            "universe": "bmv",
            "strategy": {
                "sharpe_net": 1.5,
                "total_return_net": 0.25,
                "max_drawdown": -0.15,
                "n_periods": 100,
                "auc": 0.75,
            },
            "vs_benchmark": {
                "excess_return_net": 0.10,
            },
            "config": {
                "commission_pct": 0.0025,
                "slippage_pct": 0.0005,
            },
            "model": {
                "train_start": "2020-01-01",
                "train_end": "2022-12-31",
                "test_start": "2023-01-01",
                "test_end": "2024-12-31",
            },
        }

    def test_estrategia_aprobada(self):
        resultado = self.agente.act(self.datos)
        self.assertTrue(resultado["aprobada"])
        self.assertEqual(resultado["resumen"]["cumple"], 8)

    def test_sharpe_negativo(self):
        datos = copy.deepcopy(self.datos)
        datos["strategy"]["sharpe_net"] = -1.5

        resultado = self.agente.act(datos)

        self.assertFalse(resultado["aprobada"])
        self.assertEqual(
            resultado["preguntas"][0]["estado"],
            "no_cumple",
        )

    def test_retorno_negativo(self):
        datos = copy.deepcopy(self.datos)
        datos["strategy"]["total_return_net"] = -0.20

        resultado = self.agente.act(datos)

        self.assertEqual(
            resultado["preguntas"][1]["estado"],
            "no_cumple",
        )

    def test_drawdown_excesivo(self):
        datos = copy.deepcopy(self.datos)
        datos["strategy"]["max_drawdown"] = -0.50

        resultado = self.agente.act(datos)

        self.assertEqual(
            resultado["preguntas"][2]["estado"],
            "no_cumple",
        )

    def test_costos_ausentes(self):
        datos = copy.deepcopy(self.datos)
        datos["config"] = {}

        resultado = self.agente.act(datos)

        self.assertEqual(
            resultado["preguntas"][3]["estado"],
            "pendiente",
        )

    def test_benchmark_inferior(self):
        datos = copy.deepcopy(self.datos)
        datos["vs_benchmark"]["excess_return_net"] = -0.10

        resultado = self.agente.act(datos)

        self.assertEqual(
            resultado["preguntas"][4]["estado"],
            "no_cumple",
        )

    def test_fechas_incorrectas(self):
        datos = copy.deepcopy(self.datos)
        datos["model"]["test_start"] = "2022-01-01"

        resultado = self.agente.act(datos)

        self.assertEqual(
            resultado["preguntas"][5]["estado"],
            "no_cumple",
        )

    def test_metrica_faltante(self):
        datos = copy.deepcopy(self.datos)
        datos["strategy"].pop("auc")

        resultado = self.agente.act(datos)

        self.assertEqual(
            resultado["preguntas"][7]["estado"],
            "pendiente",
        )


if __name__ == "__main__":
    unittest.main()
