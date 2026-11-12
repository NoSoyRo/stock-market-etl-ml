"""Pruebas unitarias del comparador de estrategias."""

import unittest
from copy import deepcopy

from src.agent.strategy_comparator import StrategyComparator


class TestStrategyComparator(unittest.TestCase):

    def setUp(self):
        self.comparador = StrategyComparator()

        self.datos = {
            "universe": ["TEST"],
            "strategy": {
                "sharpe_net": 1.5,
                "total_return_net": 0.20,
                "max_drawdown": -0.15,
                "n_periods": 100,
                "auc": 0.75,
            },
            "vs_benchmark": {
                "excess_return_net": 0.08,
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

    def test_multiples_estrategias(self):
        resultado = self.comparador.evaluate({
            "A": deepcopy(self.datos),
            "B": deepcopy(self.datos),
        })

        self.assertEqual(resultado["total_estrategias"], 2)
        self.assertEqual(len(resultado["evaluaciones"]), 2)

    def test_aprobacion(self):
        aprobada = deepcopy(self.datos)
        rechazada = deepcopy(self.datos)
        rechazada["strategy"]["sharpe_net"] = -1.5

        resultado = self.comparador.evaluate({
            "A": rechazada,
            "B": aprobada,
        })

        self.assertEqual(resultado["aprobadas"], 1)
        self.assertFalse(resultado["evaluaciones"]["A"]["aprobada"])
        self.assertTrue(resultado["evaluaciones"]["B"]["aprobada"])
        self.assertEqual(
            resultado["comparacion"][0]["estrategia"], "B"
        )

    def test_coleccion_vacia(self):
        with self.assertRaises(ValueError):
            self.comparador.evaluate({})

    def test_datos_invalidos(self):
        with self.assertRaises(ValueError):
            self.comparador.evaluate({"A": None})

    def test_no_modifica_datos(self):
        datos = deepcopy(self.datos)
        original = deepcopy(datos)

        self.comparador.evaluate({"A": datos})

        self.assertEqual(datos, original)


if __name__ == "__main__":
    unittest.main()
