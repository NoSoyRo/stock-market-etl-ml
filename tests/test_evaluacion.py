"""Pruebas del producto de datos Evaluacion/Riesgo."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_evaluacion import (
    exportar_evaluacion,
    validar_proporcion,
)


class TestEvaluacion(unittest.TestCase):

    def setUp(self):
        self.datos = {
            "universe": ["AAPL", "MSFT"],
            "generated_at": "2026-10-28T08:00:00",
            "strategy": {
                "sharpe_gross": 0.5,
                "sharpe_net": -1.2,
                "max_drawdown": -0.3,
                "calmar_ratio": -0.5,
                "win_rate_net": 0.45,
                "total_return_gross": 0.1,
                "total_return_net": -0.15,
                "auc": 0.62,
                "n_periods": 20,
            },
            "config": {
                "horizon": 5,
                "top_k": 5,
                "score_mode": "auc",
                "weight_mode": "equal",
                "commission_pct": 0.001,
                "slippage_pct": 0.001,
                "round_trip_cost": 0.004,
            },
            "model": {
                "train_start": "2024-01-01",
                "train_end": "2024-12-31",
                "test_start": "2025-01-01",
                "test_end": "2025-12-31",
                "n_train": 250,
                "n_test": 100,
            },
        }

    def exportar(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "backtest.json"
            ruta.write_text(
                json.dumps(self.datos),
                encoding="utf-8",
            )
            return exportar_evaluacion(ruta)

    def test_producto_correcto(self):
        resultado = self.exportar()
        self.assertEqual(
            resultado["product"], "evaluacion_riesgo"
        )

    def test_configuracion(self):
        resultado = self.exportar()
        ejecucion = resultado["backtest_runs"]

        self.assertEqual(ejecucion["top_k"], 5)
        self.assertEqual(ejecucion["n_periods"], 20)

    def test_metricas(self):
        resultado = self.exportar()
        metricas = resultado["risk_metrics"]

        self.assertEqual(metricas["sharpe_net"], -1.2)
        self.assertEqual(metricas["max_drawdown"], -0.3)
        self.assertEqual(metricas["auc"], 0.62)

    def test_universe_serializado(self):
        resultado = self.exportar()
        universo = json.loads(resultado["backtest_runs"]["universe"])
        self.assertEqual(universo, ["AAPL", "MSFT"])

    def test_dsr_pendiente(self):
        resultado = self.exportar()
        self.assertIsNone(
            resultado["risk_metrics"]["deflated_sharpe"]
        )

    def test_auc_invalido(self):
        self.datos["strategy"]["auc"] = 1.5

        with self.assertRaises(ValueError):
            self.exportar()

    def test_win_rate_invalido(self):
        self.datos["strategy"]["win_rate_net"] = -0.1

        with self.assertRaises(ValueError):
            self.exportar()

    def test_top_k_invalido(self):
        self.datos["config"]["top_k"] = 0

        with self.assertRaises(ValueError):
            self.exportar()

    def test_periodos_invalidos(self):
        self.datos["strategy"]["n_periods"] = -1

        with self.assertRaises(ValueError):
            self.exportar()

    def test_proporcion_valida(self):
        self.assertEqual(
            validar_proporcion(0.75, "auc"), 0.75
        )

    def test_nan_invalido(self):
        with self.assertRaises(ValueError):
            validar_proporcion(float("nan"), "auc")


if __name__ == "__main__":
    unittest.main()
