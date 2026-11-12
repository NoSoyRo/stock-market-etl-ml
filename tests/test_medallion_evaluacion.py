
"""Pruebas del pipeline Medallion de Evaluación/Riesgo."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import medallion_evaluacion as medallion


class TestMedallionEvaluacion(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.base = Path(self.temp.name)
        self.eval_dir = self.base / "eval"
        self.bronze_dir = self.eval_dir / "medallion" / "bronze"

        self.eval_dir.mkdir()
        self.bronze_dir.mkdir(parents=True)

        self.datos = {
            "backtest_topk.json": {
                "strategy": {
                    "sharpe_net": -1.78,
                    "max_drawdown": -0.53,
                    "win_rate_net": 0.44,
                    "total_return_net": -0.46,
                    "auc": 0.50,
                }
            },
            "portfolio_metrics.json": {
                "portfolio_dual": {
                    "sharpe_net": -1.78,
                    "max_drawdown": -0.53,
                    "win_rate_net": 0.44,
                    "total_return_net": -0.46,
                    "auc": 0.50,
                }
            },
            "experiments.json": {
                "summary": {
                    "news_helps": False,
                    "stacking_beats_auc_weight": True,
                    "positive_sharpe_found": False,
                }
            },
        }

        self.patches = [
            patch.object(medallion, "EVAL_DIR", self.eval_dir),
            patch.object(
                medallion,
                "MEDALLION_DIR",
                self.eval_dir / "medallion",
            ),
            patch.object(medallion, "BRONZE_DIR", self.bronze_dir),
        ]

        for parche in self.patches:
            parche.start()
            self.addCleanup(parche.stop)

        self.guardar_entradas()

    def guardar_entradas(self):
        for nombre, datos in self.datos.items():
            (self.eval_dir / nombre).write_text(
                json.dumps(datos),
                encoding="utf-8",
            )

    def ejecutar_pipeline(self):
        medallion.ejecutar_bronze()
        return medallion.ejecutar_silver()

    def test_datos_validos(self):
        reporte = self.ejecutar_pipeline()

        self.assertEqual(reporte["estado"], "VALIDO")
        self.assertEqual(len(reporte["archivos_validados"]), 3)
        self.assertEqual(reporte["errores"], [])

    def test_sharpe_negativo_es_valido(self):
        reporte = self.ejecutar_pipeline()

        self.assertEqual(reporte["estado"], "VALIDO")

    def test_auc_fuera_de_rango(self):
        self.datos["backtest_topk.json"]["strategy"]["auc"] = 1.5
        self.guardar_entradas()

        reporte = self.ejecutar_pipeline()

        self.assertEqual(reporte["estado"], "INVALIDO")
        self.assertTrue(
            any("auc" in error for error in reporte["errores"])
        )

    def test_drawdown_fuera_de_rango(self):
        self.datos["backtest_topk.json"]["strategy"][
            "max_drawdown"
        ] = -1.5
        self.guardar_entradas()

        reporte = self.ejecutar_pipeline()

        self.assertEqual(reporte["estado"], "INVALIDO")

    def test_campo_faltante(self):
        del self.datos["backtest_topk.json"]["strategy"][
            "sharpe_net"
        ]
        self.guardar_entradas()

        reporte = self.ejecutar_pipeline()

        self.assertEqual(reporte["estado"], "INVALIDO")

    def test_archivos_originales_no_se_modifican(self):
        hashes_antes = {
            nombre: hashlib.sha256(
                (self.eval_dir / nombre).read_bytes()
            ).hexdigest()
            for nombre in self.datos
        }

        self.ejecutar_pipeline()

        hashes_despues = {
            nombre: hashlib.sha256(
                (self.eval_dir / nombre).read_bytes()
            ).hexdigest()
            for nombre in self.datos
        }

        self.assertEqual(hashes_antes, hashes_despues)

    def test_detecta_bronze_alterado(self):
        medallion.ejecutar_bronze()

        ruta = self.bronze_dir / "backtest_topk.json"
        ruta.write_text('{"alterado": true}', encoding="utf-8")

        reporte = medallion.ejecutar_silver()

        self.assertEqual(reporte["estado"], "INVALIDO")
        self.assertTrue(
            any("Integridad" in error for error in reporte["errores"])
        )

    def test_inconsistencia_entre_fuentes(self):
        self.datos["portfolio_metrics.json"][
            "portfolio_dual"
        ]["sharpe_net"] = 0.5
        self.guardar_entradas()

        reporte = self.ejecutar_pipeline()

        self.assertEqual(reporte["estado"], "VALIDO")
        self.assertTrue(reporte["advertencias"])


    def test_gold_con_datos_validos(self):
        reporte_silver = self.ejecutar_pipeline()

        # Agregar métricas que Gold necesita.
        # Se modifica únicamente el entorno temporal.
        self.datos["backtest_topk.json"]["strategy"].update({
            "sharpe_gross": 1.52,
            "n_periods": 172,
        })

        self.datos["backtest_topk.json"][
            "benchmark_naftrac"
        ] = {
            "sharpe_ratio": 0.81,
            "total_return": 0.15,
            "max_drawdown": -0.21,
        }

        self.guardar_entradas()
        reporte_silver = self.ejecutar_pipeline()

        gold = medallion.ejecutar_gold(reporte_silver)

        self.assertEqual(gold["capa"], "gold")
        self.assertEqual(gold["producto"], "evaluacion_riesgo")
        self.assertEqual(gold["decision"], "NO_BET_LIVE")
        self.assertEqual(gold["calidad"]["estado"], "VALIDO")

    def test_gold_rechaza_silver_invalido(self):
        reporte_silver = self.ejecutar_pipeline()
        reporte_silver["estado"] = "INVALIDO"

        with self.assertRaises(ValueError):
            medallion.ejecutar_gold(reporte_silver)

    def test_gold_detecta_bronze_alterado(self):
        reporte_silver = self.ejecutar_pipeline()

        ruta = self.bronze_dir / "backtest_topk.json"
        ruta.write_text('{"alterado": true}', encoding="utf-8")

        with self.assertRaises(ValueError):
            medallion.ejecutar_gold(reporte_silver)

    def test_gold_anterior_se_elimina_si_silver_falla(self):
        gold_dir = self.eval_dir / "medallion" / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)

        salida = gold_dir / "evaluacion_riesgo.json"
        salida.write_text(
            '{"estado": "ANTERIOR"}',
            encoding="utf-8",
        )

        reporte_invalido = {
            "estado": "INVALIDO",
            "archivos_validados": [],
            "advertencias": [],
        }

        with self.assertRaises(ValueError):
            medallion.ejecutar_gold(reporte_invalido)

        self.assertFalse(salida.exists())




if __name__ == "__main__":
    unittest.main()
