"""Pruebas unitarias del complemento Gemini."""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.agent.strategy_ai import StrategyAI


class TestStrategyAI(unittest.TestCase):

    def setUp(self):
        self.validation = {
            "preguntas": [
                {
                    "pregunta": "Sharpe positivo?",
                    "estado": "no_cumple",
                    "evidencia": -1.78,
                },
                {
                    "pregunta": "Costos incluidos?",
                    "estado": "cumple",
                    "evidencia": True,
                },
            ]
        }

    @patch.dict("os.environ", {"GEMINI_API_KEY": ""})
    def test_sin_api_key(self):
        result = StrategyAI().generate_questions(self.validation)

        self.assertEqual(result["estado"], "sin_api_key")
        self.assertEqual(result["preguntas"], [])

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test"})
    @patch("src.agent.strategy_ai.create_client")
    def test_tres_preguntas(self, mock_client):
        client = MagicMock()
        client.models.generate_content.return_value.text = json.dumps({
            "preguntas": [
                {
                    "pregunta": "Pregunta sobre riesgo?",
                    "respuesta": "Riesgo elevado.",
                    "estado": "no_cumple",
                    "evidencia": "Sharpe de -1.78",
                },
                {
                    "pregunta": "Pregunta sobre robustez?",
                    "respuesta": "No hay datos suficientes.",
                    "estado": "pendiente",
                    "evidencia": "Sin pruebas de estres disponibles.",
                },
                {
                    "pregunta": "Pregunta sobre metodologia?",
                    "respuesta": "Requiere revision.",
                    "estado": "pendiente",
                    "evidencia": "Informacion limitada.",
                },
            ]
        })

        mock_client.return_value = (client, None)

        result = StrategyAI().generate_questions(self.validation)

        self.assertEqual(result["estado"], "completado")
        self.assertEqual(len(result["preguntas"]), 3)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test"})
    @patch("src.agent.strategy_ai.create_client")
    def test_respuesta_invalida(self, mock_client):
        client = MagicMock()
        client.models.generate_content.return_value.text = "No es JSON"

        mock_client.return_value = (client, None)

        result = StrategyAI().generate_questions(self.validation)

        self.assertEqual(result["estado"], "error")
        self.assertEqual(result["preguntas"], [])

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test"})
    @patch("src.agent.strategy_ai.create_client")
    def test_sin_conexion(self, mock_client):
        mock_client.return_value = (None, None)

        result = StrategyAI().generate_questions(self.validation)

        self.assertEqual(result["estado"], "sin_conexion")


if __name__ == "__main__":
    unittest.main()
