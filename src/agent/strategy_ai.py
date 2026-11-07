"""Generacion de preguntas de validacion con Gemini."""

import json
import os
import time

from src.ml.gemini_config import create_client


class StrategyAI:
    """Complemento de IA para el validador de estrategias."""

    def __init__(self, model=None):
        self.model = model or os.getenv(
            "GEMINI_MODEL_STRATEGY",
            "gemini-3.8-flash",
        )

    def generate_questions(self, validation):
        """Genera preguntas adicionales a partir de la validacion."""

        if not os.getenv("GEMINI_API_KEY"):
            return {
                "estado": "sin_api_key",
                "modelo": self.model,
                "preguntas": [],
            }

        client, _ = create_client()

        if client is None:
            return {
                "estado": "sin_conexion",
                "modelo": self.model,
                "preguntas": [],
            }

        evidence = [
            {
                "pregunta": item.get("pregunta"),
                "estado": item.get("estado"),
                "evidencia": item.get("evidencia"),
            }
            for item in validation.get("preguntas", [])
            if item.get("estado") != "cumple"
        ]

        prompt = (
            "Eres un auditor tecnico de estrategias cuantitativas.\n"
            "Analiza exclusivamente las evidencias proporcionadas.\n"
            "Genera exactamente 3 preguntas adicionales de validacion "
            "sobre riesgos, metodologia y robustez.\n"
            "Responde cada pregunta usando exclusivamente las evidencias.\n"
            "Si faltan datos, indica que no se puede determinar y usa "
            "el estado pendiente.\n"
            "Los estados permitidos son cumple, no_cumple y pendiente.\n"
            "No inventes datos, no recomiendes operaciones financieras "
            "y no cambies el resultado de las reglas cuantitativas.\n"
            "Responde solamente con un objeto JSON con esta estructura:\n"
            '{"preguntas": [{"pregunta": "texto", '
            '"respuesta": "texto", "estado": "pendiente", '
            '"evidencia": "texto"}]}\n'
            "Incluye exactamente 3 objetos en preguntas.\n\n"
            "EVIDENCIAS:\n"
            + json.dumps(evidence, ensure_ascii=False, default=str)
        )

        try:
            for intento in range(3):
                try:
                    response = client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                    )
                    break

                except Exception as error:
                    codigo_error = getattr(error, "code", None)

                    if codigo_error is None:
                        codigo_error = getattr(error, "status_code", None)

                    if codigo_error not in (429, 503) or intento == 2:
                        raise

                    espera = 2 ** (intento + 1)
                    print(
                        f"Gemini temporalmente no disponible "
                        f"({codigo_error}). Reintentando en {espera}s..."
                    )
                    time.sleep(espera)

            text = (response.text or "").strip()

            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                text = text.rsplit("```", 1)[0].strip()

            result = json.loads(text)
            questions = result.get("preguntas", [])

            if (
                not isinstance(questions, list)
                or len(questions) != 3
                or not all(
                    isinstance(q, dict)
                    and all(
                        isinstance(q.get(campo), str)
                        and q[campo].strip()
                        for campo in (
                            "pregunta", "respuesta", "evidencia"
                        )
                    )
                    and q.get("estado") in (
                        "cumple", "no_cumple", "pendiente"
                    )
                    for q in questions
                )
            ):
                raise ValueError("Gemini devolvio preguntas invalidas")

            return {
                "estado": "completado",
                "modelo": self.model,
                "preguntas": questions,
            }

        except Exception as error:
            return {
                "estado": "error",
                "modelo": self.model,
                "preguntas": [],
                "detalle": str(error)[:300],
            }
