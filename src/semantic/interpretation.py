"""Interpretaciones acotadas a los valores gold y a su evidencia."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).with_name("metrics.json")


def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


_METRICS = {
    item["field"]: item for item in load_catalog()["metrics"]
}


def business_metrics(row: dict[str, Any], recent_article_count: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for field, definition in _METRICS.items():
        name = definition["business_name"]
        if field == "n_articles":
            metrics[name] = int(recent_article_count)
        elif field == "news_count_7d":
            metrics[name] = row[field] if recent_article_count else 0.0
        elif field == "news_esg_score":
            metrics[name] = None
        elif recent_article_count == 0:
            metrics[name] = None
        else:
            metrics[name] = row[field]

    if recent_article_count:
        score = metrics["sentimiento_general"]
        if score is None or not math.isfinite(score):
            metrics["sentimiento_general"] = None
            interpretation = "Sentimiento no disponible: no hay una evaluación válida."
        elif score > 0:
            interpretation = "Balance positivo según el método de palabras clave."
        elif score < 0:
            interpretation = "Balance negativo según el método de palabras clave."
        else:
            interpretation = (
                "Resultado numéricamente neutral según palabras clave; "
                "no demuestra neutralidad económica."
            )
    else:
        interpretation = "Sin evidencia reciente para interpretar."
    return {
        "metrics": metrics,
        "interpretation": interpretation,
        "esg_coverage": "unknown",
    }
