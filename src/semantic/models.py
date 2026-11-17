"""Contratos compartidos de la capa semántica de noticias."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
import re
from typing import Any


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SemanticError(Exception):
    """Error esperado con estado seguro para consumidores."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class SemanticResponse:
    schema_version: int
    status: str
    query: dict[str, Any]
    data: dict[str, Any] | None
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "query": json_safe(self.query),
            "data": json_safe(self.data),
            "warnings": list(self.warnings),
            "provenance": json_safe(self.provenance),
        }


def json_safe(value: Any) -> Any:
    """Convierte valores de pandas/numpy en JSON estándar y rechaza NaN."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def response(
    status: str,
    query: dict[str, Any],
    data: dict[str, Any] | None = None,
    warnings: list[str] | tuple[str, ...] = (),
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return SemanticResponse(
        schema_version=1,
        status=status,
        query=query,
        data=data,
        warnings=tuple(warnings),
        provenance=provenance or {},
    ).to_dict()


def normalize_identifier(value: Any, entity_type: str) -> str:
    if not isinstance(value, str):
        raise SemanticError("invalid_input", "El identificador debe ser texto.")
    normalized = value.strip()
    if not normalized or not IDENTIFIER_PATTERN.fullmatch(normalized):
        raise SemanticError(
            "invalid_input",
            "El identificador está vacío o contiene caracteres no permitidos.",
        )
    return normalized.upper() if entity_type == "ticker" else normalized.lower()


def parse_iso_date(value: Any, field: str = "fecha") -> date:
    if not isinstance(value, str) or not ISO_DATE_PATTERN.fullmatch(value):
        raise SemanticError("invalid_input", f"{field} debe tener formato YYYY-MM-DD.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SemanticError("invalid_input", f"{field} no es una fecha válida.") from exc


def dumps(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)
