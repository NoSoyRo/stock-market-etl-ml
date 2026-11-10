"""Lectura validada de las tablas Parquet del medallion de noticias."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.news_medallion import GOLD_COLUMNS, SILVER_COLUMNS
from src.semantic.models import SemanticError

logger = logging.getLogger(__name__)

DEFAULT_NEWS_ROOT = Path(__file__).resolve().parents[2] / "data" / "news"
GOLD_KEY = ["date", "ticker", "topic"]
SILVER_KEY = ["article_id", "ticker", "topic", "batch_id"]
GOLD_METRICS = (
    "news_sent_score",
    "news_count_7d",
    "news_sent_delta",
    "news_sent_vol",
    "news_relevance_avg",
    "news_esg_score",
    "n_articles",
)


class NewsRepository:
    def __init__(self, news_root: Path = DEFAULT_NEWS_ROOT):
        self.news_root = Path(news_root)

    def read_today(self) -> pd.DataFrame:
        return self._read_gold(self.news_root / "gold" / "today_features.parquet")

    def read_history(self) -> pd.DataFrame:
        return self._read_gold(self.news_root / "gold" / "historical_features.parquet")

    def read_silver(self) -> pd.DataFrame:
        path = self.news_root / "silver" / "articles.parquet"
        frame = self._read(path, "silver")
        missing = sorted(set(SILVER_COLUMNS) - set(frame.columns))
        if missing:
            raise SemanticError(
                "data_integrity_error",
                f"El esquema silver no contiene las columnas requeridas: {', '.join(missing)}.",
            )
        frame = frame[SILVER_COLUMNS].copy()
        if frame.duplicated(SILVER_KEY).any():
            raise SemanticError("data_integrity_error", "Silver contiene observaciones duplicadas.")

        for column in ("fetched_at", "available_at"):
            frame[column] = self._timestamps(frame[column], column, nullable=False)
        frame["published_at"] = self._timestamps(
            frame["published_at"], "published_at", nullable=True
        )
        if (frame["available_at"] < frame["fetched_at"]).any():
            raise SemanticError(
                "data_integrity_error", "Silver contiene available_at anterior a fetched_at."
            )
        for column in ("article_id", "batch_id", "ticker", "topic", "source", "title"):
            if frame[column].isna().any():
                raise SemanticError(
                    "data_integrity_error", f"Silver contiene valores nulos en {column}."
                )
            frame[column] = frame[column].astype(str)
        for column in ("text", "url", "bronze_path"):
            frame[column] = frame[column].map(self._nullable_string)
        frame["sentiment"] = self._numeric(frame["sentiment"], "sentiment", nullable=True)
        frame["relevance_score"] = self._numeric(
            frame["relevance_score"], "relevance_score", nullable=False
        )
        if ((frame["sentiment"].dropna() < -1) | (frame["sentiment"].dropna() > 1)).any():
            raise SemanticError("data_integrity_error", "Silver contiene sentimiento fuera de [-1, 1].")
        if (frame["relevance_score"] < 0).any():
            raise SemanticError("data_integrity_error", "Silver contiene relevancia negativa.")
        return frame

    def _read_gold(self, path: Path) -> pd.DataFrame:
        frame = self._read(path, "gold")
        missing = sorted(set(GOLD_COLUMNS) - set(frame.columns))
        if missing:
            raise SemanticError(
                "data_integrity_error",
                f"El esquema gold no contiene las columnas requeridas: {', '.join(missing)}.",
            )
        frame = frame[GOLD_COLUMNS].copy()
        if frame.duplicated(GOLD_KEY).any():
            raise SemanticError("data_integrity_error", "Gold contiene claves lógicas duplicadas.")
        frame["date"] = self._timestamps(frame["date"], "date", nullable=False)
        frame["as_of"] = self._timestamps(frame["as_of"], "as_of", nullable=False)
        if (
            (frame["as_of"] < frame["date"]).any()
            or frame["as_of"].dt.date.ne(frame["date"].dt.date).any()
            or frame["date"].dt.time.map(lambda value: value.isoformat() != "00:00:00").any()
        ):
            raise SemanticError(
                "data_integrity_error", "Gold contiene fechas UTC o cortes fuera de su día."
            )
        for column in ("ticker", "topic", "scoring_method"):
            if frame[column].isna().any():
                raise SemanticError("data_integrity_error", f"Gold contiene valores nulos en {column}.")
            frame[column] = frame[column].astype(str)
        for column in ("window_days", "publication_date_missing", "n_articles"):
            numeric = self._numeric(frame[column], column, nullable=False)
            if (numeric < 0).any() or (numeric % 1 != 0).any():
                raise SemanticError(
                    "data_integrity_error", f"Gold contiene conteos inválidos en {column}."
                )
            frame[column] = numeric.astype("int64")
        if (frame["window_days"] < 1).any():
            raise SemanticError("data_integrity_error", "Gold contiene ventanas no positivas.")
        for column in GOLD_METRICS:
            if column not in {"n_articles"}:
                frame[column] = self._numeric(frame[column], column, nullable=True)
        for column, low, high in (
            ("news_sent_score", -1, 1),
            ("news_count_7d", 0, None),
            ("news_sent_delta", -2, 2),
            ("news_sent_vol", 0, 1),
            ("news_relevance_avg", 0, 1),
            ("news_esg_score", -1, 1),
        ):
            values = frame[column].dropna()
            if ((values < low) | (False if high is None else values > high)).any():
                raise SemanticError(
                    "data_integrity_error", f"Gold contiene {column} fuera de su rango esperado."
                )
        frame["article_ids"] = frame["article_ids"].map(
            lambda value: self._json_string_list(value, "article_ids")
        )
        frame["batch_ids"] = frame["batch_ids"].map(
            lambda value: self._json_string_list(value, "batch_ids")
        )
        frame["sources"] = frame["sources"].map(self._source_list)
        return frame

    def _read(self, path: Path, dataset: str) -> pd.DataFrame:
        if not path.is_file():
            raise SemanticError("data_unavailable", f"No está disponible el dataset {dataset} solicitado.")
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.error(
                "No se pudo leer el dataset Parquet %s (%s)",
                path.name,
                type(exc).__name__,
            )
            raise SemanticError(
                "data_unavailable", f"No se pudo leer el dataset {dataset} solicitado."
            ) from exc

    @staticmethod
    def _timestamps(values: pd.Series, field: str, nullable: bool) -> pd.Series:
        parsed: list[pd.Timestamp | pd.NaT] = []
        for value in values:
            if pd.isna(value):
                if nullable:
                    parsed.append(pd.NaT)
                    continue
                raise SemanticError("data_integrity_error", f"Gold/silver contiene {field} nulo.")
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise SemanticError(
                    "data_integrity_error", f"El campo temporal {field} no tiene zona horaria."
                )
            parsed.append(timestamp.tz_convert("UTC"))
        return pd.Series(pd.to_datetime(parsed, utc=True), index=values.index)

    @staticmethod
    def _numeric(values: pd.Series, field: str, nullable: bool) -> pd.Series:
        converted = pd.to_numeric(values, errors="coerce")
        invalid = converted.isna() & values.notna()
        if invalid.any() or (not nullable and converted.isna().any()):
            raise SemanticError("data_integrity_error", f"El campo numérico {field} es inválido.")
        finite = converted.dropna().map(math.isfinite)
        if not finite.all():
            raise SemanticError("data_integrity_error", f"El campo numérico {field} no es finito.")
        return converted.astype("float64")

    @staticmethod
    def _json_string_list(value: Any, field: str) -> list[str]:
        if not isinstance(value, str):
            raise SemanticError("data_integrity_error", f"El campo {field} debe ser una lista JSON.")
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SemanticError("data_integrity_error", f"El campo {field} contiene JSON inválido.") from exc
        if not isinstance(decoded, list) or any(not isinstance(item, str) or not item for item in decoded):
            raise SemanticError("data_integrity_error", f"El campo {field} debe contener textos.")
        if len(decoded) != len(set(decoded)):
            raise SemanticError("data_integrity_error", f"El campo {field} contiene duplicados.")
        return decoded

    @staticmethod
    def _source_list(value: Any) -> list[str]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise SemanticError("data_integrity_error", "El campo sources contiene JSON inválido.") from exc
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise SemanticError("data_integrity_error", "El campo sources no es una lista.")
        if any(not isinstance(item, str) for item in value):
            raise SemanticError("data_integrity_error", "El campo sources contiene valores no textuales.")
        return [str(item) for item in value]

    @staticmethod
    def _nullable_string(value: Any) -> str | None:
        if pd.isna(value):
            return None
        return str(value)
