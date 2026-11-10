"""Consultas semánticas de solo lectura para noticias medallion."""
from __future__ import annotations

from datetime import date, datetime, timezone
import logging
import math
from pathlib import Path
from typing import Callable, Any

import pandas as pd

from src.semantic.interpretation import business_metrics
from src.semantic.models import (
    SemanticError,
    normalize_identifier,
    parse_iso_date,
    response,
)
from src.semantic.repository import DEFAULT_NEWS_ROOT, NewsRepository

MAX_HISTORY_DAYS = 366
MAX_EVIDENCE_LIMIT = 100
logger = logging.getLogger(__name__)


class NewsSemanticService:
    def __init__(
        self,
        news_root: Path = DEFAULT_NEWS_ROOT,
        *,
        clock: Callable[[], datetime] | None = None,
        max_calculation_age_hours: float = 24,
    ):
        if not math.isfinite(max_calculation_age_hours) or max_calculation_age_hours <= 0:
            raise ValueError("max_calculation_age_hours debe ser un número positivo finito.")
        self.repository = NewsRepository(Path(news_root))
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.max_calculation_age_hours = float(max_calculation_age_hours)

    def consultar_empresa(
        self,
        ticker: str,
        *,
        fecha: str | None = None,
        window_days: int | None = None,
        corte: str | datetime | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"entity_type": "ticker", "entity_id": ticker, "mode": "today"}
        try:
            ticker = normalize_identifier(ticker, "ticker")
            query["entity_id"] = ticker
            return self._consultar_entidad(
                "ticker", ticker, fecha=fecha, window_days=window_days, corte=corte, query=query
            )
        except SemanticError as exc:
            return self._failure(exc, query)

    def consultar_tema(
        self,
        topic: str,
        *,
        fecha: str | None = None,
        window_days: int | None = None,
        corte: str | datetime | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"entity_type": "topic", "entity_id": topic, "mode": "today"}
        try:
            topic = normalize_identifier(topic, "topic")
            query["entity_id"] = topic
            return self._consultar_entidad(
                "topic", topic, fecha=fecha, window_days=window_days, corte=corte, query=query
            )
        except SemanticError as exc:
            return self._failure(exc, query)

    def _consultar_entidad(
        self,
        entity_type: str,
        entity_id: str,
        *,
        fecha: str | None,
        window_days: int | None,
        corte: str | datetime | None,
        query: dict[str, Any],
    ) -> dict[str, Any]:
        if window_days is not None and (
            isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 1
        ):
            raise SemanticError("invalid_input", "window_days debe ser un entero positivo.")
        cutoff = self._parse_cutoff(corte)
        if cutoff is not None:
            query["cutoff"] = cutoff.isoformat()

        if fecha is None:
            query["mode"] = "today"
            frame = self.repository.read_today()
            today = self._now().date()
            if not frame.empty and any(value.date() != today for value in frame["date"]):
                dates = sorted({value.date().isoformat() for value in frame["date"]})
                return response(
                    "stale",
                    query,
                    data={"expected_date_utc": today.isoformat(), "available_dates_utc": dates},
                    warnings=["La tabla del día no corresponde al día UTC actual."],
                    provenance={"dataset": "gold/today_features.parquet"},
                )
            target_date = today
            dataset = "gold/today_features.parquet"
        else:
            target_date = parse_iso_date(fecha)
            query["mode"] = "historical"
            query["date"] = target_date.isoformat()
            today = self._now().date()
            if target_date > today:
                raise SemanticError("invalid_input", "No se admiten fechas futuras.")
            frame = self.repository.read_history()
            dataset = "gold/historical_features.parquet"

        row = self._select_row(frame, entity_type, entity_id, target_date)
        if row is None:
            return response(
                "not_found",
                query,
                warnings=[
                    f"No hay una fila gold guardada para {entity_type}={entity_id} "
                    f"en {target_date.isoformat()}."
                ],
                provenance={"dataset": dataset, "search_date_utc": target_date.isoformat()},
            )
        if window_days is not None and int(row["window_days"]) != window_days:
            return response(
                "incompatible_metrics",
                query,
                data={"available_window_days": int(row["window_days"])},
                warnings=["La fila guardada usa otra ventana; no se recalculó ni sustituyó."],
                provenance={"dataset": dataset, "key": self._key(row)},
            )
        if cutoff is not None and row["as_of"] > cutoff:
            return response(
                "not_found",
                query,
                warnings=[
                    "Gold conserva una sola versión por fecha y no hay una versión "
                    "disponible para el corte solicitado."
                ],
                provenance={"dataset": dataset, "key": self._key(row)},
            )

        data, warnings = self._business_row(row)
        age_hours = (self._now() - row["as_of"].to_pydatetime()).total_seconds() / 3600
        if age_hours < 0:
            return response(
                "data_integrity_error",
                query,
                warnings=["El cálculo gold tiene un as_of posterior al reloj de consulta."],
                provenance={"dataset": dataset, "key": self._key(row)},
            )
        data["calculation_age_hours"] = round(age_hours, 3)
        data["latest_evidence_fetched_at"] = None
        data["evidence_freshness"] = "unknown"
        status = "ok" if int(row["n_articles"]) > 0 else "no_recent_articles"
        if status == "no_recent_articles":
            warnings.append(
                "Gold no contiene artículos en la ventana reciente; no se interpreta "
                "el score como neutralidad."
            )
        if row["date"].date() != self._now().date() and query["mode"] == "today":
            status = "stale"
        elif age_hours > self.max_calculation_age_hours:
            status = "stale"
            warnings.append("La edad del cálculo supera el umbral operativo configurado.")
        if row["news_sent_delta"] is not None and int(row["n_articles"]) > 0:
            warnings.append(
                "La cobertura de la ventana previa no se verificó; cambio_sentimiento "
                "usa la fórmula gold y no garantiza evidencia en ambas ventanas."
            )
        warnings.append("La cobertura ESG es desconocida hasta revisar los titulares.")
        return response(
            status,
            query,
            data=data,
            warnings=warnings,
            provenance={
                "dataset": dataset,
                "key": self._key(row),
                "evidence_expected_article_ids": len(row["article_ids"]),
                "evidence_resolved": False,
                "evidence_available": None,
            },
        )

    def consultar_historial(
        self,
        entity_type: str,
        entity_id: str,
        inicio: str,
        fin: str,
        *,
        window_days: int | None = None,
        incluir_conteo_unico: bool = False,
    ) -> dict[str, Any]:
        query = {"entity_type": entity_type, "entity_id": entity_id, "mode": "history"}
        try:
            entity_type, entity_id = self._normalize_entity(entity_type, entity_id)
            query.update({"entity_type": entity_type, "entity_id": entity_id})
            start = parse_iso_date(inicio, "inicio")
            end = parse_iso_date(fin, "fin")
            query.update({"start_date": start.isoformat(), "end_date": end.isoformat()})
            if start > end:
                raise SemanticError("invalid_input", "inicio debe ser anterior o igual a fin.")
            if (end - start).days + 1 > MAX_HISTORY_DAYS:
                raise SemanticError(
                    "invalid_input", f"El rango histórico no puede superar {MAX_HISTORY_DAYS} días."
                )
            if window_days is not None and (
                isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 1
            ):
                raise SemanticError("invalid_input", "window_days debe ser un entero positivo.")
            frame = self.repository.read_history()
            frame = self._filter_entity(frame, entity_type, entity_id)
            mask = frame["date"].map(lambda value: start <= value.date() <= end)
            frame = frame[mask].sort_values("date")
            window_mismatches: list[dict[str, Any]] = []
            if window_days is not None:
                available_windows = sorted(int(value) for value in frame.window_days.unique())
                window_mismatches = [
                    {"date_utc": row.date.date().isoformat(), "window_days": int(row.window_days)}
                    for row in frame.itertuples()
                    if int(row.window_days) != window_days
                ]
                frame = frame[frame.window_days == window_days]
                if frame.empty and available_windows:
                    return response(
                        "incompatible_metrics",
                        query,
                        data={"available_window_days": available_windows},
                        warnings=["No existe la ventana solicitada para las fechas encontradas."],
                        provenance={"dataset": "gold/historical_features.parquet"},
                    )
            if frame.empty:
                return response(
                    "not_found",
                    query,
                    data={"results": []},
                    warnings=["No hay resultados guardados en el rango solicitado."],
                    provenance={"dataset": "gold/historical_features.parquet"},
                )
            results = [self._history_item(row) for _, row in frame.iterrows()]
            unique_count = None
            count_warnings = []
            if incluir_conteo_unico:
                try:
                    silver = self.repository.read_silver()
                    article_ids = set()
                    for _, row in frame.iterrows():
                        evidence, missing_articles, missing_batches = self._resolve_evidence(row, silver)
                        if missing_articles or missing_batches:
                            raise SemanticError("data_integrity_error", "La evidencia histórica está incompleta.")
                        recent = evidence[evidence["_period"] == "reciente"]
                        if len(recent) != int(row["n_articles"]):
                            raise SemanticError("data_integrity_error", "La evidencia no coincide con el conteo gold.")
                        article_ids.update(recent.article_id)
                    unique_count = len(article_ids)
                except SemanticError as exc:
                    count_warnings.append(
                        f"No se calculó el total único de artículos: {exc.message}"
                    )
            return response(
                "ok",
                query,
                data={
                    "results": results,
                    "observed_dates_utc": [item["date_utc"] for item in results],
                    "window_days_filter": window_days,
                    "dates_with_other_windows": window_mismatches,
                    "unique_recent_articles": unique_count,
                },
                warnings=[
                    "Los días sin fila se conservan como huecos; n_articles no se suma "
                    "entre ventanas superpuestas.",
                    *(
                        ["Algunas fechas tienen otra ventana y no se incluyeron."]
                        if window_mismatches
                        else []
                    ),
                    *count_warnings,
                ],
                provenance={"dataset": "gold/historical_features.parquet"},
            )
        except SemanticError as exc:
            return self._failure(exc, query)

    def comparar_fechas(
        self,
        entity_type: str,
        entity_id: str,
        fecha_a: str,
        fecha_b: str,
    ) -> dict[str, Any]:
        query = {"entity_type": entity_type, "entity_id": entity_id, "mode": "compare"}
        try:
            entity_type, entity_id = self._normalize_entity(entity_type, entity_id)
            first = parse_iso_date(fecha_a, "fecha_a")
            second = parse_iso_date(fecha_b, "fecha_b")
            query.update(
                {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "date_a": first.isoformat(),
                    "date_b": second.isoformat(),
                }
            )
            frame = self._filter_entity(self.repository.read_history(), entity_type, entity_id)
            row_a = self._select_row(frame, entity_type, entity_id, first)
            row_b = self._select_row(frame, entity_type, entity_id, second)
            if row_a is None or row_b is None:
                missing = [
                    day.isoformat()
                    for day, row in ((first, row_a), (second, row_b))
                    if row is None
                ]
                return response(
                    "not_found",
                    query,
                    warnings=[f"No hay fila gold guardada para: {', '.join(missing)}."],
                    provenance={"dataset": "gold/historical_features.parquet"},
                )
            if (
                int(row_a["window_days"]) != int(row_b["window_days"])
                or row_a["scoring_method"] != row_b["scoring_method"]
            ):
                return response(
                    "incompatible_metrics",
                    query,
                    data={
                        "window_days": [int(row_a["window_days"]), int(row_b["window_days"])],
                        "scoring_method": [row_a["scoring_method"], row_b["scoring_method"]],
                    },
                    warnings=["Las métricas no se pueden comparar con ventanas/métodos distintos."],
                    provenance={"dataset": "gold/historical_features.parquet"},
                )
            a, warnings_a = self._business_row(row_a)
            b, warnings_b = self._business_row(row_b)
            score_a = a["metrics"]["sentimiento_general"]
            score_b = b["metrics"]["sentimiento_general"]
            return response(
                "ok",
                query,
                data={
                    "date_a": a,
                    "date_b": b,
                    "diferencia_entre_fechas": (
                        None if score_a is None or score_b is None else round(score_b - score_a, 4)
                    ),
                    "window_days": int(row_a["window_days"]),
                    "scoring_method": row_a["scoring_method"],
                },
                warnings=list(dict.fromkeys(warnings_a + warnings_b)),
                provenance={"dataset": "gold/historical_features.parquet"},
            )
        except SemanticError as exc:
            return self._failure(exc, query)

    def consultar_evidencia(
        self,
        entity_type: str,
        entity_id: str,
        fecha: str,
        *,
        periodo: str = "reciente",
        limit: int = 20,
        offset: int = 0,
        corte: str | datetime | None = None,
    ) -> dict[str, Any]:
        query = {"entity_type": entity_type, "entity_id": entity_id, "mode": "evidence"}
        try:
            entity_type, entity_id = self._normalize_entity(entity_type, entity_id)
            target_date = parse_iso_date(fecha)
            cutoff = self._parse_cutoff(corte)
            query.update(
                {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "date": target_date.isoformat(),
                    "period": periodo,
                    "limit": limit,
                    "offset": offset,
                }
            )
            if periodo not in {"reciente", "anterior", "ambos"}:
                raise SemanticError("invalid_input", "periodo debe ser reciente, anterior o ambos.")
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_EVIDENCE_LIMIT:
                raise SemanticError(
                    "invalid_input", f"limit debe estar entre 1 y {MAX_EVIDENCE_LIMIT}."
                )
            if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 1_000_000:
                raise SemanticError("invalid_input", "offset debe ser un entero entre 0 y 1000000.")
            row = self._select_row(
                self._filter_entity(self.repository.read_history(), entity_type, entity_id),
                entity_type,
                entity_id,
                target_date,
            )
            if row is None:
                return response(
                    "not_found",
                    query,
                    warnings=["No hay resultado gold para la asociación y fecha solicitadas."],
                    provenance={"dataset": "gold/historical_features.parquet"},
                )
            if cutoff is not None and row["as_of"] > cutoff:
                return response(
                    "not_found",
                    query,
                    warnings=["No hay una versión gold disponible para el corte solicitado."],
                    provenance={"dataset": "gold/historical_features.parquet", "key": self._key(row)},
                )

            silver = self.repository.read_silver()
            evidence, missing_articles, missing_batches = self._resolve_evidence(row, silver)
            if periodo == "reciente":
                evidence = evidence[evidence["_period"] == "reciente"]
            elif periodo == "anterior":
                evidence = evidence[evidence["_period"] == "anterior"]
            evidence = evidence.sort_values(
                ["event_at", "article_id", "batch_id"], ascending=[False, True, True]
            )
            total = len(evidence)
            page = evidence.iloc[offset : offset + limit]
            articles = [self._evidence_item(item) for _, item in page.iterrows()]
            warnings: list[str] = []
            if missing_articles:
                warnings.append(
                    f"Falta resolver {len(missing_articles)} article_id(s) referenciados por gold."
                )
            if missing_batches:
                warnings.append(
                    f"Falta resolver {len(missing_batches)} batch_id(s) referenciados por gold."
                )
            fetched = evidence["fetched_at"].max() if not evidence.empty else pd.NaT
            fetched_iso = None if pd.isna(fetched) else fetched.isoformat()
            age_hours = (
                None
                if pd.isna(fetched)
                else (self._now() - fetched.to_pydatetime()).total_seconds() / 3600
            )
            freshness = (
                "unknown"
                if age_hours is None or age_hours < 0
                else "fresh" if age_hours <= self.max_calculation_age_hours else "stale"
            )
            data: dict[str, Any] = {
                "articles": articles,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(articles) < total,
                "integrity": {
                    "complete": not missing_articles and not missing_batches,
                    "missing_article_ids": missing_articles,
                    "missing_batch_ids": missing_batches,
                },
                "latest_evidence_fetched_at": fetched_iso,
                "evidence_freshness": freshness,
                "esg_coverage": self._esg_coverage(evidence),
            }
            return response(
                "data_integrity_error" if missing_articles or missing_batches else "ok",
                query,
                data=data,
                warnings=warnings,
                provenance={
                    "dataset": "silver/articles.parquet",
                    "gold_dataset": "gold/historical_features.parquet",
                    "gold_key": self._key(row),
                },
            )
        except SemanticError as exc:
            return self._failure(exc, query)

    def consultar_cobertura(self) -> dict[str, Any]:
        query = {"mode": "coverage"}
        try:
            gold = self.repository.read_history()
            if gold.empty:
                return response(
                    "ok",
                    query,
                    data={
                        "result_count": 0,
                        "date_range_utc": None,
                        "entities": {"tickers": [], "topics": []},
                        "window_days": [],
                        "scoring_methods": [],
                        "last_calculated_at": None,
                    },
                    warnings=["El dataset histórico está vacío; no se infiere una falla de ingesta."],
                    provenance={"dataset": "gold/historical_features.parquet"},
                )
            silver = self.repository.read_silver()
            latest_by_source: dict[str, str] = {}
            for source, group in silver.groupby("source"):
                latest_by_source[str(source)] = group.fetched_at.max().isoformat()
            return response(
                "ok",
                query,
                data={
                    "result_count": len(gold),
                    "date_range_utc": {
                        "first": gold.date.min().date().isoformat(),
                        "last": gold.date.max().date().isoformat(),
                    },
                    "entities": {
                        "tickers": sorted(value for value in gold.ticker.unique() if value),
                        "topics": sorted(value for value in gold.topic.unique() if value),
                    },
                    "window_days": sorted(int(value) for value in gold.window_days.unique()),
                    "scoring_methods": sorted(str(value) for value in gold.scoring_method.unique()),
                    "last_calculated_at": gold.as_of.max().isoformat(),
                    "sources": sorted(str(value) for value in silver.source.unique()),
                    "latest_evidence_fetched_at_by_source": latest_by_source,
                },
                warnings=[
                    "El rango mínimo/máximo no implica continuidad ni éxito de todas las fuentes."
                ],
                provenance={
                    "dataset": "gold/historical_features.parquet",
                    "evidence_dataset": "silver/articles.parquet",
                },
            )
        except SemanticError as exc:
            return self._failure(exc, query)

    def _business_row(self, row: pd.Series) -> tuple[dict[str, Any], list[str]]:
        article_count = int(row["n_articles"])
        interpreted = business_metrics(row.to_dict(), article_count)
        data = {
            "date_utc": row["date"].date().isoformat(),
            "as_of": row["as_of"].isoformat(),
            "window_days": int(row["window_days"]),
            "scoring_method": row["scoring_method"],
            "metrics": interpreted["metrics"],
            "sources": row["sources"],
            "publication_date_missing": int(row["publication_date_missing"]),
            "interpretation": interpreted["interpretation"],
            "esg_coverage": interpreted["esg_coverage"],
            "calculated_at": row["as_of"].isoformat(),
        }
        return data, []

    @staticmethod
    def _failure(error: SemanticError, query: dict[str, Any]) -> dict[str, Any]:
        logger.warning("Semantic news query failed (%s): %s", error.status, error.message)
        return response(error.status, query, warnings=[error.message])

    def _history_item(self, row: pd.Series) -> dict[str, Any]:
        data, _warnings = self._business_row(row)
        return data

    @staticmethod
    def _key(row: pd.Series) -> dict[str, str]:
        return {
            "date": row["date"].date().isoformat(),
            "ticker": row["ticker"],
            "topic": row["topic"],
        }

    @staticmethod
    def _select_row(
        frame: pd.DataFrame, entity_type: str, entity_id: str, target_date: date
    ) -> pd.Series | None:
        selected = NewsSemanticService._filter_entity(frame, entity_type, entity_id)
        selected = selected[selected.date.map(lambda value: value.date() == target_date)]
        if selected.empty:
            return None
        if len(selected) != 1:
            raise SemanticError(
                "data_integrity_error", "La asociación y fecha no identifican una sola fila gold."
            )
        return selected.iloc[0]

    @staticmethod
    def _filter_entity(frame: pd.DataFrame, entity_type: str, entity_id: str) -> pd.DataFrame:
        if entity_type == "ticker":
            return frame[(frame.ticker == entity_id) & (frame.topic == "")]
        return frame[(frame.topic == entity_id) & (frame.ticker == "")]

    @staticmethod
    def _normalize_entity(entity_type: str, entity_id: str) -> tuple[str, str]:
        if entity_type not in {"ticker", "topic"}:
            raise SemanticError("invalid_input", "entity_type debe ser ticker o topic.")
        return entity_type, normalize_identifier(entity_id, entity_type)

    def _parse_cutoff(self, value: str | datetime | None) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            cutoff = value
        elif isinstance(value, str):
            try:
                cutoff = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise SemanticError(
                    "invalid_input", "corte debe ser una fecha/hora ISO-8601 con zona horaria."
                ) from exc
        else:
            raise SemanticError(
                "invalid_input", "corte debe ser una fecha/hora ISO-8601 con zona horaria."
            )
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise SemanticError("invalid_input", "corte debe incluir una zona horaria.")
        return cutoff.astimezone(timezone.utc)

    def _now(self) -> datetime:
        current = self.clock()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("El reloj inyectado debe devolver una fecha con zona horaria.")
        return current.astimezone(timezone.utc)

    @staticmethod
    def _resolve_evidence(
        row: pd.Series, silver: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        expected_articles = set(row["article_ids"])
        expected_batches = set(row["batch_ids"])
        if not expected_articles and int(row["n_articles"]) > 0:
            raise SemanticError(
                "data_integrity_error", "Gold declara artículos recientes pero no contiene article_ids."
            )
        association = silver[
            (silver.ticker == row["ticker"])
            & (silver.topic == row["topic"])
            & (silver.article_id.isin(expected_articles))
            & (silver.batch_id.isin(expected_batches))
            & (silver.available_at <= row["as_of"])
        ].copy()
        association["event_at"] = association.published_at.fillna(association.fetched_at)
        lower_bound = row["as_of"] - pd.Timedelta(days=int(row["window_days"]) * 2)
        association = association[
            (association.event_at >= lower_bound) & (association.event_at <= row["as_of"])
        ]
        association = association.sort_values("fetched_at").drop_duplicates(
            ["article_id", "ticker", "topic"], keep="last"
        )
        association["_period"] = association.event_at.map(
            lambda value: (
                "reciente"
                if value >= row["as_of"] - pd.Timedelta(days=int(row["window_days"]))
                else "anterior"
            )
        )
        found_articles = set(association.article_id)
        found_batches = set(association.batch_id)
        return (
            association,
            sorted(expected_articles - found_articles),
            sorted(expected_batches - found_batches),
        )

    @staticmethod
    def _evidence_item(row: pd.Series) -> dict[str, Any]:
        text = row["text"]
        if text is not None and text.strip().casefold() == row["title"].strip().casefold():
            text = None
        published = row["published_at"]
        return {
            "article_id": row["article_id"],
            "batch_id": row["batch_id"],
            "title": row["title"],
            "summary": text,
            "source": row["source"],
            "url": row["url"],
            "published_at": None if pd.isna(published) else published.isoformat(),
            "fetched_at": row["fetched_at"].isoformat(),
            "event_at": row["event_at"].isoformat(),
            "period": row["_period"],
            "sentiment": row["sentiment"],
            "provenance": {"bronze_path": row["bronze_path"]},
        }

    @staticmethod
    def _esg_coverage(evidence: pd.DataFrame) -> str:
        if evidence.empty:
            return "unknown"
        from src.ml.news_sources import is_esg_headline

        recent = evidence[evidence["_period"] == "reciente"]
        if recent.empty:
            return "unknown"
        return "present" if any(is_esg_headline(title) for title in recent.title) else "none_detected"
