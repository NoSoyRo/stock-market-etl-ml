"""Servidor MCP para el dominio de Noticias Segmentadas.

Expone herramientas basadas en el estándar Model Context Protocol (MCP)
para que agentes conversacionales consulten información preparada por
la capa semántica y Gold, sin realizar scraping ni recalcular pipelines.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from src.semantic.models import json_safe
from src.semantic.repository import DEFAULT_NEWS_ROOT
from src.semantic.service import NewsSemanticService

logger = logging.getLogger(__name__)


def create_news_mcp_server(news_root: Path | None = None) -> MCPServer:
    """Crea y configura el servidor MCP de Noticias Segmentadas."""
    root_path = Path(news_root) if news_root is not None else DEFAULT_NEWS_ROOT
    service = NewsSemanticService(news_root=root_path)

    server = MCPServer(
        name="noticias-segmentadas",
        title="Servidor MCP de Noticias Segmentadas",
        description=(
            "Servidor MCP para consultar métricas semánticas y Gold de noticias financieras. "
            "Consume la capa semántica sobre datasets Gold (today y historical) y Silver. "
            "No realiza scraping, descargas ni recálculos."
        ),
        version="1.0.0",
    )

    @server.tool(name="consultar_tema_noticias", description="Consulta indicadores de un tema como trump o market, sin atribuirlos a una empresa. Fecha opcional YYYY-MM-DD.")
    def consultar_tema_noticias(topic: str, fecha: str | None = None) -> dict[str, Any]:
        return service.consultar_tema(topic, fecha=fecha)

    @server.tool(name="consultar_evidencia_noticias", description="Recupera titulares, enlaces y procedencia que respaldan un resultado. entity_type es ticker o topic; fecha es YYYY-MM-DD. Consultar antes de explicar eventos.")
    def consultar_evidencia_noticias(
        entity_type: str, entity_id: str, fecha: str, limit: int = 10, offset: int = 0,
    ) -> dict[str, Any]:
        return service.consultar_evidencia(entity_type, entity_id, fecha, limit=limit, offset=offset)

    @server.tool(name="consultar_historial_noticias", description="Serie diaria para una empresa o un tema. entity_type: ticker o topic. inicio y fin: YYYY-MM-DD. Conserva huecos y ventanas de cálculo.")
    def consultar_historial_noticias(
        entity_type: str, entity_id: str, inicio: str, fin: str,
    ) -> dict[str, Any]:
        return service.consultar_historial(entity_type, entity_id, inicio, fin)

    @server.tool(
        name="ver_noticias",
        description=(
            "Consulta la información semántica y métricas Gold disponibles para una empresa (ticker). "
            "Devuelve sentimiento general, volumen ponderado, relevancia promedio, cantidad de artículos, "
            "fuentes, fecha de cálculo, advertencias y resumen semántico interpretado."
        ),
    )
    def ver_noticias(
        ticker: str,
        fecha: str | None = None,
        limite: int | None = None,
    ) -> dict[str, Any]:
        """Consulta métricas semánticas de noticias para un ticker."""
        if not ticker or not isinstance(ticker, str) or not ticker.strip():
            return {
                "ok": False,
                "status": "invalid_input",
                "error": "El parámetro 'ticker' es obligatorio y debe ser una cadena no vacía.",
            }

        ticker_clean = ticker.strip().upper()
        fecha_clean = fecha.strip() if isinstance(fecha, str) and fecha.strip() else None

        try:
            res = service.consultar_empresa(ticker=ticker_clean, fecha=fecha_clean)
        except Exception as exc:
            logger.exception("Error al consultar empresa %s: %s", ticker_clean, exc)
            return {
                "ok": False,
                "status": "error",
                "ticker": ticker_clean,
                "error": f"Error inesperado al consultar la capa semántica: {exc}",
            }

        status = res.get("status", "unknown")
        if status in {"ok", "stale", "no_recent_articles"}:
            data = res.get("data") or {}
            metrics = data.get("metrics") or {}
            payload: dict[str, Any] = {
                "ok": True,
                "status": status,
                "ticker": ticker_clean,
                "fecha_utc": data.get("date_utc"),
                "window_days": data.get("window_days"),
                "scoring_method": data.get("scoring_method"),
                "metricas": {
                    "sentimiento_general": metrics.get("sentimiento_general"),
                    "volumen_informativo_ponderado": metrics.get("volumen_informativo_ponderado"),
                    "relevancia_promedio": metrics.get("relevancia_promedio"),
                    "cantidad_articulos": metrics.get("cantidad_articulos"),
                    "cambio_sentimiento": metrics.get("cambio_sentimiento"),
                    "dispersion_sentimiento": metrics.get("dispersion_sentimiento"),
                    "sentimiento_esg": metrics.get("sentimiento_esg"),
                },
                "fuentes": data.get("sources", []),
                "resumen_semantico": data.get("interpretation"),
                "cobertura_esg": data.get("esg_coverage", "unknown"),
                "fecha_calculo_utc": data.get("calculated_at"),
                "antiguedad_calculo_horas": data.get("calculation_age_hours"),
                "advertencias": res.get("warnings", []),
                "procedencia": res.get("provenance", {}),
            }
            if limite is not None and isinstance(limite, int) and limite > 0:
                payload["limite_solicitado"] = limite
            return json_safe(payload)

        warnings = res.get("warnings", [])
        msg = warnings[0] if warnings else f"No se encontraron datos para {ticker_clean} (estado: {status})."
        return json_safe({
            "ok": False,
            "status": status,
            "ticker": ticker_clean,
            "fecha": fecha_clean,
            "error": msg,
            "advertencias": warnings,
            "procedencia": res.get("provenance", {}),
        })

    @server.tool(
        name="resumen_noticias",
        description=(
            "Obtiene un resumen de la evolución de las métricas de noticias para un ticker en un periodo. "
            "Devuelve promedio de sentimiento, tendencia (al alza, a la baja o estable), volumen ponderado, "
            "total de artículos y serie cronológica de observaciones. "
            "Si no se proporcionan fechas, consulta el rango disponible en la capa Gold."
        ),
    )
    def resumen_noticias(
        ticker: str,
        fecha_inicio: str | None = None,
        fecha_fin: str | None = None,
    ) -> dict[str, Any]:
        """Resume la evolución de métricas de noticias en un periodo."""
        if not ticker or not isinstance(ticker, str) or not ticker.strip():
            return {
                "ok": False,
                "status": "invalid_input",
                "error": "El parámetro 'ticker' es obligatorio y debe ser una cadena no vacía.",
            }

        ticker_clean = ticker.strip().upper()
        ini = fecha_inicio.strip() if isinstance(fecha_inicio, str) and fecha_inicio.strip() else None
        fin = fecha_fin.strip() if isinstance(fecha_fin, str) and fecha_fin.strip() else None

        if ini is None or fin is None:
            cov = service.consultar_cobertura()
            date_range = (cov.get("data") or {}).get("date_range_utc")
            if not date_range or not date_range.get("first") or not date_range.get("last"):
                return {
                    "ok": False,
                    "status": "data_unavailable",
                    "ticker": ticker_clean,
                    "error": "No se encontraron fechas disponibles en la cobertura histórica de Gold.",
                }
            if ini is None:
                ini = date_range["first"]
            if fin is None:
                fin = date_range["last"]

        try:
            res = service.consultar_historial(
                entity_type="ticker",
                entity_id=ticker_clean,
                inicio=ini,
                fin=fin,
                incluir_conteo_unico=True,
            )
        except Exception as exc:
            logger.exception("Error al consultar historial para %s: %s", ticker_clean, exc)
            return {
                "ok": False,
                "status": "error",
                "ticker": ticker_clean,
                "error": f"Error inesperado al consultar historial: {exc}",
            }

        if res.get("status") == "ok":
            results = (res.get("data") or {}).get("results", [])
            if not results:
                return json_safe({
                    "ok": True,
                    "status": "no_data",
                    "ticker": ticker_clean,
                    "periodo": {"fecha_inicio": ini, "fecha_fin": fin, "dias_observados": 0},
                    "mensaje": f"No se registraron observaciones históricas para {ticker_clean} en el rango indicado.",
                    "advertencias": res.get("warnings", []),
                })

            contracts = {(r.get("window_days"), r.get("scoring_method")) for r in results}
            if len(contracts) != 1:
                return json_safe({
                    "ok": False, "status": "incompatible_metrics", "ticker": ticker_clean,
                    "error": "El historial combina ventanas o métodos distintos; no se agregaron las métricas.",
                })
            scores: list[float] = []
            volumenes: list[float] = []

            for item in results:
                m = item.get("metrics") or {}
                s = m.get("sentimiento_general")
                v = m.get("volumen_informativo_ponderado")
                if s is not None and isinstance(s, (int, float)):
                    scores.append(float(s))
                if v is not None and isinstance(v, (int, float)):
                    volumenes.append(float(v))

            delta = 0.0
            tendencia = "insuficientes_datos"
            if len(scores) >= 2:
                delta = round(scores[-1] - scores[0], 4)
                if delta > 0.05:
                    tendencia = "favorable / al alza"
                elif delta < -0.05:
                    tendencia = "desfavorable / a la baja"
                else:
                    tendencia = "estable"
            elif len(scores) == 1:
                tendencia = "estable (única observación)"

            avg_score = round(sum(scores) / len(scores), 4) if scores else None
            avg_vol = round(sum(volumenes) / len(volumenes), 4) if volumenes else 0.0
            total_articulos = (res.get("data") or {}).get("unique_recent_articles")

            serie = [
                {
                    "fecha_utc": r.get("date_utc"),
                    "sentimiento": (r.get("metrics") or {}).get("sentimiento_general"),
                    "volumen_ponderado": (r.get("metrics") or {}).get("volumen_informativo_ponderado"),
                    "articulos": (r.get("metrics") or {}).get("cantidad_articulos"),
                    "interpretacion": r.get("interpretation"),
                }
                for r in results
            ]

            return json_safe({
                "ok": True,
                "status": "ok",
                "ticker": ticker_clean,
                "periodo": {
                    "fecha_inicio": ini,
                    "fecha_fin": fin,
                    "dias_con_datos": len(results),
                },
                "evolucion_sentimiento": {
                    "promedio": avg_score,
                    "inicial": scores[0] if scores else None,
                    "final": scores[-1] if scores else None,
                    "cambio_en_periodo": delta,
                    "tendencia": tendencia,
                },
                "resumen_volumen": {
                    "promedio_volumen_ponderado": avg_vol,
                    "articulos_totales_observados": total_articulos,
                    "criterio_conteo": "Artículos únicos de las ventanas recientes de las fechas observadas; no suma diaria ni publicaciones estrictamente dentro del periodo.",
                },
                "serie_temporal": serie,
                "advertencias": res.get("warnings", []),
            })

        warnings = res.get("warnings", [])
        msg = warnings[0] if warnings else f"No se pudo obtener el historial (estado: {res.get('status')})."
        return json_safe({
            "ok": False,
            "status": res.get("status", "error"),
            "ticker": ticker_clean,
            "periodo": {"fecha_inicio": ini, "fecha_fin": fin},
            "error": msg,
            "advertencias": warnings,
        })

    @server.tool(
        name="comparar_noticias",
        description=(
            "Compara múltiples activos utilizando exclusivamente métricas de la capa semántica o Gold. "
            "Genera rankings comparativos por sentimiento y volumen informativo ponderado, "
            "detallando las métricas de cada ticker en la fecha indicada (o hoy por defecto)."
        ),
    )
    def comparar_noticias(
        tickers: list[str] | str,
        fecha: str | None = None,
    ) -> dict[str, Any]:
        """Compara las métricas de noticias de varios tickers en una fecha."""
        if isinstance(tickers, str):
            tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        elif isinstance(tickers, (list, tuple)):
            tickers_list = [str(t).strip().upper() for t in tickers if str(t).strip()]
        else:
            tickers_list = []

        if not tickers_list:
            return {
                "ok": False,
                "status": "invalid_input",
                "error": "Se debe proporcionar al menos un ticker en 'tickers' (lista o texto separado por comas).",
            }

        # Deduplicar preservando orden
        tickers_unicos = list(dict.fromkeys(tickers_list))
        fecha_clean = fecha.strip() if isinstance(fecha, str) and fecha.strip() else None

        comparativas: list[dict[str, Any]] = []
        no_disponibles: list[dict[str, Any]] = []

        for ticker in tickers_unicos:
            try:
                res = service.consultar_empresa(ticker=ticker, fecha=fecha_clean)
                status = res.get("status")
                if status in {"ok", "stale", "no_recent_articles"}:
                    data = res.get("data") or {}
                    m = data.get("metrics") or {}
                    comparativas.append({
                        "ticker": ticker,
                        "status": status,
                        "fecha_utc": data.get("date_utc"),
                        "window_days": data.get("window_days"),
                        "scoring_method": data.get("scoring_method"),
                        "sentimiento_general": m.get("sentimiento_general"),
                        "volumen_informativo_ponderado": m.get("volumen_informativo_ponderado"),
                        "cantidad_articulos": m.get("cantidad_articulos"),
                        "relevancia_promedio": m.get("relevancia_promedio"),
                        "dispersion_sentimiento": m.get("dispersion_sentimiento"),
                        "fuentes": data.get("sources", []),
                        "resumen": data.get("interpretation"),
                    })
                else:
                    warnings = res.get("warnings", [])
                    no_disponibles.append({
                        "ticker": ticker,
                        "status": status,
                        "motivo": warnings[0] if warnings else "Sin datos disponibles.",
                    })
            except Exception as exc:
                no_disponibles.append({
                    "ticker": ticker,
                    "status": "error",
                    "motivo": f"Fallo al consultar: {exc}",
                })

        contracts = {(r["fecha_utc"], r["window_days"], r["scoring_method"])
                     for r in comparativas}
        if len(contracts) > 1 or any(None in contract for contract in contracts):
            return json_safe({
                "ok": False, "status": "incompatible_metrics",
                "error": "No se pueden comparar fechas, ventanas o métodos distintos o desconocidos.",
                "comparativa": comparativas, "no_disponibles": no_disponibles,
                "ranking_por_sentimiento": [], "ranking_por_volumen": [],
            })

        # Rankings only after validating the metric contracts.
        def _score_key(item: dict[str, Any]) -> float:
            val = item.get("sentimiento_general")
            return float(val) if val is not None else -999.0

        def _vol_key(item: dict[str, Any]) -> float:
            val = item.get("volumen_informativo_ponderado")
            return float(val) if val is not None else -999.0

        ranking_sentimiento = [
            item["ticker"]
            for item in sorted(comparativas, key=_score_key, reverse=True)
            if item.get("sentimiento_general") is not None
        ]

        ranking_volumen = [
            item["ticker"]
            for item in sorted(comparativas, key=_vol_key, reverse=True)
            if item.get("volumen_informativo_ponderado") is not None
        ]

        return json_safe({
            "ok": True,
            "status": "ok",
            "fecha_consulta": fecha_clean,
            "total_solicitados": len(tickers_unicos),
            "total_encontrados": len(comparativas),
            "total_no_disponibles": len(no_disponibles),
            "ranking_por_sentimiento": ranking_sentimiento,
            "ranking_por_volumen": ranking_volumen,
            "comparativa": comparativas,
            "no_disponibles": no_disponibles,
        })

    @server.tool(
        name="listar_tickers_noticias",
        description=(
            "Lista todos los tickers y temas disponibles en la capa Gold de noticias, "
            "incluyendo el rango de fechas cubierto, fuentes observadas y fecha del último cálculo."
        ),
    )
    def listar_tickers_noticias() -> dict[str, Any]:
        """Indica qué tickers, temas y fechas tienen métricas disponibles."""
        try:
            res = service.consultar_cobertura()
        except Exception as exc:
            logger.exception("Error al consultar cobertura: %s", exc)
            return {
                "ok": False,
                "status": "error",
                "error": f"Error al consultar la cobertura de datos: {exc}",
            }

        if res.get("status") == "ok":
            data = res.get("data") or {}
            entities = data.get("entities") or {}
            tickers = entities.get("tickers") or []
            topics = entities.get("topics") or []

            return json_safe({
                "ok": True,
                "status": "ok",
                "total_tickers": len(tickers),
                "total_temas": len(topics),
                "tickers_disponibles": tickers,
                "temas_disponibles": topics,
                "rango_fechas_utc": data.get("date_range_utc"),
                "fuentes_detectadas": data.get("sources", []),
                "ultimo_calculo_utc": data.get("last_calculated_at"),
                "metodos_scoring": data.get("scoring_methods", []),
                "ventanas_dias": data.get("window_days", []),
                "advertencias": res.get("warnings", []),
            })

        warnings = res.get("warnings", [])
        return json_safe({
            "ok": False,
            "status": res.get("status", "error"),
            "error": warnings[0] if warnings else "No se pudo obtener la cobertura.",
            "advertencias": warnings,
        })

    return server


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada de línea de comandos para ejecutar el servidor MCP."""
    parser = argparse.ArgumentParser(
        description="Servidor MCP de Noticias Segmentadas (Model Context Protocol)."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Protocolo de transporte para MCP (por defecto: stdio).",
    )
    parser.add_argument(
        "--news-root",
        type=Path,
        default=None,
        help="Ruta alternativa al directorio data/news del medallion.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Iniciando servidor MCP de Noticias Segmentadas (transporte: %s)...", args.transport)

    server = create_news_mcp_server(news_root=args.news_root)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()

