"""Pruebas unitarias para el servidor MCP de Noticias Segmentadas."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.news_medallion import GOLD_COLUMNS
from src.mcp.server import create_news_mcp_server


def _call_tool(server, name: str, args: dict) -> dict:
    """Helper síncrono para invocar una herramienta MCP y parsear su respuesta JSON."""
    result = asyncio.run(server.call_tool(name, args))
    assert not result.is_error, f"Tool {name} devolvió un error inesperado de protocolo"
    assert len(result.content) > 0, "La herramienta devolvió contenido vacío"
    return json.loads(result.content[0].text)


# ── 1. Inicialización y registro de herramientas ─────────────────────────────

def test_mcp_server_initialization_and_tool_registration():
    server = create_news_mcp_server()
    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}

    expected_tools = {
        "ver_noticias",
        "resumen_noticias",
        "comparar_noticias",
        "listar_tickers_noticias",
    }
    assert expected_tools.issubset(tool_names), f"Faltan herramientas: {expected_tools - tool_names}"
    for tool in tools:
        assert tool.description, f"La herramienta {tool.name} carece de descripción"


# ── 2. Herramienta ver_noticias ──────────────────────────────────────────────

def test_ver_noticias_valid_ticker():
    server = create_news_mcp_server()
    res = _call_tool(server, "ver_noticias", {"ticker": "AC"})

    assert res["ok"] is True
    assert res["status"] in {"ok", "stale", "no_recent_articles"}
    assert res["ticker"] == "AC"
    assert "fecha_utc" in res
    assert "metricas" in res

    metricas = res["metricas"]
    expected_metrics = [
        "sentimiento_general",
        "volumen_informativo_ponderado",
        "relevancia_promedio",
        "cantidad_articulos",
        "cambio_sentimiento",
        "dispersion_sentimiento",
        "sentimiento_esg",
    ]
    for m in expected_metrics:
        assert m in metricas, f"Falta la métrica {m} en la respuesta de ver_noticias"

    assert "fuentes" in res
    assert isinstance(res["fuentes"], list)
    assert "resumen_semantico" in res


def test_ver_noticias_with_historical_date():
    server = create_news_mcp_server()
    res = _call_tool(server, "ver_noticias", {"ticker": "BOLSAA", "fecha": "2026-06-30"})

    assert res["ok"] is True
    assert res["ticker"] == "BOLSAA"
    assert res["fecha_utc"] == "2026-06-30"
    assert res["metricas"]["cantidad_articulos"] == 16


def test_ver_noticias_nonexistent_ticker():
    server = create_news_mcp_server()
    res = _call_tool(server, "ver_noticias", {"ticker": "INEXISTENTE999"})

    assert res["ok"] is False
    assert res["status"] == "not_found"
    assert "error" in res


def test_ver_noticias_invalid_date():
    server = create_news_mcp_server()
    res = _call_tool(server, "ver_noticias", {"ticker": "AC", "fecha": "fecha-invalida"})

    assert res["ok"] is False
    assert res["status"] == "invalid_input"
    assert "error" in res


def test_ver_noticias_empty_ticker():
    server = create_news_mcp_server()
    res = _call_tool(server, "ver_noticias", {"ticker": ""})

    assert res["ok"] is False
    assert res["status"] == "invalid_input"


# ── 3. Herramienta resumen_noticias ──────────────────────────────────────────

def test_resumen_noticias_evolution():
    server = create_news_mcp_server()
    res = _call_tool(
        server,
        "resumen_noticias",
        {"ticker": "BOLSAA", "fecha_inicio": "2026-06-30", "fecha_fin": "2026-10-31"},
    )

    assert res["ok"] is True
    assert res["status"] == "ok"
    assert res["ticker"] == "BOLSAA"
    assert res["periodo"]["dias_con_datos"] > 0
    assert "evolucion_sentimiento" in res
    assert "resumen_volumen" in res
    assert "serie_temporal" in res
    assert isinstance(res["serie_temporal"], list)
    assert len(res["serie_temporal"]) == res["periodo"]["dias_con_datos"]


def test_resumen_noticias_inverted_dates():
    server = create_news_mcp_server()
    res = _call_tool(
        server,
        "resumen_noticias",
        {"ticker": "BOLSAA", "fecha_inicio": "2026-10-31", "fecha_fin": "2026-06-30"},
    )

    assert res["ok"] is False
    assert res["status"] == "invalid_input"


def test_resumen_noticias_default_dates():
    server = create_news_mcp_server()
    res = _call_tool(server, "resumen_noticias", {"ticker": "BOLSAA"})

    assert res["ok"] is True
    assert res["periodo"]["fecha_inicio"] is not None
    assert res["periodo"]["fecha_fin"] is not None


# ── 4. Herramienta comparar_noticias ─────────────────────────────────────────

def test_comparar_noticias_multiple_tickers():
    server = create_news_mcp_server()
    res = _call_tool(server, "comparar_noticias", {"tickers": ["AC", "BOLSAA"]})

    assert res["ok"] is True
    assert res["total_solicitados"] == 2
    assert res["total_encontrados"] == 2
    assert "ranking_por_sentimiento" in res
    assert "ranking_por_volumen" in res
    assert len(res["comparativa"]) == 2


def test_comparar_noticias_comma_separated_and_missing():
    server = create_news_mcp_server()
    res = _call_tool(
        server,
        "comparar_noticias",
        {"tickers": "AC, TICKER_FALSO_123, BOLSAA"},
    )

    assert res["ok"] is True
    assert res["total_solicitados"] == 3
    assert res["total_encontrados"] == 2
    assert res["total_no_disponibles"] == 1
    assert res["no_disponibles"][0]["ticker"] == "TICKER_FALSO_123"


def test_comparar_noticias_empty_list():
    server = create_news_mcp_server()
    res = _call_tool(server, "comparar_noticias", {"tickers": []})

    assert res["ok"] is False
    assert res["status"] == "invalid_input"


# ── 5. Herramienta listar_tickers_noticias ───────────────────────────────────

def test_listar_tickers_noticias():
    server = create_news_mcp_server()
    res = _call_tool(server, "listar_tickers_noticias", {})

    assert res["ok"] is True
    assert res["total_tickers"] > 0
    assert isinstance(res["tickers_disponibles"], list)
    assert "AC" in res["tickers_disponibles"]
    assert "rango_fechas_utc" in res
    assert res["rango_fechas_utc"]["first"] is not None
    assert res["rango_fechas_utc"]["last"] is not None
    assert "fuentes_detectadas" in res


# ── 6. Aislamiento con fixture temporal (lee Gold, no Bronze) ────────────────

def test_server_with_isolated_mock_gold(tmp_path: Path):
    """Comprueba que el servidor opera con un news_root aislado leyendo exclusivamente Gold."""
    gold_dir = tmp_path / "gold"
    silver_dir = tmp_path / "silver"
    gold_dir.mkdir(parents=True)
    silver_dir.mkdir(parents=True)

    today_dt = pd.Timestamp("2026-10-31 00:00:00", tz="UTC")
    row = {
        "date": today_dt,
        "ticker": "TESTCORP",
        "topic": "",
        "as_of": pd.Timestamp("2026-10-31 01:00:00", tz="UTC"),
        "window_days": 7,
        "scoring_method": "keywords",
        "publication_date_missing": 0,
        "article_ids": json.dumps(["art_1", "art_2"]),
        "batch_ids": json.dumps(["batch_1"]),
        "news_sent_score": 0.42,
        "news_count_7d": 1.5,
        "news_sent_delta": 0.1,
        "news_sent_vol": 0.05,
        "news_relevance_avg": 0.95,
        "news_esg_score": 0.0,
        "n_articles": 2,
        "sources": ["test_source"],
    }
    df = pd.DataFrame([row])[GOLD_COLUMNS]
    df.to_parquet(gold_dir / "today_features.parquet", index=False)
    df.to_parquet(gold_dir / "historical_features.parquet", index=False)

    # Silver mínimo requerido para cobertura
    silver_row = {
        "article_id": "art_1",
        "batch_id": "batch_1",
        "bronze_path": "bronze/test.json",
        "source": "test_source",
        "ticker": "TESTCORP",
        "topic": "",
        "title": "Titular de prueba",
        "text": "Texto de prueba",
        "text_kind": "snippet",
        "url": "https://example.com",
        "published_at": today_dt,
        "fetched_at": today_dt,
        "available_at": today_dt,
        "sentiment": 0.42,
        "relevance_score": 0.95,
    }
    silver_df = pd.DataFrame([silver_row])
    silver_df.to_parquet(silver_dir / "articles.parquet", index=False)

    # Servidor apuntando exclusivamente a tmp_path (no existe bronze/ en tmp_path)
    isolated_server = create_news_mcp_server(news_root=tmp_path)
    res = _call_tool(isolated_server, "ver_noticias", {"ticker": "TESTCORP"})

    assert res["ok"] is True
    assert res["ticker"] == "TESTCORP"
    assert res["metricas"]["sentimiento_general"] == 0.42
    assert res["metricas"]["volumen_informativo_ponderado"] == 1.5
    assert res["fuentes"] == ["test_source"]
    # Verificar que no necesitó carpeta bronze
    assert not (tmp_path / "bronze").exists()


# ── 7. Serialización JSON estricta ───────────────────────────────────────────

def test_all_tool_responses_strictly_json_serializable():
    server = create_news_mcp_server()

    # Cada herramienta debe devolver un dict que pase json.dumps sin error de tipos
    calls = [
        ("ver_noticias", {"ticker": "AC"}),
        ("ver_noticias", {"ticker": "NO_EXISTE"}),
        ("resumen_noticias", {"ticker": "BOLSAA", "fecha_inicio": "2026-06-30", "fecha_fin": "2026-10-31"}),
        ("comparar_noticias", {"tickers": ["AC", "BOLSAA"]}),
        ("listar_tickers_noticias", {}),
    ]

    for name, args in calls:
        res = _call_tool(server, name, args)
        serialized = json.dumps(res)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert isinstance(deserialized, dict)

