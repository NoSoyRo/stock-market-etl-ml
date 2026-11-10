"""Pruebas para el Golden diario de noticias (lab.news_feature_daily) y lab_store."""
from __future__ import annotations

import importlib.util
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_lab_store():
    """Carga src/db/lab_store.py sin requerir driver pyodbc."""
    path = ROOT / "src" / "db" / "lab_store.py"
    spec = importlib.util.spec_from_file_location("src.db.lab_store", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 1. Contrato de esquema SQL ────────────────────────────────────────────────

def test_sql_schema_defines_news_feature_daily_table():
    sql_path = ROOT / "sql" / "03_lab_schema.sql"
    content = sql_path.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS lab.news_feature_daily" in content
    # Comprobar columnas mínimas exigidas
    expected_columns = [
        "ticker",
        "trade_date",
        "universe",
        "news_sent_score",
        "news_count_7d",
        "news_sent_delta",
        "news_sent_vol",
        "news_relevance_avg",
        "news_esg_score",
        "n_articles",
        "sources",
        "loaded_at",
    ]
    for col in expected_columns:
        assert col in content, f"Falta la columna {col} en sql/03_lab_schema.sql"

    # Comprobar llave primaria compuesta (ticker, trade_date)
    assert "PRIMARY KEY (ticker, trade_date)" in content
    # Comprobar índice por fecha
    assert "CREATE INDEX IF NOT EXISTS news_feature_daily_date_idx" in content


# ── 2. Contrato de campos en lab_store ────────────────────────────────────────

def test_lab_store_defines_news_feature_fields_and_function():
    lab_store = _load_lab_store()

    assert hasattr(lab_store, "_NEWS_FEATURE_FIELDS")
    expected_fields = (
        "ticker",
        "trade_date",
        "universe",
        "news_sent_score",
        "news_count_7d",
        "news_sent_delta",
        "news_sent_vol",
        "news_relevance_avg",
        "news_esg_score",
        "n_articles",
        "sources",
        "loaded_at",
    )
    assert lab_store._NEWS_FEATURE_FIELDS == expected_fields
    assert hasattr(lab_store, "upsert_news_features")
    assert callable(lab_store.upsert_news_features)


# ── 3. Cláusula SQL de UPSERT idempotente ─────────────────────────────────────

def test_lab_store_sql_has_on_conflict_update():
    lab_store = _load_lab_store()

    sql = lab_store._NEWS_FEATURE_SQL
    assert "INSERT INTO lab.news_feature_daily" in sql
    assert "ON CONFLICT (ticker, trade_date) DO UPDATE SET" in sql
    assert "universe = EXCLUDED.universe" in sql
    assert "news_sent_score = EXCLUDED.news_sent_score" in sql
    assert "news_count_7d = EXCLUDED.news_count_7d" in sql
    assert "loaded_at = EXCLUDED.loaded_at" in sql


# ── 4. Validación de campos obligatorios ───────────────────────────────────────

def test_upsert_news_features_requires_ticker_and_date():
    lab_store = _load_lab_store()

    with pytest.raises(ValueError, match="upsert_news_features exige ticker"):
        lab_store.upsert_news_features([{"trade_date": "2026-10-24"}])

    with pytest.raises(ValueError, match="upsert_news_features exige trade_date"):
        lab_store.upsert_news_features([{"ticker": "WALMEX"}])


# ── 5. Transformación y normalización de datos ────────────────────────────────

def test_upsert_news_features_normalizes_payload():
    lab_store = _load_lab_store()

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    rows = [
        {
            "ticker": "WALMEX",
            "trade_date": datetime(2026, 10, 24, 18, 30),
            "universe": "bmv",
            "news_sent_score": "0.45",
            "news_count_7d": 1.25,
            "news_sent_delta": 0.1,
            "news_sent_vol": 0.05,
            "news_relevance_avg": 0.95,
            "news_esg_score": 0.0,
            "n_articles": "5",
            "sources": ["google_news", "expansion"],
        }
    ]

    count = lab_store.upsert_news_features(rows, conn=mock_conn)
    assert count == 1
    mock_cursor.executemany.assert_called_once()
    sql_called, payload = mock_cursor.executemany.call_args[0]
    assert sql_called == lab_store._NEWS_FEATURE_SQL
    item = payload[0]

    assert item["ticker"] == "WALMEX"
    assert item["trade_date"] == date(2026, 10, 24)
    assert item["universe"] == "bmv"
    assert item["news_sent_score"] == 0.45
    assert item["news_count_7d"] == 1.25
    assert item["n_articles"] == 5
    assert item["sources"] == "expansion,google_news"


def test_upsert_news_features_empty_returns_zero():
    lab_store = _load_lab_store()
    mock_conn = MagicMock()
    assert lab_store.upsert_news_features([], conn=mock_conn) == 0
    mock_conn.cursor.assert_not_called()


# ── 6. Integración con scripts/gcp_daily_job.py ───────────────────────────────

def test_gcp_daily_job_write_news_features():
    from scripts.gcp_daily_job import _write_news_features

    mock_store = MagicMock()
    mock_store.upsert_news_features.return_value = 3

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("scripts.gcp_daily_job._import_lab_store", lambda: mock_store)
        mp.setattr("scripts.gcp_daily_job._db_ready", lambda s: True)

        res = _write_news_features([{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}])
        assert res == 3
        mock_store.upsert_news_features.assert_called_once()


def test_gcp_daily_job_fetch_news_feature_rows_from_payload(tmp_path, monkeypatch):
    import json
    from scripts.gcp_daily_job import _fetch_news_feature_rows

    dummy_data = [
        {
            "ticker": "WALMEX",
            "news_sent_score": 0.25,
            "news_count_7d": 1.5,
            "news_sent_delta": 0.1,
            "news_sent_vol": 0.2,
            "news_relevance_avg": 0.9,
            "news_esg_score": 0.0,
            "n_articles": 3,
            "sources": ["google_news", "expansion"],
        },
        {
            "ticker": "ALSEA",
            "news_sent_score": 0.8,
            "news_count_7d": 2.1,
            "news_sent_delta": 0.5,
            "news_sent_vol": 0.1,
            "news_relevance_avg": 1.0,
            "news_esg_score": 0.5,
            "n_articles": 4,
            "sources": "expansion",
        },
    ]

    session = date(2026, 10, 24)
    cache_file = tmp_path / "2026-10-24.json"
    cache_file.write_text(json.dumps(dummy_data), encoding="utf-8")

    monkeypatch.setattr("scripts.gcp_daily_job.ROOT", tmp_path.parent)
    news_dir = tmp_path.parent / "data" / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    (news_dir / "2026-10-24.json").write_text(json.dumps(dummy_data), encoding="utf-8")

    rows = _fetch_news_feature_rows(session)
    assert len(rows) == 2
    walmex = next(r for r in rows if r["ticker"] == "WALMEX")
    assert walmex["trade_date"] == session
    assert walmex["universe"] == "bmv"
    assert walmex["news_sent_score"] == 0.25
    assert walmex["news_count_7d"] == 1.5
    assert walmex["n_articles"] == 3
    assert walmex["sources"] == "expansion,google_news"
