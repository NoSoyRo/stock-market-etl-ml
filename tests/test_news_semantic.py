import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.news_medallion import GOLD_COLUMNS, SILVER_COLUMNS
from src.semantic.service import NewsSemanticService


NOW = datetime(2026, 10, 30, 18, tzinfo=timezone.utc)
TODAY = "2026-10-30"


def gold_row(
    date=TODAY,
    *,
    ticker="WALMEX",
    topic="",
    as_of=None,
    window_days=7,
    scoring_method="keywords",
    n_articles=1,
    article_ids=("article-1",),
    batch_ids=("batch-1",),
    sources=("local",),
    news_sent_score=0.25,
    news_count_7d=0.69,
    news_sent_delta=0.1,
    news_sent_vol=0.2,
    news_relevance_avg=0.8,
    news_esg_score=0.0,
):
    return {
        "date": pd.Timestamp(f"{date}T00:00:00Z"),
        "ticker": ticker,
        "topic": topic,
        "as_of": pd.Timestamp(as_of or f"{date}T17:00:00Z"),
        "window_days": window_days,
        "scoring_method": scoring_method,
        "publication_date_missing": 0,
        "article_ids": json.dumps(list(article_ids)),
        "batch_ids": json.dumps(list(batch_ids)),
        "news_sent_score": news_sent_score,
        "news_count_7d": news_count_7d,
        "news_sent_delta": news_sent_delta,
        "news_sent_vol": news_sent_vol,
        "news_relevance_avg": news_relevance_avg,
        "news_esg_score": news_esg_score,
        "n_articles": n_articles,
        "sources": list(sources),
    }


def silver_row(
    article_id="article-1",
    *,
    batch_id="batch-1",
    ticker="WALMEX",
    topic="",
    fetched_at="2026-10-30T11:00:00Z",
    available_at="2026-10-30T11:00:00Z",
    published_at="2026-10-30T10:00:00Z",
    title="Walmex anuncia crecimiento",
    text="",
    sentiment=None,
):
    return {
        "article_id": article_id,
        "batch_id": batch_id,
        "bronze_path": f"bronze/test/{batch_id}.json",
        "source": "local",
        "ticker": ticker,
        "topic": topic,
        "title": title,
        "text": text or None,
        "text_kind": "headline" if not text else "excerpt",
        "url": f"https://example.test/{article_id}",
        "published_at": None if published_at is None else pd.Timestamp(published_at),
        "fetched_at": pd.Timestamp(fetched_at),
        "available_at": pd.Timestamp(available_at),
        "sentiment": sentiment,
        "relevance_score": 1.0,
    }


def write_gold(root, rows, dataset="historical_features.parquet"):
    path = root / "gold" / dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=GOLD_COLUMNS).to_parquet(path, index=False)
    if dataset == "historical_features.parquet":
        today_path = root / "gold" / "today_features.parquet"
        pd.DataFrame(rows, columns=GOLD_COLUMNS).to_parquet(today_path, index=False)
    return path


def write_silver(root, rows):
    path = root / "silver" / "articles.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SILVER_COLUMNS).to_parquet(path, index=False)
    return path


def service(root):
    return NewsSemanticService(root, clock=lambda: NOW)


def test_company_query_filters_exact_ticker_and_returns_business_context(tmp_path):
    write_gold(
        tmp_path,
        [
            gold_row(),
            gold_row(ticker="", topic="trump", article_ids=("topic-article",)),
            gold_row(ticker="", topic="market", article_ids=("market-article",)),
        ],
    )
    result = service(tmp_path).consultar_empresa(" walmex ")
    assert result["status"] == "ok"
    assert result["query"]["entity_id"] == "WALMEX"
    assert result["data"]["metrics"]["sentimiento_general"] == 0.25
    assert result["data"]["metrics"]["cantidad_articulos"] == 1
    assert result["data"]["window_days"] == 7
    assert result["data"]["esg_coverage"] == "unknown"


def test_topic_query_does_not_mix_market_or_company_rows(tmp_path):
    write_gold(
        tmp_path,
        [
            gold_row(ticker="", topic="trump", article_ids=("trump-article",)),
            gold_row(ticker="", topic="market", article_ids=("market-article",)),
            gold_row(ticker="WALMEX", topic="", article_ids=("company-article",)),
        ],
    )
    result = service(tmp_path).consultar_tema(" Trump ")
    assert result["status"] == "ok"
    assert result["query"]["entity_id"] == "trump"
    assert result["provenance"]["key"] == {
        "date": TODAY,
        "ticker": "",
        "topic": "trump",
    }


def test_today_table_for_previous_utc_date_is_stale(tmp_path):
    write_gold(tmp_path, [gold_row(date="2026-10-29")])
    result = service(tmp_path).consultar_empresa("WALMEX")
    assert result["status"] == "stale"
    assert result["data"]["available_dates_utc"] == ["2026-10-29"]


def test_missing_historical_day_is_not_replaced_by_latest(tmp_path):
    write_gold(tmp_path, [gold_row(date="2026-10-29")])
    result = service(tmp_path).consultar_empresa("WALMEX", fecha=TODAY)
    assert result["status"] == "not_found"


def test_requested_cutoff_does_not_use_later_gold_snapshot(tmp_path):
    write_gold(tmp_path, [gold_row(as_of="2026-10-30T17:00:00Z")])
    result = service(tmp_path).consultar_empresa(
        "WALMEX", fecha=TODAY, corte="2026-10-30T16:00:00Z"
    )
    assert result["status"] == "not_found"


def test_no_recent_articles_are_not_interpreted_as_neutral(tmp_path):
    write_gold(
        tmp_path,
        [
            gold_row(
                n_articles=0,
                article_ids=("older-article",),
                news_sent_score=0.0,
                news_sent_vol=0.0,
                news_relevance_avg=0.0,
            )
        ],
    )
    result = service(tmp_path).consultar_empresa("WALMEX")
    assert result["status"] == "no_recent_articles"
    assert result["data"]["metrics"]["sentimiento_general"] is None
    assert result["data"]["metrics"]["dispersion_sentimiento"] is None
    assert result["data"]["metrics"]["relevancia_promedio"] is None
    assert result["data"]["metrics"]["cantidad_articulos"] == 0


def test_evidence_keeps_null_summary_and_distinguishes_null_from_neutral_sentiment(tmp_path):
    rows = [
        silver_row("article-1", sentiment=None),
        silver_row(
            "article-2",
            batch_id="batch-2",
            title="Mercado en equilibrio",
            sentiment=0.0,
        ),
    ]
    write_gold(
        tmp_path,
        [
            gold_row(
                article_ids=("article-1", "article-2"),
                batch_ids=("batch-1", "batch-2"),
                n_articles=2,
            )
        ],
    )
    write_silver(tmp_path, rows)
    result = service(tmp_path).consultar_evidencia("ticker", "WALMEX", TODAY, limit=10)
    assert result["status"] == "ok"
    by_id = {article["article_id"]: article for article in result["data"]["articles"]}
    assert by_id["article-1"]["title"] == "Walmex anuncia crecimiento"
    assert by_id["article-1"]["summary"] is None
    assert by_id["article-1"]["sentiment"] is None
    assert by_id["article-2"]["sentiment"] == 0.0


def test_evidence_selects_latest_available_observation_and_excludes_after_cutoff(tmp_path):
    write_gold(
        tmp_path,
        [gold_row(article_ids=("article-1",), batch_ids=("batch-1",))],
    )
    write_silver(
        tmp_path,
        [
            silver_row("article-1", batch_id="batch-old", fetched_at="2026-10-30T09:00:00Z"),
            silver_row("article-1", batch_id="batch-1", fetched_at="2026-10-30T11:00:00Z"),
            silver_row(
                "article-1",
                batch_id="batch-future",
                fetched_at="2026-10-30T18:00:00Z",
                available_at="2026-10-30T18:00:00Z",
            ),
        ],
    )
    result = service(tmp_path).consultar_evidencia("ticker", "WALMEX", TODAY)
    assert result["status"] == "ok"
    assert [item["batch_id"] for item in result["data"]["articles"]] == ["batch-1"]


def test_comparison_rejects_different_windows_or_scoring_methods(tmp_path):
    write_gold(
        tmp_path,
        [
            gold_row(date="2026-10-29", as_of="2026-10-29T17:00:00Z"),
            gold_row(
                date=TODAY,
                window_days=14,
                as_of="2026-10-30T17:00:00Z",
            ),
        ],
    )
    result = service(tmp_path).comparar_fechas("ticker", "WALMEX", "2026-10-29", TODAY)
    assert result["status"] == "incompatible_metrics"

    write_gold(
        tmp_path,
        [
            gold_row(date="2026-10-29", as_of="2026-10-29T17:00:00Z"),
            gold_row(date=TODAY, scoring_method="other"),
        ],
    )
    result = service(tmp_path).comparar_fechas("ticker", "WALMEX", "2026-10-29", TODAY)
    assert result["status"] == "incompatible_metrics"


def test_malformed_gold_ids_are_integrity_errors_without_evaluation(tmp_path):
    row = gold_row()
    row["article_ids"] = "__import__('os').system('echo unsafe')"
    write_gold(tmp_path, [row])
    result = service(tmp_path).consultar_empresa("WALMEX")
    assert result["status"] == "data_integrity_error"


def test_duplicate_gold_key_is_an_integrity_error(tmp_path):
    write_gold(tmp_path, [gold_row(), gold_row()])
    result = service(tmp_path).consultar_empresa("WALMEX")
    assert result["status"] == "data_integrity_error"


def test_incomplete_evidence_is_reported(tmp_path):
    write_gold(tmp_path, [gold_row(article_ids=("missing",), batch_ids=("missing-batch",))])
    write_silver(tmp_path, [])
    result = service(tmp_path).consultar_evidencia("ticker", "WALMEX", TODAY)
    assert result["status"] == "data_integrity_error"
    assert result["data"]["integrity"]["complete"] is False
    assert result["data"]["integrity"]["missing_article_ids"] == ["missing"]
    assert result["data"]["integrity"]["missing_batch_ids"] == ["missing-batch"]


def test_missing_or_unreadable_parquet_is_data_unavailable(tmp_path):
    missing = service(tmp_path).consultar_empresa("WALMEX")
    assert missing["status"] == "data_unavailable"
    path = tmp_path / "gold" / "today_features.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not parquet")
    unreadable = service(tmp_path).consultar_empresa("WALMEX")
    assert unreadable["status"] == "data_unavailable"


def test_history_preserves_overlapping_daily_counts_without_summing(tmp_path):
    write_gold(
        tmp_path,
        [
            gold_row(date="2026-10-29", as_of="2026-10-29T17:00:00Z", n_articles=4),
            gold_row(date=TODAY, as_of="2026-10-30T17:00:00Z", n_articles=5),
        ],
    )
    result = service(tmp_path).consultar_historial(
        "ticker", "WALMEX", "2026-10-29", TODAY
    )
    assert result["status"] == "ok"
    assert [item["metrics"]["cantidad_articulos"] for item in result["data"]["results"]] == [4, 5]
    assert "total_articles" not in result["data"]


def test_regular_query_does_not_write_to_news_files(tmp_path):
    path = write_gold(tmp_path, [gold_row()])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = service(tmp_path).consultar_empresa("WALMEX")
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["status"] == "ok"
    assert before == after


def test_invalid_inputs_and_window_mismatch_are_explicit(tmp_path):
    write_gold(tmp_path, [gold_row()])
    svc = service(tmp_path)
    assert svc.consultar_empresa("../WALMEX")["status"] == "invalid_input"
    assert svc.consultar_empresa("WALMEX", window_days=30)["status"] == "incompatible_metrics"
    assert svc.consultar_historial(
        "ticker", "WALMEX", TODAY, "2026-10-01"
    )["status"] == "invalid_input"


def test_stale_calculation_age_and_missing_requested_history_window(tmp_path):
    write_gold(tmp_path, [gold_row()])
    stale = NewsSemanticService(
        tmp_path, clock=lambda: NOW, max_calculation_age_hours=0.5
    ).consultar_empresa("WALMEX")
    assert stale["status"] == "stale"

    history = service(tmp_path).consultar_historial(
        "ticker", "WALMEX", TODAY, TODAY, window_days=14
    )
    assert history["status"] == "incompatible_metrics"
    assert history["data"]["available_window_days"] == [7]


def test_evidence_pagination_is_stable_and_applies_limit(tmp_path):
    articles = [silver_row(f"article-{i}", batch_id=f"batch-{i}") for i in range(3)]
    write_gold(
        tmp_path,
        [
            gold_row(
                n_articles=3,
                article_ids=tuple(f"article-{i}" for i in range(3)),
                batch_ids=tuple(f"batch-{i}" for i in range(3)),
            )
        ],
    )
    write_silver(tmp_path, articles)
    first = service(tmp_path).consultar_evidencia(
        "ticker", "WALMEX", TODAY, limit=1, offset=0
    )
    second = service(tmp_path).consultar_evidencia(
        "ticker", "WALMEX", TODAY, limit=1, offset=1
    )
    assert first["data"]["total"] == 3
    assert first["data"]["articles"][0]["article_id"] != second["data"]["articles"][0]["article_id"]


def test_cli_prints_json_and_returns_business_absence_successfully(tmp_path):
    write_gold(tmp_path, [])
    script = Path(__file__).resolve().parents[1] / "scripts" / "query_news_semantic.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--news-root",
            str(tmp_path),
            "empresa",
            "--ticker",
            "WALMEX",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "not_found"


def test_agent_semantic_mode_is_opt_in_and_can_attach_evidence(monkeypatch):
    class FakeService:
        def __init__(self):
            pass

        def consultar_empresa(self, ticker):
            return {"status": "ok", "data": {"date_utc": TODAY}, "ticker": ticker}

        def consultar_evidencia(self, entity_type, entity_id, fecha, *, periodo):
            return {
                "status": "ok",
                "query": {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "date": fecha,
                    "period": periodo,
                },
            }

    monkeypatch.setattr("src.semantic.service.NewsSemanticService", FakeService)
    from src.agent.tools import AgenteNoticias

    result = AgenteNoticias().observe("WALMEX", semantica=True, con_evidencia=True)
    assert result["api_version"] == 2
    assert result["fuente"] == "news_semantic_gold"
    assert result["evidencia"]["query"]["period"] == "reciente"

    separate = AgenteNoticias().observe("WALMEX", semantica=True, en_vivo=True)
    assert separate["status"] == "invalid_input"
