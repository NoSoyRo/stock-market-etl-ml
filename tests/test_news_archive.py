"""Tests offline del archivador de noticias PIT."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.ml.news_archive import (
    ARCHIVE_COLUMNS,
    append_snapshot,
    classify_manifest,
    coverage_report,
    load_archive,
    rows_from_articles,
)

T0 = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)


def _art(title, url, ts=1782907200, source="google_news"):
    return {"title": title, "url": url, "published_ts": ts, "source": source,
            "summary": "cuerpo que no debe guardarse"}


def test_rows_tienen_fetched_at_y_sin_cuerpo():
    df = rows_from_articles([_art("a", "u1")], "WALMEX", T0)
    assert list(df.columns) == ARCHIVE_COLUMNS
    assert df["fetched_at"].notna().all()
    assert "summary" not in df.columns
    assert df["fetched_at"].iloc[0] == pd.Timestamp(T0)


def test_dedupe_por_url_y_ticker_dentro_del_snapshot():
    df = rows_from_articles([_art("a", "u1"), _art("a dup", "u1")], "WALMEX", T0)
    assert len(df) == 1
    # misma URL en otro ticker es otra fila
    both = pd.concat([df, rows_from_articles([_art("a", "u1")], "ALSEA", T0)])
    assert len(both) == 2


def test_idempotente_y_conserva_primer_fetched_at(tmp_path):
    df0 = rows_from_articles([_art("a", "u1")], "WALMEX", T0)
    assert append_snapshot(df0, tmp_path) == 1
    assert append_snapshot(df0, tmp_path) == 0
    df1 = rows_from_articles([_art("a", "u1"), _art("b", "u2")], "WALMEX", T1)
    assert append_snapshot(df1, tmp_path) == 1
    out = load_archive(tmp_path)
    assert len(out) == 2
    assert out.loc[out["url"] == "u1", "fetched_at"].iloc[0] == pd.Timestamp(T0)


def test_archivo_mensual(tmp_path):
    append_snapshot(rows_from_articles([_art("a", "u1")], "X", T0), tmp_path)
    assert (tmp_path / "2026-07.parquet").exists()


def test_cobertura_por_ticker():
    ts = int(datetime(2026, 7, 1, 15, tzinfo=timezone.utc).timestamp())
    ts2 = int(datetime(2026, 7, 3, 15, tzinfo=timezone.utc).timestamp())
    df = pd.concat([
        rows_from_articles([_art("a", "u1", ts), _art("b", "u2", ts2)], "WALMEX", T1),
        rows_from_articles([], "ALSEA", T1),
    ])
    rep = coverage_report(df, ["WALMEX", "ALSEA"], start="2026-07-01", end="2026-07-04")
    w = rep["per_ticker"]["WALMEX"]
    assert w["n_ticker_days"] == 4 and w["days_with_headline"] == 2
    assert w["pct_days_with_headline"] == 50.0
    assert rep["per_ticker"]["ALSEA"]["days_with_headline"] == 0


def test_clasifica_manifiesto():
    # 3 chunks esperados: 1 ok con artículos, 1 ok vacío, 1 nunca consultado
    st = classify_manifest("empty", n_expected=3, n_cached=2, n_cached_with_articles=1)
    assert st == "parcial_pendiente"
    assert classify_manifest("empty", 3, 0, 0) == "nunca_consultado"
    assert classify_manifest("empty", 3, 3, 0) == "sin_resultados"
    assert classify_manifest("ok", 3, 3, 3) == "ok"
    assert classify_manifest("error:x", 3, 1, 1) == "error"
