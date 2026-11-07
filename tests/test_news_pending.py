"""Un 429 de GDELT no es "sin resultados": el manifiesto debe decir `pending`, no `empty`."""
import json
from datetime import datetime, timezone

import src.ml.news_history as nh


def _rango():
    return (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 20, tzinfo=timezone.utc),
    )


def test_rate_limit_llena_pending_out(monkeypatch):
    monkeypatch.setattr(nh, "_load_chunk_cache", lambda p: None)
    monkeypatch.setattr(nh, "fetch_gdelt_range", lambda *a, **k: ([], False))
    pending: list = []
    start, end = _rango()
    arts = nh.fetch_ticker_gdelt_history(
        "WALMEX", start, end, verbose=False, pending_out=pending,
    )
    assert arts == []
    assert len(pending) > 0


def test_sin_resultados_reales_no_es_pending(monkeypatch):
    monkeypatch.setattr(nh, "_load_chunk_cache", lambda p: None)
    monkeypatch.setattr(nh, "_save_chunk_cache", lambda p, raw: None)
    monkeypatch.setattr(nh, "fetch_gdelt_range", lambda *a, **k: ([], True))
    pending: list = []
    start, end = _rango()
    nh.fetch_ticker_gdelt_history("WALMEX", start, end, verbose=False, pending_out=pending)
    assert pending == []


def test_cache_only_sin_cache_es_pending(monkeypatch):
    monkeypatch.setattr(nh, "_load_chunk_cache", lambda p: None)
    pending: list = []
    start, end = _rango()
    nh.fetch_ticker_gdelt_history(
        "WALMEX", start, end, verbose=False, cache_only=True, pending_out=pending,
    )
    assert len(pending) > 0


def _panel_con_fetch(monkeypatch, tmp_path, fetch):
    monkeypatch.setattr(nh, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(nh, "ARTICLES_PATH", tmp_path / "articles.parquet")
    monkeypatch.setattr(nh, "DAILY_PATH", tmp_path / "daily.parquet")
    monkeypatch.setattr(nh, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(nh, "_load_chunk_cache", lambda p: None)
    monkeypatch.setattr(nh, "fetch_ticker_gdelt_history", fetch)
    nh.build_history_panel({"WALMEX": "WALMEX.MX"}, "2024-01-01", "2024-01-20", verbose=False)
    return json.loads((tmp_path / "manifest.json").read_text())


def test_manifiesto_marca_pending_si_hubo_429(monkeypatch, tmp_path):
    def fetch(name, s, e, *, pending_out=None, **k):
        pending_out.append((s, e))
        return []

    manifest = _panel_con_fetch(monkeypatch, tmp_path, fetch)
    assert list(manifest.values()) == ["pending"]


def test_manifiesto_marca_empty_si_gdelt_no_tiene_nada(monkeypatch, tmp_path):
    manifest = _panel_con_fetch(monkeypatch, tmp_path, lambda *a, **k: [])
    assert list(manifest.values()) == ["empty"]
