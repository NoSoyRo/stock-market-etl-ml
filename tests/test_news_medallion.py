from pathlib import Path
import json
import pandas as pd

from src.data.news_medallion import ingest, normalize, build_gold
from scripts.build_news_medallion import migrate


def article(**updates):
    return dict(title='Walmex reporta crecimiento', ticker='WALMEX',
                url='https://example.com/1', published_at='2026-10-20T10:00Z', **updates)


def test_captures_are_immutable_and_replay_is_idempotent(tmp_path):
    a = ingest('google_news', [article(), article()], root=tmp_path, fetched_at='2026-10-20T12:00Z')
    original = a.read_bytes()
    b = ingest('google_news', [article()], root=tmp_path, fetched_at='2026-10-20T13:00Z')
    assert a != b and a.read_bytes() == original
    normalize(a, root=tmp_path)
    silver = pd.read_parquet(tmp_path / 'silver/articles.parquet')
    assert len(silver) == 2  # One observation per capture, duplicates removed within each capture.
    assert silver.article_id.nunique() == 1
    gold = build_gold(root=tmp_path, as_of='2026-10-20T14:00Z')
    assert gold.iloc[0].n_articles == 1
    assert len(json.loads(gold.iloc[0].article_ids)) == 1
    assert set(json.loads(gold.iloc[0].batch_ids)) <= set(silver.batch_id)
    assert all((tmp_path / relative).exists() for relative in silver.bronze_path)


def test_cutoff_excludes_late_discovery_and_future_publication(tmp_path):
    ingest('gdelt', [article()], root=tmp_path, fetched_at='2026-10-22T12:00Z')
    assert build_gold(root=tmp_path, as_of='2026-10-21T12:00Z').empty
    assert len(build_gold(root=tmp_path, as_of='2026-10-22T13:00Z')) == 1
    ingest('gdelt', [dict(article(), url='https://example.com/future', published_at='2026-11-01T12:00Z')],
           root=tmp_path, fetched_at='2026-10-22T12:00Z')
    assert build_gold(root=tmp_path, as_of='2026-10-22T13:00Z').iloc[0].n_articles == 1


def test_trump_stays_thematic_without_tickers(tmp_path):
    ingest('trump', [{'titulo': 'Trump anuncia aranceles', 'texto': 'Trump anuncia aranceles',
                      'fecha': 'Tue, 20 Oct 2026 10:00:00 GMT', 'url': 'https://example.com/trump'}],
           root=tmp_path, fetched_at='2026-10-20T12:00Z')
    silver = pd.read_parquet(tmp_path / 'silver/articles.parquet')
    assert silver.iloc[0].ticker == '' and silver.iloc[0].topic == 'trump'
    assert silver.iloc[0].text_kind == 'headline'
    result = build_gold(root=tmp_path, as_of='2026-10-20T13:00Z')
    assert result.iloc[0].topic == 'trump' and result.iloc[0].n_articles == 1


def test_databursatil_does_not_invent_publication_dates(tmp_path):
    ingest('databursatil', {'noticias': [{'n': 'Walmex crece', 'f': 'https://example.com/w'}],
                           'cables': {'top': [{'t': 'Mercado general', 'f': 'https://example.com/m'}]}},
           tickers=['WALMEX'], root=tmp_path, fetched_at='2026-10-20T12:00Z')
    silver = pd.read_parquet(tmp_path / 'silver/articles.parquet')
    assert silver.published_at.isna().all()
    assert set(silver.ticker) == {'', 'WALMEX'}
    result = build_gold(root=tmp_path, as_of='2026-10-20T13:00Z')
    assert result.publication_date_missing.sum() == 2


def test_migration_preserves_legacy_files_and_can_repeat(tmp_path):
    csv = tmp_path / 'noticias_trump.csv'
    csv.write_text('titulo,fecha,url,fuente,texto\nTrump,,,Google News (Trump),Trump\n')
    before = csv.read_bytes()
    assert migrate(tmp_path) == ['noticias_trump.csv']
    assert migrate(tmp_path) == []
    assert csv.read_bytes() == before
    assert len(pd.read_parquet(tmp_path / 'silver/articles.parquet')) == 1


def test_rebuild_keeps_other_days_and_replaces_same_day(tmp_path):
    ingest('gdelt', [article()], root=tmp_path, fetched_at='2026-10-20T12:00Z')
    build_gold(root=tmp_path, as_of='2026-10-20T13:00Z')
    build_gold(root=tmp_path, as_of='2026-10-21T13:00Z')
    build_gold(root=tmp_path, as_of='2026-10-21T14:00Z')
    gold = pd.read_parquet(tmp_path / 'gold/daily_features.parquet')
    assert len(gold) == 2
    assert not gold.duplicated(['date', 'ticker', 'topic']).any()


def test_scraping_captures_before_deduplication(tmp_path, monkeypatch):
    import src.data.news_medallion as lake
    import src.ml.news as news
    original_ingest = lake.ingest
    monkeypatch.setattr(lake, 'ingest', lambda source, payload: original_ingest(source, payload, root=tmp_path))
    monkeypatch.delenv('ALPHA_VANTAGE_API_KEY', raising=False)
    item = news.Article('TEST', 'A headline', '', 'https://example.com/1', 0, 'yfinance')
    monkeypatch.setattr(news, 'fetch_yfinance', lambda *args: [item, item])
    monkeypatch.setattr(news, 'fetch_google_news_rss', lambda *args, **kwargs: [])
    assert len(news._scrape_articles('TEST', 'TEST')) == 1
    bronze = list((tmp_path / 'bronze').glob('*/*/*.json'))
    assert len(json.loads(bronze[0].read_text())['payload']) == 2
    assert len(pd.read_parquet(tmp_path / 'silver/articles.parquet')) == 1


def test_trump_ingestion_keeps_csv_and_populates_lake(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.ingesta_trump as trump
    import src.data.news_medallion as lake
    (tmp_path / 'data/news').mkdir(parents=True)
    original_ingest, original_build = lake.ingest, lake.build_gold
    root = tmp_path / 'data/news'
    monkeypatch.setattr(trump, 'ROOT', tmp_path)
    monkeypatch.setattr(trump.feedparser, 'parse', lambda _: SimpleNamespace(entries=[
        trump.feedparser.FeedParserDict(title='Trump aranceles', published='Tue, 20 Oct 2026 10:00:00 GMT', link='https://example.com/trump', summary='<p>Resumen &amp; detalles</p>')]))
    monkeypatch.setattr(lake, 'ingest', lambda source, payload, **kw: original_ingest(source, payload, root=root, **kw))
    monkeypatch.setattr(lake, 'build_gold', lambda: original_build(root=root))
    trump.buscar_noticias_trump()
    assert pd.read_csv(root / 'noticias_trump.csv').columns.tolist() == ['titulo', 'fecha', 'url', 'fuente']
    silver = pd.read_parquet(root / 'silver/articles.parquet')
    assert silver.iloc[0].text == 'Resumen & detalles'
    assert silver.iloc[0].text_kind == 'excerpt'
    assert (root / 'silver/articles.parquet').exists()
    assert (root / 'gold/daily_features.parquet').exists()


def test_empty_gold_keeps_schema(tmp_path):
    from src.data.news_medallion import GOLD_COLUMNS
    result = build_gold(root=tmp_path, as_of='2026-10-20T12:00Z')
    stored = pd.read_parquet(tmp_path / 'gold/daily_features.parquet')
    assert list(result.columns) == list(stored.columns) == GOLD_COLUMNS
    assert json.loads((tmp_path / 'gold/latest.json').read_text()) == []


def test_html_cleanup_and_replay_preserve_bronze(tmp_path):
    from src.data.news_medallion import clean_news_text
    assert clean_news_text('<p>Mercado&nbsp;al alza</p><p>A &amp; B</p><script>ignore()</script>') == 'Mercado al alza A & B'
    path = ingest('trump', [{'titulo': 'Trump &amp; comercio', 'texto': '<p>Subida &amp; ganancias</p>',
                            'fecha': 'Tue, 20 Oct 2026 10:00:00 GMT', 'url': 'https://example.com/trump'}],
                  root=tmp_path, fetched_at='2026-10-20T12:00Z')
    original = path.read_bytes()
    normalize(path, root=tmp_path)
    assert path.read_bytes() == original
    row = pd.read_parquet(tmp_path / 'silver/articles.parquet').iloc[0]
    assert row.title == 'Trump & comercio' and row.text == 'Subida & ganancias'
    assert row.text_kind == 'excerpt'
    gold = build_gold(root=tmp_path, as_of='2026-10-20T13:00Z')
    assert gold.iloc[0].news_sent_score > 0


def test_four_column_csv_falls_back_to_headline(tmp_path):
    path = ingest('trump', [{'titulo': 'Trump aranceles', 'fecha': '',
                            'url': 'https://example.com/trump', 'fuente': 'Google News (Trump)'}], root=tmp_path)
    row = normalize(path, root=tmp_path).iloc[0]
    assert pd.isna(row.text) and row.text_kind == 'headline'


def test_legacy_integration_uses_silver_excerpt_with_four_column_csv(tmp_path, monkeypatch):
    import src.etl.shared.integrar_tfidf as integration
    root = tmp_path / 'data/news'
    root.mkdir(parents=True)
    pd.DataFrame([{'titulo': 'Trump', 'fecha': 'Tue, 20 Oct 2026 10:00:00 GMT',
                   'url': 'https://example.com/trump', 'fuente': 'Google News (Trump)'}]).to_csv(root / 'noticias_trump.csv', index=False)
    ingest('trump', [{'titulo': 'Trump', 'texto': 'Subida de ganancias',
                      'url': 'https://example.com/trump'}], root=root, fetched_at='2026-10-20T12:00Z')
    monkeypatch.setattr(integration, 'ROOT', tmp_path)
    monkeypatch.setattr(integration, '_scrape_articles', lambda *args: [])
    seen = []
    def features(articles):
        seen.extend(articles)
        return {'news_sent_score': 1, 'n_articles': 1, 'sources': ['Trump_RSS']}
    monkeypatch.setattr(integration, '_compute_features_from_articles', features)
    integration.integrar_noticias()
    assert seen[0].summary == 'Subida de ganancias'
    assert (tmp_path / 'data/golden/walmex_integrado.json').exists()


def test_bronze_parquet_retains_raw_values_and_duplicates(tmp_path):
    from src.data.news_medallion import capture, rebuild_bronze
    raw = {'titulo': '  Trump &amp; comercio  ', 'texto': '<p>Texto&nbsp;original</p>',
           'fecha': 'not a date', 'url': 'https://example.com/trump', 'custom': {'x': 3}}
    path = capture('trump', [raw, raw, {'titulo': ''}], root=tmp_path,
                   fetched_at='2026-10-20T12:00Z')
    before = path.read_bytes()
    bronze = pd.read_parquet(tmp_path / 'bronze/news.parquet')
    assert len(bronze) == 3  # No article deduplication or title filtering in bronze.
    assert bronze.record_id.nunique() == 3
    assert bronze.iloc[0].title_raw == raw['titulo']
    assert bronze.iloc[0].text_raw == raw['texto']
    assert bronze.iloc[0].published_at_raw == 'not a date'
    assert json.loads(bronze.iloc[0].raw_payload) == raw
    batch = bronze.iloc[0].batch_id
    assert set(bronze.batch_id) == {batch}
    normalize(path, root=tmp_path)
    silver = pd.read_parquet(tmp_path / 'silver/articles.parquet')
    assert len(silver) == 1 and silver.iloc[0].text == 'Texto original'
    pd.testing.assert_frame_equal(bronze, rebuild_bronze(root=tmp_path))
    assert path.read_bytes() == before


def test_bronze_databursatil_grain_is_before_ticker_expansion(tmp_path):
    payload = {'noticias': [{'n': 'Walmex y Alsea', 'f': 'https://example.com/n'}],
               'cables': {'top': [{'t': 'Mercado', 'f': 'https://example.com/c'}], 'flash': []}}
    ingest('databursatil', payload, root=tmp_path, tickers=['WALMEX', 'ALSEA'])
    bronze = pd.read_parquet(tmp_path / 'bronze/news.parquet')
    assert len(bronze) == 2
    assert bronze.record_kind.tolist() == ['noticia', 'cable']
    assert bronze.category.tolist() == ['', 'top']
    assert json.loads(bronze.iloc[1].raw_payload) == payload['cables']['top'][0]
    silver = pd.read_parquet(tmp_path / 'silver/articles.parquet')
    assert len(silver) == 3
    assert set(silver.ticker) == {'WALMEX', 'ALSEA', ''}


def test_bronze_combines_sources_and_preserves_repeated_captures(tmp_path):
    from src.data.news_medallion import capture, rebuild_bronze
    for source in ['trump', 'google_news', 'gdelt', 'expansion', 'elfinanciero']:
        capture(source, [article()], root=tmp_path)
    capture('trump', [article()], root=tmp_path)
    bronze = pd.read_parquet(tmp_path / 'bronze/news.parquet')
    assert len(bronze) == 6
    assert bronze.batch_id.nunique() == 6
    assert set(bronze.source) == {'trump', 'google_news', 'gdelt', 'expansion', 'elfinanciero'}
    assert len(rebuild_bronze(root=tmp_path)) == 6
    assert len(rebuild_bronze(root=tmp_path)) == 6


def test_empty_bronze_has_stable_schema(tmp_path):
    from src.data.news_medallion import BRONZE_COLUMNS, capture, rebuild_bronze
    frame = rebuild_bronze(root=tmp_path)
    assert frame.empty and list(frame.columns) == BRONZE_COLUMNS
    capture('trump', [], root=tmp_path)
    stored = pd.read_parquet(tmp_path / 'bronze/news.parquet')
    pd.testing.assert_frame_equal(stored, frame)


def test_gold_history_and_today_are_separate_and_backfill_preserves_today(tmp_path):
    import src.data.news_medallion as lake
    today = lake.utc().normalize()
    yesterday = today - pd.Timedelta(days=1)
    ingest('gdelt', [dict(article(), published_at=yesterday.isoformat())],
           root=tmp_path, fetched_at=yesterday.isoformat())
    current = build_gold(root=tmp_path, as_of=(today + pd.Timedelta(hours=12)).isoformat())
    stored_today = pd.read_parquet(tmp_path / 'gold/today_features.parquet')
    pd.testing.assert_frame_equal(current, stored_today)
    build_gold(root=tmp_path, as_of=(yesterday + pd.Timedelta(hours=12)).isoformat())
    history = pd.read_parquet(tmp_path / 'gold/historical_features.parquet')
    assert len(history) == 2 and history.date.nunique() == 2
    assert not history.duplicated(['date', 'ticker', 'topic']).any()
    pd.testing.assert_frame_equal(stored_today, pd.read_parquet(tmp_path / 'gold/today_features.parquet'))
    latest = json.loads((tmp_path / 'gold/latest.json').read_text())
    assert len(latest) == len(current)
    assert pd.Timestamp(latest[0]['date']) == today
    pd.testing.assert_frame_equal(history, pd.read_parquet(tmp_path / 'gold/daily_features.parquet'))


def test_gold_today_clears_previous_day_on_rollover(tmp_path, monkeypatch):
    import src.data.news_medallion as lake
    real_utc = lake.utc
    now = [pd.Timestamp('2026-10-20T12:00Z')]
    monkeypatch.setattr(lake, 'utc', lambda value=None: now[0] if value is None else real_utc(value))
    ingest('gdelt', [article()], root=tmp_path, fetched_at='2026-10-20T12:00Z')
    build_gold(root=tmp_path)
    assert len(pd.read_parquet(tmp_path / 'gold/today_features.parquet')) == 1
    now[0] = pd.Timestamp('2026-10-21T12:00Z')
    build_gold(root=tmp_path, as_of='2026-10-20T13:00Z')
    assert pd.read_parquet(tmp_path / 'gold/today_features.parquet').empty
    assert json.loads((tmp_path / 'gold/latest.json').read_text()) == []


def test_gold_history_bootstraps_legacy_file(tmp_path):
    ingest('gdelt', [article()], root=tmp_path, fetched_at='2026-10-20T12:00Z')
    build_gold(root=tmp_path, as_of='2026-10-20T13:00Z')
    (tmp_path / 'gold/historical_features.parquet').unlink()
    build_gold(root=tmp_path, as_of='2026-10-21T13:00Z')
    assert len(pd.read_parquet(tmp_path / 'gold/historical_features.parquet')) == 2


def test_gold_history_reconstruction_is_point_in_time_and_repeatable(tmp_path):
    from src.data.news_medallion import rebuild_gold_history
    ingest('gdelt', [dict(article(), published_at='2026-10-18T12:00Z')],
           root=tmp_path, fetched_at='2026-10-20T12:00Z')
    result = rebuild_gold_history(root=tmp_path, as_of='2026-10-22T12:00Z')
    assert len(result) == 3
    assert result.date.min() == pd.Timestamp('2026-10-20T00:00Z')
    assert result.date.max() == pd.Timestamp('2026-10-22T00:00Z')
    pd.testing.assert_frame_equal(result, rebuild_gold_history(root=tmp_path, as_of='2026-10-22T12:00Z'))


def test_silver_removes_redundancy_and_distinguishes_unscored_from_neutral(tmp_path):
    records = [
        {'title': 'Subida &amp; ganancias', 'summary': '<p>subida &amp; ganancias</p>', 'url': 'u1'},
        {'title': 'Neutral', 'summary': '', 'url': 'u2', 'sentiment': 0},
        {'title': 'Positive', 'summary': 'Un extracto distinto', 'url': 'u3', 'sentiment': 0.5},
        {'title': 'Legacy default', 'url': 'u4', 'sentiment': 0, 'weight': 1, 'av_sentiment': None},
        {'title': 'Provider neutral', 'url': 'u5', 'sentiment': 0, 'weight': 1, 'av_sentiment': 0},
    ]
    path = ingest('google_news', records, root=tmp_path)
    row = pd.read_parquet(tmp_path / 'silver/articles.parquet').set_index('url')
    assert pd.isna(row.loc['u1', 'text']) and pd.isna(row.loc['u1', 'sentiment'])
    assert row.loc['u1', 'text_kind'] == 'headline'
    assert row.loc['u2', 'sentiment'] == 0
    assert row.loc['u3', 'sentiment'] == 0.5
    assert row.loc['u3', 'text'] == 'Un extracto distinto'
    assert pd.isna(row.loc['u4', 'sentiment'])
    assert row.loc['u5', 'sentiment'] == 0
    assert json.loads(path.read_text())['payload'] == records


def test_gold_scores_headline_once_when_silver_text_is_null(tmp_path):
    from src.ml.news import _score_keywords
    ingest('trump', [{'titulo': 'Subida de ganancias', 'texto': 'Subida de ganancias',
                      'fecha': 'Tue, 20 Oct 2026 10:00:00 GMT', 'url': 'u'}],
           root=tmp_path, fetched_at='2026-10-20T12:00Z')
    gold = build_gold(root=tmp_path, as_of='2026-10-20T13:00Z')
    assert gold.iloc[0].news_sent_score == round(_score_keywords('Subida de ganancias '), 4)
    assert gold.iloc[0].n_articles == 1


def test_integration_falls_back_to_title_for_null_silver_text(tmp_path, monkeypatch):
    import src.etl.shared.integrar_tfidf as integration
    root = tmp_path / 'data/news'
    root.mkdir(parents=True)
    pd.DataFrame([{'titulo': 'Solo titular', 'fecha': 'Tue, 20 Oct 2026 10:00:00 GMT',
                   'url': 'u', 'fuente': 'Google News (Trump)'}]).to_csv(root / 'noticias_trump.csv', index=False)
    ingest('trump', [{'titulo': 'Solo titular', 'url': 'u'}], root=root, fetched_at='2026-10-20T12:00Z')
    monkeypatch.setattr(integration, 'ROOT', tmp_path)
    monkeypatch.setattr(integration, '_scrape_articles', lambda *args: [])
    seen = []
    def features(articles):
        seen.extend(articles)
        return {'news_sent_score': 0, 'n_articles': 1, 'sources': ['Trump_RSS']}
    monkeypatch.setattr(integration, '_compute_features_from_articles', features)
    integration.integrar_noticias()
    assert seen[0].title == 'Solo titular' and seen[0].summary == ''
