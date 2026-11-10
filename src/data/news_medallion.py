"""Local news lake: immutable captures, normalized observations and auditable features."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import fcntl
import json
import os
import re
import uuid

import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'news'
BRONZE_COLUMNS = ['record_id', 'batch_id', 'record_index', 'source', 'capture_source',
                  'fetched_at', 'bronze_path', 'representation', 'origin',
                  'record_kind', 'category', 'source_record_id', 'title_raw',
                  'text_raw', 'url_raw', 'published_at_raw', 'raw_payload']
SILVER_COLUMNS = ['article_id', 'batch_id', 'bronze_path', 'source', 'ticker',
                  'topic', 'title', 'text', 'text_kind', 'url', 'published_at',
                  'fetched_at', 'available_at', 'sentiment', 'relevance_score']
GOLD_COLUMNS = ['date', 'ticker', 'topic', 'as_of', 'window_days', 'scoring_method',
                'publication_date_missing', 'article_ids', 'batch_ids',
                'news_sent_score', 'news_count_7d', 'news_sent_delta', 'news_sent_vol',
                'news_relevance_avg', 'news_esg_score', 'n_articles', 'sources']



class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.hidden += 1
        elif tag in {'br', 'p', 'div', 'li', 'tr'}:
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {'p', 'div', 'li', 'tr'}:
            self.parts.append(' ')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_news_text(value):
    """Normalize RSS HTML, entities and whitespace; preserve plain text content."""
    parser = _PlainText()
    parser.feed(unescape(str(value or '')))
    parser.close()
    return ' '.join(''.join(parser.parts).split())


def _received_sentiment(raw, representation):
    # Older captures serialized Article before scoring, including its default 0.
    # That default was not an assessment. Alpha Vantage can supply a real score.
    if representation == 'normalized_extract' and {'weight', 'av_sentiment'} <= raw.keys():
        value = raw.get('av_sentiment')
    else:
        value = raw.get('sentiment')
        if value is None:
            value = raw.get('av_sentiment')
    if value is None or value == '':
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if pd.notna(value) and -1 <= value <= 1 else None


def utc(value=None):
    if value is None:
        return pd.Timestamp(datetime.now(timezone.utc))
    ts = pd.to_datetime(value, utc=True, errors='coerce')
    if pd.isna(ts):
        raise ValueError(f'Invalid timestamp: {value!r}')
    return ts


def timestamp(value):
    if value is None or value == '' or value == 0:
        return pd.NaT
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit='s', utc=True, errors='coerce')
    return pd.to_datetime(value, utc=True, errors='coerce')


def atomic_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.medallion.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def capture(source, payload, *, root=DEFAULT_ROOT, fetched_at=None,
            representation='normalized_extract', origin=None):
    """Append a capture. Never put request headers, tokens or credential URLs here."""
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', source):
        raise ValueError('Invalid source name')
    root = Path(root)
    fetched = utc(fetched_at)
    batch = uuid.uuid4().hex
    relative = Path('bronze') / source / fetched.strftime('%Y-%m-%d') / (batch + '.json')
    envelope = dict(schema_version=1, batch_id=batch, source=source,
                    fetched_at=fetched.isoformat(), representation=representation,
                    origin=origin, payload=payload)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(envelope, handle, ensure_ascii=False, default=str)
    materialize_bronze(path, root=root)
    return path


def _capture_records(env):
    """Flatten a capture without cleaning, filtering or expanding ticker matches."""
    payload = env['payload']
    if env['source'] == 'databursatil' and isinstance(payload, dict) and (
            'noticias' in payload or 'cables' in payload):
        for raw in payload.get('noticias') or []:
            yield 'noticia', '', raw
        for category, items in (payload.get('cables') or {}).items():
            if isinstance(items, list):
                for raw in items:
                    yield 'cable', category, raw
    else:
        records = payload if isinstance(payload, list) else payload.get('articles', [])
        for raw in records:
            yield 'article', '', raw


def _raw_string(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def materialize_bronze(path, *, root=DEFAULT_ROOT):
    """Upsert a capture into bronze/news.parquet; retain duplicate news records.

    The key is (batch_id, record_index), not an article identity. The raw JSON
    record and the backup capture preserve fields outside the common projection.
    """
    root, path = Path(root), Path(path)
    env = json.loads(path.read_text(encoding='utf-8'))
    rows = []
    for index, (kind, category, raw) in enumerate(_capture_records(env)):
        if not isinstance(raw, dict):
            raise ValueError(f'Non-object news record in {path} at position {index}')
        published = raw.get('published_at', raw.get('published_dt',
                            raw.get('published_ts', raw.get('fecha'))))
        rows.append(dict(
            record_id=sha256(f"{env['batch_id']}:{index}".encode()).hexdigest(),
            batch_id=env['batch_id'], record_index=index,
            source=raw.get('source') or env['source'], capture_source=env['source'],
            fetched_at=utc(env['fetched_at']), bronze_path=str(path.relative_to(root)),
            representation=env.get('representation'), origin=env.get('origin'),
            record_kind=kind, category=category,
            source_record_id=_raw_string(raw.get('id', raw.get('guid'))),
            title_raw=_raw_string(raw.get('title') or raw.get('titulo') or raw.get('n') or raw.get('t')),
            text_raw=_raw_string(raw.get('summary') or raw.get('texto') or raw.get('c')),
            url_raw=_raw_string(raw.get('url') or raw.get('f')),
            published_at_raw=_raw_string(published),
            raw_payload=json.dumps(raw, ensure_ascii=False, default=str)))
    frame = pd.DataFrame(rows, columns=BRONZE_COLUMNS)
    # Explicit dtypes keep empty batches and nullable projections interoperable.
    for col in BRONZE_COLUMNS:
        if col not in {'record_index', 'fetched_at'}:
            frame[col] = frame[col].astype('string')
    frame['record_index'] = frame.record_index.astype('int64')
    frame['fetched_at'] = pd.to_datetime(frame.fetched_at, utc=True)
    target = root / 'bronze' / 'news.parquet'
    with lock(root):
        old = pd.read_parquet(target) if target.exists() else frame.iloc[:0]
        remaining = old[old.batch_id != env['batch_id']]
        combined = frame if remaining.empty else remaining if frame.empty else pd.concat(
            [remaining, frame], ignore_index=True)
        atomic_frame(combined, target)
    return frame


def rebuild_bronze(*, root=DEFAULT_ROOT):
    """Backfill/replay every JSON capture without changing capture contents."""
    root = Path(root)
    paths = sorted((root / 'bronze').glob('*/*/*.json'))
    for path in paths:
        materialize_bronze(path, root=root)
    if not paths:
        # Use a valid empty schema even before the first download.
        frame = pd.DataFrame({c: pd.Series(dtype=(
            'int64' if c == 'record_index' else 'datetime64[ns, UTC]' if c == 'fetched_at'
            else 'string')) for c in BRONZE_COLUMNS})
        with lock(root):
            target = root / 'bronze' / 'news.parquet'
            if not target.exists():
                atomic_frame(frame, target)
    return pd.read_parquet(root / 'bronze' / 'news.parquet')


def _snapshot_records(payload, tickers):
    from src.ml.news_sources import get_entity_aliases
    records = list(payload.get('noticias') or [])
    for items in (payload.get('cables') or {}).values():
        if isinstance(items, list):
            records.extend(items)
    for raw in records:
        title = raw.get('n') or raw.get('t') or ''
        text = raw.get('c') or ''
        matches = [ticker for ticker in tickers if any(
            re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', title + ' ' + text, re.I)
            for alias in get_entity_aliases(ticker) if alias)]
        for ticker in matches or ['']:
            yield dict(title=title, summary=text, url=raw.get('f', ''), ticker=ticker,
                       topic='' if ticker else 'market', published_at=None)


def normalize(path, *, root=DEFAULT_ROOT, tickers=()):
    """Materialize one bronze batch into silver; safe to replay the same batch."""
    root, path = Path(root), Path(path)
    env = json.loads(path.read_text(encoding='utf-8'))
    materialize_bronze(path, root=root)
    # Silver consumes the unified raw table, rather than a source-specific file.
    with lock(root):
        bronze = pd.read_parquet(root / 'bronze' / 'news.parquet')
        batch = bronze[bronze.batch_id == env['batch_id']].sort_values('record_index')

    def records_from_bronze():
        for record in batch.itertuples():
            raw = json.loads(record.raw_payload)
            if record.record_kind in {'noticia', 'cable'}:
                yield from _snapshot_records({'noticias': [raw]}, tickers)
            else:
                yield raw

    records = records_from_bronze()
    rows = []
    for raw in records:
        title = clean_news_text(raw.get('title') or raw.get('titulo'))
        if not title:
            continue
        source = str(raw.get('source') or env['source'])
        url = str(raw.get('url') or '')
        identity = json.dumps([source, url or ' '.join(title.casefold().split())], ensure_ascii=False)
        published = timestamp(raw.get('published_at', raw.get('published_dt', raw.get('published_ts', raw.get('fecha')))))
        # DataBursátil's connector can synthesize timestamps; do not treat those as publication evidence.
        if source == 'databursatil':
            published = pd.NaT
        fetched = utc(env['fetched_at'])
        text = clean_news_text(raw.get('summary') or raw.get('texto'))
        headline_only = not text or text.casefold() == title.casefold()
        rows.append(dict(article_id=sha256(identity.encode()).hexdigest(),
                         batch_id=env['batch_id'], bronze_path=str(path.relative_to(root)),
                         source=source, ticker=str(raw.get('ticker') or ''),
                         topic=str(raw.get('topic') or ('trump' if env['source'] == 'trump' else '')),
                         title=title, text=None if headline_only else text,
                         text_kind='headline' if headline_only else 'excerpt',
                         url=url, published_at=published, fetched_at=fetched,
                         available_at=max(published, fetched) if pd.notna(published) else fetched,
                         sentiment=_received_sentiment(raw, env.get('representation')),
                         relevance_score=float(raw.get('relevance_score', 1.0))))
    frame = pd.DataFrame(rows, columns=SILVER_COLUMNS)
    for col in ['published_at', 'fetched_at', 'available_at']:
        frame[col] = pd.to_datetime(frame[col], utc=True)
    frame['sentiment'] = pd.to_numeric(frame.sentiment, errors='coerce').astype('float64')
    frame = frame.drop_duplicates(['article_id', 'ticker', 'topic', 'batch_id'])
    target = root / 'silver' / 'articles.parquet'
    with lock(root):
        old = pd.read_parquet(target) if target.exists() else frame.iloc[:0]
        combined = pd.concat([old[old.batch_id != env['batch_id']], frame], ignore_index=True)
        atomic_frame(combined, target)
    return frame


def ingest(source, payload, *, root=DEFAULT_ROOT, fetched_at=None,
           representation='normalized_extract', origin=None, tickers=()):
    path = capture(source, payload, root=root, fetched_at=fetched_at,
                   representation=representation, origin=origin)
    normalize(path, root=root, tickers=tickers)
    return path


def build_gold(*, root=DEFAULT_ROOT, as_of=None, days_back=7):
    """Compute ticker/topic features from evidence available at the cutoff."""
    from src.ml.news import Article, _compute_features_from_articles, _score_keywords
    if days_back < 1:
        raise ValueError('days_back must be positive')
    root = Path(root)
    cutoff = utc(as_of)
    with lock(root):
        path = root / 'silver' / 'articles.parquet'
        data = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=SILVER_COLUMNS)
        if not data.empty:
            data = data[data.available_at <= cutoff].copy()
            data['event_at'] = data.published_at.fillna(data.fetched_at)
            data = data[(data.event_at <= cutoff) & (data.event_at >= cutoff - pd.Timedelta(days=days_back * 2))]
            data = data.sort_values('fetched_at').drop_duplicates(['article_id', 'ticker', 'topic'], keep='last')
        rows = []
        for (ticker, topic), group in data.groupby(['ticker', 'topic']):
            arts = []
            for r in group.itertuples():
                summary = '' if pd.isna(r.text) else r.text
                arts.append(Article(
                    ticker=ticker, title=r.title, summary=summary, url=r.url,
                    published_ts=int(r.event_at.timestamp()), source=r.source,
                    sentiment=_score_keywords(r.title + ' ' + summary),
                    relevance_score=r.relevance_score))
            feat = _compute_features_from_articles(arts, days_back=days_back, as_of=cutoff.to_pydatetime())
            rows.append(dict(date=cutoff.normalize(), ticker=ticker, topic=topic,
                             as_of=cutoff, window_days=days_back, scoring_method='keywords',
                             publication_date_missing=int(group.published_at.isna().sum()),
                             article_ids=json.dumps(sorted(group.article_id.unique().tolist())),
                             batch_ids=json.dumps(sorted(group.batch_id.unique().tolist())), **feat))
        frame = pd.DataFrame(rows, columns=GOLD_COLUMNS)
        for col in ['date', 'as_of']:
            frame[col] = pd.to_datetime(frame[col], utc=True)
        target = root / 'gold' / 'historical_features.parquet'
        legacy = root / 'gold' / 'daily_features.parquet'
        previous = target if target.exists() else legacy
        old = pd.read_parquet(previous) if previous.exists() else pd.DataFrame()
        # Replace this calculation date, including a now-empty result, without leaving stale rows.
        if not old.empty:
            old = old[old.date != cutoff.normalize()]
        combined = frame.copy() if old.empty else old.copy() if frame.empty else pd.concat([old, frame], ignore_index=True)
        combined = combined.sort_values(['date', 'ticker', 'topic']).reset_index(drop=True)
        atomic_frame(combined, target)
        # Keep the previous path as a compatibility export of the historical table.
        atomic_frame(combined, legacy)
        today_target = root / 'gold' / 'today_features.parquet'
        today = utc().normalize()
        if cutoff.normalize() == today:
            daily = frame.copy()
        else:
            # Backfills must never replace today's results with a historical day.
            daily = pd.read_parquet(today_target) if today_target.exists() else frame.iloc[:0].copy()
            daily = daily[daily.date == today].reset_index(drop=True)
        atomic_frame(daily, today_target)
        latest = root / 'gold' / 'latest.json'
        tmp = latest.with_suffix('.json.tmp')
        tmp.write_text(daily.to_json(orient='records', date_format='iso', force_ascii=False), encoding='utf-8')
        os.replace(tmp, latest)
    return frame


def rebuild_gold_history(*, root=DEFAULT_ROOT, days_back=7, as_of=None):
    """Recompute daily cuts from first known availability, without future leakage."""
    root = Path(root)
    end = utc(as_of)
    path = root / 'silver' / 'articles.parquet'
    data = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=SILVER_COLUMNS)
    eligible = data[data.available_at <= end]
    start = eligible.available_at.min() if not eligible.empty else end
    for day in pd.date_range(start.normalize(), end.normalize(), freq='D'):
        cutoff = min(day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1), end)
        build_gold(root=root, as_of=cutoff, days_back=days_back)
    return pd.read_parquet(root / 'gold' / 'historical_features.parquet')
