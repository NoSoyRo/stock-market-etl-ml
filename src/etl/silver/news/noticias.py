"""Normalización distribuida de noticias y expansión de entidades."""
import json
import math
from hashlib import sha256

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from src.data.news_medallion import (
    SILVER_COLUMNS, _received_sentiment, _snapshot_records, clean_news_text, timestamp,
)

SCHEMA = StructType([StructField(c, StringType(), True) for c in SILVER_COLUMNS])


def _normalize(record, tickers):
    raw = json.loads(record.raw_payload)
    records = (_snapshot_records({'noticias': [raw]}, tickers)
               if record.record_kind in {'noticia', 'cable'} else [raw])
    rows = []
    for item in records:
        title = clean_news_text(item.get('title') or item.get('titulo'))
        if not title:
            continue
        source = str(item.get('source') or record.capture_source)
        url = str(item.get('url') or '')
        identity = json.dumps([source, url or ' '.join(title.casefold().split())], ensure_ascii=False)
        text = clean_news_text(item.get('summary') or item.get('texto'))
        headline = not text or text.casefold() == title.casefold()
        published = timestamp(item.get('published_at', item.get('published_dt',
                              item.get('published_ts', item.get('fecha')))))
        relevance = item.get('relevance_score', 1.0)
        try:
            relevance = float(relevance)
        except (TypeError, ValueError):
            relevance = 1.0
        if not math.isfinite(relevance):
            relevance = 1.0
        rows.append(dict(
            article_id=sha256(identity.encode()).hexdigest(), batch_id=record.batch_id,
            bronze_path=record.bronze_path, source=source,
            ticker=str(item.get('ticker') or ''),
            topic=str(item.get('topic') or ('trump' if record.capture_source == 'trump' else '')),
            title=title, text=None if headline else text,
            text_kind='headline' if headline else 'excerpt', url=url,
            published_at=None if source == 'databursatil' or str(published) == 'NaT' else published.isoformat(),
            fetched_at=record.fetched_at.isoformat(), available_at=None,
            sentiment=str(_received_sentiment(item, record.representation))
                      if _received_sentiment(item, record.representation) is not None else None,
            relevance_score=str(relevance)))
    return rows


def transformar(bronce, tickers=()):
    normalize = F.udf(lambda row: _normalize(row, tuple(tickers)), ArrayType(SCHEMA))
    data = bronce.select(F.explode(normalize(F.struct('*'))).alias('article')).select('article.*')
    for column in ('published_at', 'fetched_at'):
        data = data.withColumn(column, F.to_timestamp(column))
    return (data.withColumn('available_at', F.greatest('published_at', 'fetched_at'))
            .withColumn('sentiment', F.col('sentiment').cast('double'))
            .withColumn('relevance_score', F.col('relevance_score').cast('double'))
            .dropDuplicates(['article_id', 'ticker', 'topic', 'batch_id']))


def ejecutar(spark, entrada, salida, tickers=()):
    transformar(spark.read.parquet(str(entrada)), tickers).write.mode('overwrite').parquet(str(salida))
