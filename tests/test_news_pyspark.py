"""Verifica el contrato entre capas Spark con capturas pequeñas y sin red."""
import json

import pytest

pytest.importorskip('pyspark')
from pyspark.sql import SparkSession

from src.etl.bronce.news.noticias import ejecutar as bronce
from src.etl.silver.news.noticias import ejecutar as silver
from src.etl.golden.news.noticias import ejecutar as golden


def test_pipeline(tmp_path):
    spark = (SparkSession.builder.master('local[1]').appName('test-noticias')
             .config('spark.sql.session.timeZone', 'UTC')
             .config('spark.sql.shuffle.partitions', '1').getOrCreate())
    try:
        captures = tmp_path / 'capturas'
        source = captures / 'rss' / '2026-10-30'
        source.mkdir(parents=True)
        article = dict(title='<b>Alsea growth</b>', summary='profit',
                       ticker='ALSEA', url='https://example.org/a',
                       published_at='2026-10-29T12:00:00Z')
        for batch, fetched in [('a', '2026-10-30T12:00:00Z'), ('b', '2026-11-01T12:00:00Z')]:
            (source / f'{batch}.json').write_text(json.dumps(dict(
                batch_id=batch, source='rss', fetched_at=fetched,
                representation='normalized_extract', payload=[article, article])))
        db = captures / 'databursatil' / '2026-10-30'
        db.mkdir(parents=True)
        (db / 'c.json').write_text(json.dumps(dict(
            batch_id='c', source='databursatil', fetched_at='2026-10-30T12:00:00Z',
            payload={'noticias': [{'n': 'Alsea anuncia inversión', 'c': '<p>Texto</p>'}],
                     'cables': {'mercado': [{'t': 'Mercado general', 'c': ''}]}})))
        b, s, g = [tmp_path / layer for layer in ('bronce', 'silver', 'golden')]
        bronce(spark, captures, b)
        assert spark.read.parquet(str(b)).count() == 6
        silver(spark, b, s, ['ALSEA'])
        normalized = spark.read.parquet(str(s))
        assert normalized.count() == 4
        assert normalized.filter("title = 'Alsea growth'").count() == 2
        assert normalized.filter("source = 'databursatil' AND published_at IS NOT NULL").count() == 0
        golden(spark, s, g, '2026-10-31T12:00:00Z')
        result = spark.read.parquet(str(g))
        alsea = result.filter("ticker = 'ALSEA'").first()
        assert alsea.n_articles == 2
        assert set(alsea.batch_ids) == {'a', 'c'}  # El lote futuro nunca participa.
        expected = result.orderBy('ticker', 'topic').collect()
        golden(spark, s, g, '2026-10-31T12:00:00Z')
        assert spark.read.parquet(str(g)).orderBy('ticker', 'topic').collect() == expected
        golden(spark, s, g, '2020-01-01T00:00:00Z')
        assert spark.read.parquet(str(g)).count() == 0  # No quedan resultados obsoletos.
    finally:
        spark.stop()
