"""Agregados PySpark por ticker y tema con corte de disponibilidad temporal."""
from datetime import datetime, timedelta, timezone

from pyspark.sql import Window, functions as F


def transformar(silver, as_of=None, days=7):
    """Calcula conteos y sentimiento sin ponderación para el corte solicitado."""
    from src.ml.news import _score_keywords

    if days < 1:
        raise ValueError('days debe ser positivo')
    cutoff = datetime.fromisoformat(as_of.replace('Z', '+00:00')) if isinstance(as_of, str) else as_of
    cutoff = cutoff or datetime.now(timezone.utc)
    cutoff = cutoff.replace(tzinfo=timezone.utc) if cutoff.tzinfo is None else cutoff.astimezone(timezone.utc)
    start = cutoff - timedelta(days=days)
    data = (silver.filter(F.col('available_at') <= F.lit(cutoff))
            .withColumn('event_at', F.coalesce('published_at', 'fetched_at'))
            .filter(F.col('event_at').between(cutoff - timedelta(days=2 * days), cutoff)))
    latest = Window.partitionBy('article_id', 'ticker', 'topic').orderBy(
        F.col('fetched_at').desc(), F.col('batch_id').desc())
    data = (data.withColumn('_rank', F.row_number().over(latest)).filter('_rank = 1')
            .withColumn('_score', F.udf(_score_keywords, 'double')(
                F.concat_ws(' ', 'title', 'text'))))
    recent = F.col('event_at') >= F.lit(start)
    grouped = data.groupBy('ticker', 'topic').agg(
        F.count(F.when(recent, 1)).alias('n_articles'),
        F.avg(F.when(recent, F.col('_score'))).alias('sentiment_avg'),
        F.avg(F.when(~recent, F.col('_score'))).alias('_prior'),
        F.stddev_pop(F.when(recent, F.col('_score'))).alias('sentiment_vol'),
        F.avg(F.when(recent, F.col('relevance_score'))).alias('relevance_avg'),
        F.sort_array(F.collect_set(F.when(recent, F.col('source')))).alias('sources'),
        F.sort_array(F.collect_set('article_id')).alias('article_ids'),
        F.sort_array(F.collect_set('batch_id')).alias('batch_ids'),
        F.sum(F.when(F.col('published_at').isNull(), 1).otherwise(0)).alias('publication_date_missing'))
    return (grouped.fillna(0.0, subset=['sentiment_avg', '_prior', 'sentiment_vol', 'relevance_avg'])
            .withColumn('sentiment_delta', F.col('sentiment_avg') - F.col('_prior'))
            .drop('_prior').withColumn('date', F.to_date(F.lit(cutoff)))
            .withColumn('as_of', F.lit(cutoff)).withColumn('window_days', F.lit(days))
            .withColumn('scoring_method', F.lit('keywords_unweighted')))


def ejecutar(spark, entrada, salida, as_of=None, days=7):
    """Lee silver y reemplaza el resultado golden del corte solicitado."""
    transformar(spark.read.parquet(str(entrada)), as_of, days).write.mode('overwrite').parquet(str(salida))
