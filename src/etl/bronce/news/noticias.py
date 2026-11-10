"""Reconstruye bronce con Spark a partir de las capturas JSON existentes."""
import json
from hashlib import sha256

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from src.data.news_medallion import BRONZE_COLUMNS, _capture_records, _raw_string


SCHEMA = StructType([StructField(c, StringType(), True) for c in BRONZE_COLUMNS])


def _records(content, path):
    env = json.loads(bytes(content).decode('utf-8'))
    rows = []
    for index, (kind, category, raw) in enumerate(_capture_records(env)):
        if not isinstance(raw, dict):
            raise ValueError(f'Registro de noticia inválido: {path}, índice {index}')
        published = raw.get('published_at', raw.get('published_dt',
                            raw.get('published_ts', raw.get('fecha'))))
        row = dict(record_id=sha256(f"{env['batch_id']}:{index}".encode()).hexdigest(),
                   batch_id=env['batch_id'], record_index=str(index),
                   source=raw.get('source') or env['source'], capture_source=env['source'],
                   fetched_at=env['fetched_at'], bronze_path=path,
                   representation=env.get('representation'), origin=env.get('origin'),
                   record_kind=kind, category=category,
                   source_record_id=_raw_string(raw.get('id', raw.get('guid'))),
                   title_raw=_raw_string(raw.get('title') or raw.get('titulo') or raw.get('n') or raw.get('t')),
                   text_raw=_raw_string(raw.get('summary') or raw.get('texto') or raw.get('c')),
                   url_raw=_raw_string(raw.get('url') or raw.get('f')),
                   published_at_raw=_raw_string(published),
                   raw_payload=json.dumps(raw, ensure_ascii=False))
        rows.append({key: _raw_string(value) for key, value in row.items()})
    return rows


def transformar(spark, capturas):
    """Lee source/fecha/lote.json; conserva duplicados y el payload completo."""
    files = spark.read.format('binaryFile').load(str(capturas).rstrip('/') + '/*/*/*.json')
    return (files.select(F.explode(F.udf(_records, ArrayType(SCHEMA))('content', 'path')).alias('record'))
            .select('record.*')
            .withColumn('record_index', F.col('record_index').cast('long'))
            .withColumn('fetched_at', F.to_timestamp('fetched_at')))


def ejecutar(spark, capturas, salida):
    transformar(spark, capturas).write.mode('overwrite').parquet(str(salida))
