"""Ejecuta las tres capas de noticias con PySpark."""
import argparse
from pathlib import Path

from pyspark.sql import SparkSession

from src.etl.bronce.news.noticias import ejecutar as bronce
from src.etl.silver.news.noticias import ejecutar as silver
from src.etl.golden.news.noticias import ejecutar as golden

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capturas', type=Path, default=ROOT / 'data/news/bronze')
    parser.add_argument('--salida', type=Path, default=ROOT / 'data/news/noticias')
    parser.add_argument('--as-of', help='Corte ISO 8601; por defecto ahora UTC')
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--master', default='local[*]')
    args = parser.parse_args()
    if args.days < 1:
        parser.error('--days debe ser positivo')
    if not any(args.capturas.glob('*/*/*.json')):
        parser.error(f'No hay capturas source/fecha/lote.json en {args.capturas}')
    if args.salida.resolve() == args.capturas.resolve() or args.salida.resolve() in args.capturas.resolve().parents:
        parser.error('--salida no puede contener la carpeta de capturas')
    from src.ml.news_sources import ENTITY_ALIASES

    spark = (SparkSession.builder.master(args.master).appName('noticias-pyspark')
             .config('spark.sql.session.timeZone', 'UTC').getOrCreate())
    try:
        bronce(spark, args.capturas, args.salida / 'bronce')
        silver(spark, args.salida / 'bronce', args.salida / 'silver', tuple(ENTITY_ALIASES))
        golden(spark, args.salida / 'silver', args.salida / 'golden', args.as_of, args.days)
        print(f'Capas de noticias generadas en {args.salida}')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
