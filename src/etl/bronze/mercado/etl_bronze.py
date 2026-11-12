"""Bronze (PySpark): de los .parquet de la fuente a CSV con los campos definidos.

    python -m src.etl.bronze.mercado.etl_bronze
    python -m src.etl.bronze.mercado.etl_bronze --solo market

Entrada (no se modifica):
  data/market/*.parquet    OHLCV diario de Yahoo Finance, un archivo por símbolo
  data/intraday/*.parquet  precio cada 5 minutos de DataBursátil, un archivo por emisora

Salida:
  data/market/bronze/<SIMBOLO>.csv     ticker, Date, Open, High, Low, Close, Volume
  data/intraday/bronze/<EMISORA>.csv   ticker, datetime, price

Bronze conserva los nombres originales de la fuente. Solo agrega el ticker
(que en la fuente viene en el nombre del archivo) y fija los tipos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import DataFrame, functions as F
from pyspark.sql import types as T

from src.etl.shared.precios_fx import (
    INTRADAY, MARKET, ROOT, ErrorETL, columnas, crear_spark, error_claro, escribir_diccionario,
    guardar_por_ticker, muestra, redondear, ticker_de_archivo,
)

SCRIPT = "src.etl.bronze.mercado.etl_bronze"


def _a_texto(col_name: str, tipo, formato: str):
    """La fuente trae fechas en nanosegundos (número) o como timestamp, según quién la escribió."""
    c = F.col(col_name)
    if isinstance(tipo, (T.LongType, T.IntegerType)):
        c = F.timestamp_micros((c / 1000).cast("long"))
    return F.date_format(c, formato)


def _leer_parquets(spark, carpeta: Path) -> list[DataFrame]:
    """Un DataFrame por grupo de archivos con el mismo esquema.

    No todos los parquet guardan igual las fechas (nanosegundos o milisegundos)
    ni el volumen (entero o decimal), y Spark no puede leerlos juntos si los
    tipos no coinciden. Se agrupan por esquema y cada grupo se lee de una vez.
    """
    import pyarrow.parquet as pq

    grupos: dict[str, list[str]] = {}
    for path in sorted(carpeta.glob("*.parquet")):
        meta = pq.read_metadata(path)  # leer el pie del archivo es instantáneo
        if meta.num_rows == 0:
            print(f"  ! {path.name} vacío, se omite")
            continue
        firma = str(meta.schema.to_arrow_schema().remove_metadata())
        grupos.setdefault(firma, []).append(str(path))
    if not grupos:
        raise ErrorETL(f"No hay archivos .parquet con datos en {carpeta.relative_to(ROOT)}. "
                       "Revisa que el repo esté completo (git pull).")
    return [spark.read.parquet(*rutas).withColumn("ticker", ticker_de_archivo())
            for rutas in grupos.values()]


def bronze_market(spark) -> DataFrame:
    partes = []
    for df in _leer_parquets(spark, MARKET):
        tipo_fecha = df.schema["Date"].dataType
        partes.append(df.select(
            "ticker",
            _a_texto("Date", tipo_fecha, "yyyy-MM-dd").alias("Date"),
            F.col("Open").cast("double").alias("Open"),
            F.col("High").cast("double").alias("High"),
            F.col("Low").cast("double").alias("Low"),
            F.col("Close").cast("double").alias("Close"),
            F.col("Volume").cast("long").alias("Volume"),
        ))
    out = partes[0]
    for p in partes[1:]:
        out = out.unionByName(p)
    return redondear(out.select(*columnas("market/bronze")), "market/bronze")


def bronze_intraday(spark) -> DataFrame:
    partes = []
    for df in _leer_parquets(spark, INTRADAY):
        tipo = df.schema["datetime"].dataType
        partes.append(df.select(
            "ticker",
            _a_texto("datetime", tipo, "yyyy-MM-dd HH:mm:ss").alias("datetime"),
            F.col("price").cast("double").alias("price"),
        ))
    out = partes[0]
    for p in partes[1:]:
        out = out.unionByName(p)
    return redondear(out.select(*columnas("intraday/bronze")), "intraday/bronze")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capa bronze de Precios & FX con PySpark")
    parser.add_argument("--solo", choices=("market", "intraday"))
    args = parser.parse_args(argv)
    spark = None
    try:
        spark = crear_spark("bronze_mercado")
        if args.solo in (None, "market"):
            df = bronze_market(spark).cache()
            n = guardar_por_ticker(df, "market/bronze", MARKET / "bronze", ["Date"])
            escribir_diccionario("market/bronze", muestra(df, "Date", desde_inicio=True), n, SCRIPT)
            print(f"  bronze market: {n} símbolos, {df.count():,} filas")
        if args.solo in (None, "intraday"):
            df = bronze_intraday(spark).cache()
            n = guardar_por_ticker(df, "intraday/bronze", INTRADAY / "bronze", ["datetime"])
            escribir_diccionario("intraday/bronze", muestra(df, "datetime", ticker="WALMEX", desde_inicio=True), n,
                                 SCRIPT)
            print(f"  bronze intraday: {n} emisoras, {df.count():,} filas")
    except Exception as e:
        if (msg := error_claro(e)) is None:
            raise
        print(f"\n  ✗ {msg}", file=sys.stderr)
        return 1
    finally:
        if spark is not None:
            spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
