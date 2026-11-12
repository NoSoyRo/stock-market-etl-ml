"""Silver (PySpark): lee bronze, normaliza nombres y tipos y marca la calidad.

    python -m src.etl.silver.mercado.etl_silver
    python -m src.etl.silver.mercado.etl_silver --solo intraday

Entrada:  data/market/bronze/*.csv, data/intraday/bronze/*.csv
Salida:   data/market/silver/<SIMBOLO>.csv, data/intraday/silver/<EMISORA>.csv

Reglas:
  - Nombres en minúsculas; Date → fecha (DATE es palabra reservada en SQL).
  - No se borra ni se corrige ninguna fila: los problemas se marcan con flag_*.
  - Intradía: se marcan los días festivos con barras de relleno, los días en
    que la fuente mandó un solo precio para casi todas las emisoras, los días
    planos y los incompletos.
"""
from __future__ import annotations

import argparse
import sys

from pyspark.sql import DataFrame, Window, functions as F

from src.etl.shared.precios_fx import (
    BARRAS_DIA, FX, INTRADAY, MARKET, SALTO, UMBRAL_FALLA, columnas, crear_spark, error_claro,
    escribir_diccionario, guardar_por_ticker, leer_capa, muestra,
)

SCRIPT = "src.etl.silver.mercado.etl_silver"


def _sin_duplicados(df: DataFrame, llave: list[str]) -> DataFrame:
    """Si una llave se repite, se queda la última fila del archivo (la más reciente).

    Spark con dropDuplicates escoge cualquiera; así el resultado es siempre el mismo.
    _orden es la posición de la fila al leerla.
    """
    w = Window.partitionBy(*llave).orderBy(F.col("_orden").desc())
    return (df.withColumn("_n", F.row_number().over(w))
            .filter("_n = 1")
            .drop("_n", "_orden"))


def silver_market(bronze: DataFrame) -> DataFrame:
    df = (
        bronze.select(
            "ticker",
            F.col("Date").alias("fecha"),
            F.col("Open").alias("open"),
            F.col("High").alias("high"),
            F.col("Low").alias("low"),
            F.col("Close").alias("close"),
            F.coalesce(F.col("Volume"), F.lit(0)).cast("long").alias("volume"),
            F.monotonically_increasing_id().alias("_orden"),
        )
    )
    df = _sin_duplicados(df, ["ticker", "fecha"])
    w = Window.partitionBy("ticker").orderBy("fecha")
    prev = F.lag("close").over(w)
    es_mxn = F.col("ticker").endswith(".MX") | (F.col("ticker") == FX)
    df = (
        df.withColumn("moneda", F.when(es_mxn, "MXN").otherwise("USD"))
        .withColumn("flag_volumen_cero", (F.col("volume") == 0).cast("int"))
        .withColumn("flag_precio_no_positivo",
                    ((F.col("open") <= 0) | (F.col("high") <= 0) | (F.col("low") <= 0) | (F.col("close") <= 0)).cast(
                        "int"))
        .withColumn("flag_ohlc_incoherente",
                    ((F.col("high") < F.greatest("open", "close") - 1e-9)
                     | (F.col("low") > F.least("open", "close") + 1e-9)).cast("int"))
        .withColumn("flag_salto_25",
                    F.coalesce((F.abs(F.col("close") / prev - 1) > SALTO).cast("int"), F.lit(0)))
        .withColumn("flag_cierre_repetido",
                    F.coalesce((F.col("close") == prev).cast("int"), F.lit(0)))
    )
    return df.select(*columnas("market/silver"))


def dias_bmv(silver_mkt: DataFrame) -> DataFrame:
    """Días con sesión: operó al menos la mitad de las emisoras .MX vivas ese día."""
    mx = silver_mkt.filter(F.col("ticker").endswith(".MX")).select("ticker", "fecha")
    vida = mx.groupBy("ticker").agg(F.min("fecha").alias("ini"), F.max("fecha").alias("fin"))
    fechas = mx.select("fecha").distinct()
    vivos = (fechas.crossJoin(vida)
             .filter((F.col("fecha") >= F.col("ini")) & (F.col("fecha") <= F.col("fin")))
             .groupBy("fecha").agg(F.count("*").alias("vivos")))
    presentes = mx.groupBy("fecha").agg(F.count("*").alias("presentes"))
    return (presentes.join(vivos, "fecha")
            .filter(F.col("presentes") / F.col("vivos") >= 0.5)
            .select("fecha"))


def silver_intraday(bronze: DataFrame, dias: DataFrame) -> DataFrame:
    df = (
        bronze.select(
            "ticker",
            F.col("datetime").alias("fecha_hora"),
            F.col("price").alias("precio"),
            F.monotonically_increasing_id().alias("_orden"),
        )
    )
    df = (
        _sin_duplicados(df, ["ticker", "fecha_hora"])
        .withColumn("fecha", F.substring("fecha_hora", 1, 10))
    )
    w_dia = Window.partitionBy("ticker", "fecha")
    df = (df.withColumn("_distintos", F.size(F.collect_set("precio").over(w_dia)))
          .withColumn("_barras", F.count("*").over(w_dia))
          .withColumn("_plano", F.col("_distintos") == 1))

    # Días que no se pueden validar contra el diario (posteriores al último diario) cuentan como hábiles.
    ultimo = dias.agg(F.max("fecha")).first()[0] or ""
    habiles = dias.withColumn("_habil", F.lit(True))
    df = (df.join(F.broadcast(habiles), "fecha", "left")
          .withColumn("_habil", F.coalesce("_habil", F.col("fecha") > F.lit(ultimo))))

    # Falla de la fuente: el mismo día, casi todas las emisoras traen un solo precio.
    planos = df.select("ticker", "fecha", "_plano").distinct()
    falla = (planos.groupBy("fecha")
             .agg(F.avg(F.col("_plano").cast("double")).alias("_pct_planos"))
             .withColumn("_falla", F.col("_pct_planos") >= UMBRAL_FALLA)
             .select("fecha", "_falla"))
    df = df.join(F.broadcast(falla), "fecha", "left")

    festivo = (~F.col("_habil")) & F.col("_plano")
    df = (df.withColumn("flag_festivo", festivo.cast("int"))
          .withColumn("flag_falla_fuente", (F.col("_falla") & F.col("_plano") & ~festivo).cast("int"))
          .withColumn("flag_dia_plano", F.col("_plano").cast("int"))
          .withColumn("flag_dia_incompleto", (F.col("_barras") < BARRAS_DIA).cast("int")))
    return df.select(*columnas("intraday/silver"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capa silver de Precios & FX con PySpark")
    parser.add_argument("--solo", choices=("market", "intraday"))
    args = parser.parse_args(argv)
    spark = None
    try:
        spark = crear_spark("silver_mercado")
        mkt = silver_market(leer_capa(spark, "market", "bronze", "market/bronze")).cache()
        if args.solo in (None, "market"):
            n = guardar_por_ticker(mkt, "market/silver", MARKET / "silver", ["fecha"])
            escribir_diccionario("market/silver", muestra(mkt, "fecha"), n, SCRIPT)
            print(f"  silver market: {n} símbolos, {mkt.count():,} filas")
        if args.solo in (None, "intraday"):
            dias = dias_bmv(mkt).cache()
            intra = silver_intraday(leer_capa(spark, "intraday", "bronze", "intraday/bronze"), dias).cache()
            n = guardar_por_ticker(intra, "intraday/silver", INTRADAY / "silver", ["fecha_hora"])
            escribir_diccionario("intraday/silver", muestra(intra, "fecha_hora", ticker="WALMEX"), n, SCRIPT)
            fallas = sorted(r[0] for r in intra.filter("flag_falla_fuente = 1").select("fecha").distinct().collect())
            print(f"  silver intraday: {n} emisoras, {intra.count():,} filas")
            print(f"  días hábiles con falla de la fuente: {', '.join(fallas) or 'ninguno'}")
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
