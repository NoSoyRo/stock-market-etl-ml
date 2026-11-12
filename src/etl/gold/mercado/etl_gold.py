"""Gold (PySpark): lee silver y agrega lo que consumen el agente y ML.

    python -m src.etl.gold.mercado.etl_gold
    python -m src.etl.gold.mercado.etl_gold --solo market

Entrada:  data/market/silver/*.csv, data/intraday/silver/*.csv
Salida:
  data/market/gold/<SIMBOLO>.csv          + usdmxn_close, close_usd, ret_1d, target_next_day, es_valido
  data/intraday/gold/<EMISORA>.csv        por barra: hora, minuto, ret_5m, target_next_5m
  data/intraday/gold/diario/<EMISORA>.csv resumen por día
  data/market/gold/mercado.db             base SQLite que consulta el agente

Rendimientos y targets en porcentaje (1.2 = +1.2 %). Los targets intradía no
cruzan de un día a otro. Gold intradía excluye festivos y fallas de la fuente.
"""
from __future__ import annotations

import argparse
import sys

from pyspark.sql import DataFrame, Window, functions as F

from src.etl.shared.precios_fx import (
    BARRAS_DIA, DB_PATH, FX, INTRADAY, MARKET, MAX_DIAS_FX, ROOT, carpeta, columnas, construir_db,
    crear_spark, error_claro, escribir_diccionario, guardar_por_ticker, leer_capa, muestra, redondear,
)

SCRIPT = "src.etl.gold.mercado.etl_gold"


def gold_market(silver: DataFrame) -> DataFrame:
    # Tipo de cambio del día; si falta, el último conocido hasta MAX_DIAS_FX días antes.
    # Se arma un calendario día por día y se rellena hacia adelante, para unir por igualdad.
    fx = (silver.filter((F.col("ticker") == FX) & (F.col("close") > 0))
          .select("fecha", F.col("close").alias("fx_close")))
    rango = silver.agg(F.min("fecha").alias("a"), F.max("fecha").alias("b"))
    calendario = rango.select(
        F.explode(F.sequence(F.to_date("a"), F.to_date("b"))).alias("dia")
    ).select(F.date_format("dia", "yyyy-MM-dd").alias("fecha"))
    ventana_fx = Window.orderBy("fecha").rowsBetween(-MAX_DIAS_FX, 0)
    fx_diario = (calendario.join(fx, "fecha", "left")
                 .withColumn("usdmxn_close", F.last("fx_close", ignorenulls=True).over(ventana_fx))
                 .select("fecha", "usdmxn_close"))
    df = silver.join(F.broadcast(fx_diario), "fecha", "left")

    df = df.withColumn(
        "close_usd",
        F.when(F.col("ticker") == FX, F.lit(None).cast("double"))
        .when(F.col("moneda") == "MXN", F.col("close") / F.col("usdmxn_close"))
        .otherwise(F.col("close")),
    )
    w = Window.partitionBy("ticker").orderBy("fecha")
    prev = F.lag("close").over(w)
    df = df.withColumn("ret_1d", F.when(prev > 0, (F.col("close") / prev - 1) * 100))
    df = df.withColumn("target_next_day", F.lead("ret_1d").over(w))

    malo = (F.col("flag_precio_no_positivo") == 1) | (F.col("flag_ohlc_incoherente") == 1)
    malo = malo | ((F.col("flag_volumen_cero") == 1) & (F.col("ticker") != FX))
    df = df.withColumn("es_valido", (~malo).cast("int"))
    return redondear(df.select(*columnas("market/gold")), "market/gold")


def gold_intraday(silver: DataFrame) -> DataFrame:
    df = silver.filter((F.col("flag_festivo") == 0) & (F.col("flag_falla_fuente") == 0))
    w = Window.partitionBy("ticker", "fecha").orderBy("fecha_hora")
    prev = F.lag("precio").over(w)
    df = (df.withColumn("hora", F.substring("fecha_hora", 12, 2).cast("long"))
          .withColumn("minuto", F.substring("fecha_hora", 15, 2).cast("long"))
          .withColumn("ret_5m", F.when(prev > 0, (F.col("precio") / prev - 1) * 100))
          .withColumn("target_next_5m", F.lead("ret_5m").over(w)))
    return redondear(df.select(*columnas("intraday/gold")), "intraday/gold")


def gold_intraday_diario(barras: DataFrame) -> DataFrame:
    df = barras.groupBy("ticker", "fecha").agg(
        F.count("*").alias("n_barras"),
        F.min_by("precio", "fecha_hora").alias("precio_primero"),
        F.max_by("precio", "fecha_hora").alias("precio_ultimo"),
        F.max("precio").alias("precio_max"),
        F.min("precio").alias("precio_min"),
        F.stddev_samp("ret_5m").alias("volatilidad_5m"),
        F.countDistinct("precio").alias("_distintos"),
    )
    df = (df.withColumn("ret_intradia",
                        F.when(F.col("precio_primero") > 0,
                               (F.col("precio_ultimo") / F.col("precio_primero") - 1) * 100))
          .withColumn("dia_completo", (F.col("n_barras") == BARRAS_DIA).cast("int"))
          .withColumn("dia_plano", (F.col("_distintos") == 1).cast("int")))
    return redondear(df.select(*columnas("intraday/gold_diario")), "intraday/gold_diario")


def _gold_en_disco(spark, dominio: str, capa: str, sub: str = "") -> DataFrame | None:
    """El gold que dejó una corrida anterior, o None si todavía no existe."""
    if not any((carpeta(dominio, "gold") / sub).glob("*.csv")):
        return None
    return leer_capa(spark, dominio, "gold", capa, sub)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capa gold de Precios & FX con PySpark")
    parser.add_argument("--solo", choices=("market", "intraday"))
    args = parser.parse_args(argv)
    spark = None
    try:
        spark = crear_spark("gold_mercado")
        mkt = diario = None
        if args.solo in (None, "market"):
            mkt = gold_market(leer_capa(spark, "market", "silver", "market/silver")).cache()
            n = guardar_por_ticker(mkt, "market/gold", MARKET / "gold", ["fecha"])
            escribir_diccionario("market/gold", muestra(mkt, "fecha"), n, SCRIPT)
            print(f"  gold market: {n} símbolos, {mkt.count():,} filas")
        if args.solo in (None, "intraday"):
            barras = gold_intraday(leer_capa(spark, "intraday", "silver", "intraday/silver")).cache()
            diario = gold_intraday_diario(barras).cache()
            n = guardar_por_ticker(barras, "intraday/gold", INTRADAY / "gold", ["fecha_hora"])
            guardar_por_ticker(diario, "intraday/gold_diario", INTRADAY / "gold" / "diario", ["fecha"])
            escribir_diccionario(
                "intraday/gold", muestra(barras, "fecha_hora", ticker="WALMEX"), n, SCRIPT,
                extra=("intraday/gold_diario", muestra(diario, "fecha", ticker="WALMEX"), "diario/"),
            )
            print(f"  gold intraday: {n} emisoras, {barras.count():,} barras, {diario.count():,} días")
        # La base siempre lleva sus tres tablas: lo que no se recalculó en esta corrida
        # (por --solo) se toma del gold que ya está en disco.
        if mkt is None:
            mkt = _gold_en_disco(spark, "market", "market/gold")
        if diario is None:
            diario = _gold_en_disco(spark, "intraday", "intraday/gold_diario", "diario")
        if mkt is None:
            print("  ! la base del agente no se actualizó: falta data/market/gold (corre gold sin --solo)")
        else:
            if diario is None:
                print("  ! la base queda sin intradía: falta data/intraday/gold/diario")
            # La base se arma con pandas: SQLite no se escribe desde Spark sin un driver JDBC.
            construir_db(mkt.toPandas(), diario.toPandas() if diario is not None else None)
            print(f"  base del agente: {DB_PATH.relative_to(ROOT)}")
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
