"""Piezas comunes del ETL PySpark de Precios & FX (Equipo Mercado).

Lo usan src/etl/bronze/mercado, src/etl/silver/mercado y src/etl/gold/mercado:
rutas, esquemas de cada capa, la sesión de Spark, la escritura de un CSV por
ticker, el diccionario de datos y la base SQLite del agente.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
import warnings
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MARKET = ROOT / "data" / "market"
INTRADAY = ROOT / "data" / "intraday"
DB_PATH = MARKET / "gold" / "mercado.db"
FX = "MXN=X"
SALTO = 0.25  # cambio diario de cierre > 25 % se marca como sospechoso
MAX_DIAS_FX = 5  # si falta el tipo de cambio, el último conocido hasta 5 días antes
BARRAS_DIA = 79  # 6.5 h de sesión / 5 min + 1
UMBRAL_FALLA = 0.8  # % de emisoras con un solo precio el mismo día = falla de la fuente
DECIMALES = 10


class ErrorETL(Exception):
    """Error con un mensaje claro para quien corre el ETL; se muestra sin la traza de Java."""


# ── esquemas: una sola fuente de verdad para Spark, CSV, diccionario y SQLite ─

ESQUEMA = {
    "market/bronze": [
        ("ticker", "texto",
         "Símbolo de Yahoo Finance: WALMEX.MX (BMV, pesos), AAPL (USA, dólares), MXN=X (tipo de cambio)"),
        ("Date", "fecha", "Día de la sesión, YYYY-MM-DD"),
        ("Open", "decimal", "Precio de apertura"),
        ("High", "decimal", "Precio máximo del día"),
        ("Low", "decimal", "Precio mínimo del día"),
        ("Close", "decimal", "Precio de cierre"),
        ("Volume", "entero", "Títulos negociados. En MXN=X siempre es 0: el tipo de cambio no se negocia en una bolsa"),
    ],
    "market/silver": [
        ("ticker", "texto", "Símbolo de Yahoo Finance"),
        ("fecha", "fecha", "Día de la sesión, YYYY-MM-DD (Date se renombra porque DATE es palabra reservada en SQL)"),
        ("open", "decimal", "Precio de apertura"),
        ("high", "decimal", "Precio máximo del día"),
        ("low", "decimal", "Precio mínimo del día"),
        ("close", "decimal", "Precio de cierre"),
        ("volume", "entero", "Títulos negociados"),
        ("moneda", "texto", "MXN si el símbolo termina en .MX o es MXN=X; USD en otro caso"),
        ("flag_volumen_cero", "0/1", "1 si volume = 0 (normal en MXN=X; en una acción indica que no se negoció)"),
        ("flag_precio_no_positivo", "0/1", "1 si algún precio es <= 0"),
        ("flag_ohlc_incoherente", "0/1", "1 si high < max(open, close) o low > min(open, close)"),
        ("flag_salto_25", "0/1", "1 si el cierre cambió más de 25 % contra el día anterior"),
        ("flag_cierre_repetido", "0/1", "1 si el cierre es idéntico al del día anterior"),
    ],
    "market/gold": [
        ("ticker", "texto", "Símbolo de Yahoo Finance"),
        ("fecha", "fecha", "Día de la sesión, YYYY-MM-DD"),
        ("open", "decimal", "Precio de apertura, en la moneda del símbolo"),
        ("high", "decimal", "Precio máximo del día"),
        ("low", "decimal", "Precio mínimo del día"),
        ("close", "decimal", "Precio de cierre"),
        ("volume", "entero", "Títulos negociados"),
        ("moneda", "texto", "MXN o USD"),
        ("usdmxn_close", "decimal",
         "Pesos por dólar al cierre de ese día (MXN=X). Si ese día no hubo dato, el último de hasta 5 días antes"),
        ("close_usd", "decimal",
         "Cierre en dólares: close / usdmxn_close si moneda = MXN; close si moneda = USD. Vacío para MXN=X"),
        ("ret_1d", "decimal",
         "Rendimiento del día contra el cierre anterior, en porcentaje (1.2 = +1.2 %). Vacío en la primera fila"),
        ("target_next_day", "decimal",
         "Rendimiento en porcentaje de la siguiente sesión (el ret_1d de mañana). Vacío en la última fila. Es la etiqueta para ML: nunca usarla como variable de entrada"),
        ("es_valido", "0/1",
         "1 si la fila no tiene precio <= 0, ni OHLC incoherente, ni volumen 0 (MXN=X queda exento del volumen)"),
    ],
    "intraday/bronze": [
        ("ticker", "texto", "Emisora de DataBursátil sin serie: WALMEX, AMXL, FEMSAUBD"),
        ("datetime", "fecha y hora",
         "Marca de la barra de 5 minutos, tal como la entrega DataBursátil (07:30 a 14:00)"),
        ("price", "decimal", "Último precio negociado en esa marca. No hay open/high/low ni volumen por barra"),
    ],
    "intraday/silver": [
        ("ticker", "texto", "Emisora"),
        ("fecha_hora", "fecha y hora", "Marca de la barra, YYYY-MM-DD HH:MM:SS"),
        ("fecha", "fecha", "Día de la barra"),
        ("precio", "decimal", "Último precio negociado en esa marca"),
        ("flag_festivo", "0/1",
         "1 si ese día no hubo sesión en el diario de la BMV y todo el día trae el mismo precio (barras de relleno)"),
        ("flag_falla_fuente", "0/1",
         "1 si ese día casi todas las emisoras traen un solo precio a la vez: la fuente no mandó datos y rellenó"),
        ("flag_dia_plano", "0/1",
         "1 si la emisora tuvo un solo precio todo el día (en emisoras poco líquidas puede ser real: no hubo operaciones)"),
        ("flag_dia_incompleto", "0/1", "1 si el día tiene menos de 79 barras"),
    ],
    "intraday/gold": [
        ("ticker", "texto", "Emisora"),
        ("fecha_hora", "fecha y hora", "Marca de la barra"),
        ("fecha", "fecha", "Día de la barra"),
        ("hora", "entero", "Hora de la marca (7 a 14)"),
        ("minuto", "entero", "Minuto de la marca (0, 5, ..., 55)"),
        ("precio", "decimal", "Último precio negociado en esa marca"),
        ("ret_5m", "decimal",
         "Rendimiento contra la barra anterior del MISMO día, en porcentaje. Vacío en la primera barra de cada día"),
        ("target_next_5m", "decimal",
         "Rendimiento en porcentaje de la siguiente barra del mismo día. Vacío en la última barra. Etiqueta para ML"),
    ],
    "intraday/gold_diario": [
        ("ticker", "texto", "Emisora"),
        ("fecha", "fecha", "Día de la sesión"),
        ("n_barras", "entero", "Barras de 5 minutos ese día (79 = día completo)"),
        ("precio_primero", "decimal", "Precio de la primera barra"),
        ("precio_ultimo", "decimal", "Precio de la última barra"),
        ("precio_max", "decimal", "Precio máximo entre las barras (puede quedarse corto contra el High diario)"),
        ("precio_min", "decimal", "Precio mínimo entre las barras"),
        ("ret_intradia", "decimal", "Rendimiento de la primera a la última barra, en porcentaje"),
        ("volatilidad_5m", "decimal", "Desviación estándar de ret_5m del día, en porcentaje"),
        ("dia_completo", "0/1", "1 si n_barras = 79"),
        ("dia_plano", "0/1", "1 si todo el día hubo un solo precio"),
    ],
}


def columnas(capa: str) -> list[str]:
    return [c for c, _, _ in ESQUEMA[capa]]


def esquema_spark(capa: str):
    """StructType de una capa para leer sus CSV con tipos fijos."""
    from pyspark.sql import types as T

    tipos = {
        "texto": T.StringType(), "fecha": T.StringType(), "fecha y hora": T.StringType(),
        "decimal": T.DoubleType(), "entero": T.LongType(), "0/1": T.IntegerType(),
    }
    return T.StructType([T.StructField(c, tipos[t], True) for c, t, _ in ESQUEMA[capa]])


# ── Spark ────────────────────────────────────────────────────────────────────

JAVA_MINIMO = 17
INSTALAR_JAVA = "en Fedora: sudo dnf install java-21-openjdk-headless"


def _revisar_java() -> None:
    """PySpark necesita Java 17 o más reciente; sin él, Spark falla con un error difícil de leer."""
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        java = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if not java.exists():
            raise ErrorETL(f"JAVA_HOME apunta a {java_home}, pero ahí no está Java. "
                           "Corrige la ruta o quita esa variable.")
    else:
        encontrado = shutil.which("java")
        if encontrado is None:
            raise ErrorETL(f"Falta Java. PySpark necesita Java {JAVA_MINIMO} o más reciente ({INSTALAR_JAVA}).")
        java = Path(encontrado)
    try:
        salida = subprocess.run([str(java), "-version"], capture_output=True, text=True, timeout=30).stderr
    except (OSError, subprocess.TimeoutExpired):
        return  # si no se puede preguntar la versión, que Spark lo intente
    m = re.search(r'version "(\d+)(?:\.(\d+))?', salida)
    if m:
        mayor = int(m.group(2) if m.group(1) == "1" and m.group(2) else m.group(1))  # "1.8" es Java 8
        if mayor < JAVA_MINIMO:
            raise ErrorETL(f"Tienes Java {mayor} y PySpark necesita Java {JAVA_MINIMO} o más reciente "
                           f"({INSTALAR_JAVA}).")


def crear_spark(nombre: str):
    from pyspark.sql import SparkSession

    _revisar_java()
    # PySpark avisa que no está probado con pandas >= 3. Aquí pandas solo arma la
    # base SQLite y las muestras del diccionario, y los resultados se validaron:
    # se silencia solo ese aviso para que se lea la salida del ETL.
    warnings.filterwarnings("ignore", message="PySpark does not yet fully support pandas",
                            category=FutureWarning)
    spark = (
        SparkSession.builder.master("local[*]").appName(nombre)
        # las fechas se guardan sin zona horaria: UTC evita que Spark las mueva
        .config("spark.sql.session.timeZone", "UTC")
        # pandas guarda las fechas en nanosegundos; Spark las lee como número
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        # Arrow pasa los datos de Spark a pandas por bloques de columnas (más rápido);
        # si en alguna compu no funciona, Spark regresa solo al método anterior.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .getOrCreate()
    )
    # Solo errores fatales en el log: un error conocido se explica en una línea (error_claro) y
    # uno desconocido llega a Python con toda la traza de Java, así que no se pierde nada.
    spark.sparkContext.setLogLevel("FATAL")
    return spark


def carpeta(dominio: str, nivel: str) -> Path:
    return (MARKET if dominio == "market" else INTRADAY) / nivel


def leer_capa(spark, dominio: str, nivel: str, capa: str, sub: str = ""):
    """Lee los CSV de una capa con su esquema. Silver lee bronze y gold lee silver.

    Lectura estricta: Spark compara el encabezado de cada CSV con el esquema
    (enforceSchema=False) y se detiene si no coincide; sin esto lee por posición
    y puede meter el cierre en la columna de apertura sin avisar. Con FAILFAST,
    un valor que no es del tipo esperado detiene la lectura en vez de quedar vacío.
    """
    origen = carpeta(dominio, nivel) / sub
    if not any(origen.glob("*.csv")):
        raise ErrorETL(f"No hay CSV en {origen.relative_to(ROOT)}. Corre antes la capa anterior.")
    return spark.read.csv(str(origen / "*.csv"), header=True, schema=esquema_spark(capa),
                          enforceSchema=False, mode="FAILFAST")


def redondear(df, capa: str):
    """Recorta los decimales largos que trae yfinance (5.724499225616455 → 5.72449923)."""
    from pyspark.sql import functions as F

    for c, t, _ in ESQUEMA[capa]:
        if t == "decimal":
            df = df.withColumn(c, F.round(F.col(c), DECIMALES))
    return df


FORMATO_FECHA = {
    "fecha": r"^\d{4}-\d{2}-\d{2}$",
    "fecha y hora": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
}


def validar(df, capa: str) -> None:
    """Revisa una capa antes de escribirla; si algo falla, lo que ya había en disco no se toca.

    Columnas iguales al esquema, al menos una fila, ticker siempre lleno y
    fechas con formato AAAA-MM-DD (más HH:MM:SS en las de fecha y hora).
    """
    from pyspark.sql import functions as F

    if df.columns != columnas(capa):
        raise ErrorETL(f"{capa}: las columnas {df.columns} no coinciden con el esquema {columnas(capa)}.")
    malas = {"ticker": F.col("ticker").isNull() | (F.trim("ticker") == "")}
    for c, t, _ in ESQUEMA[capa]:
        if t in FORMATO_FECHA:
            malas[c] = F.col(c).isNull() | ~F.col(c).rlike(FORMATO_FECHA[t])
    r = df.agg(F.count("*").alias("_filas"),
               *[F.sum(cond.cast("int")).alias(c) for c, cond in malas.items()]).first()
    problemas = ([] if r["_filas"] else ["no tiene filas"])
    problemas += [f"{r[c]:,} fila{'s' if r[c] > 1 else ''} con el campo {c} vacío o mal escrito"
                  for c in malas if r[c]]
    if problemas:
        raise ErrorETL(f"{capa} no pasó la revisión: {'; '.join(problemas)}. "
                       "No se escribió nada y los CSV anteriores siguen igual.")


def guardar_por_ticker(df, capa: str, destino: Path, orden: list[str]) -> int:
    """Escribe un CSV por ticker (destino/<TICKER>.csv), igual que un parquet por ticker.

    Primero valida la capa. Spark escribe carpetas part-*.csv en _spark_tmp; se
    juntan en un CSV por ticker dentro de _nuevo y solo al final reemplazan a los
    anteriores, así un error a medio camino no deja la capa vacía ni mezclada.
    Devuelve cuántos archivos quedaron.
    """
    from pyspark.sql import functions as F

    validar(df, capa)
    destino.mkdir(parents=True, exist_ok=True)
    tmp, nuevo = destino / "_spark_tmp", destino / "_nuevo"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(nuevo, ignore_errors=True)
    (
        df.withColumn("_archivo", F.col("ticker"))
        .repartition("_archivo")
        .sortWithinPartitions("_archivo", *orden)  # el escritor exige el orden por _archivo primero
        .write.partitionBy("_archivo")
        .option("header", True)
        .mode("overwrite")
        .csv(str(tmp))
    )
    nuevo.mkdir()
    for parte in sorted(tmp.glob("_archivo=*")):
        nombre = unquote(parte.name.split("=", 1)[1])
        archivos = sorted(parte.glob("part-*.csv"))
        with open(nuevo / f"{nombre}.csv", "w", encoding="utf-8") as out:
            for i, f in enumerate(archivos):
                lineas = f.read_text(encoding="utf-8").splitlines(keepends=True)
                out.writelines(lineas if i == 0 else lineas[1:])
    nuevos = sorted(nuevo.glob("*.csv"))
    if not nuevos:
        raise ErrorETL(f"Spark no dejó archivos para {capa}. Los CSV anteriores siguen igual.")
    # Cada CSV nuevo reemplaza al viejo de un jalón; después se borran los de tickers que ya no vienen.
    vigentes = {f.name for f in nuevos}
    for f in nuevos:
        f.replace(destino / f.name)
    for viejo in destino.glob("*.csv"):
        if viejo.name not in vigentes:
            viejo.unlink()
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(nuevo, ignore_errors=True)
    return len(nuevos)


def ticker_de_archivo():
    """Saca el ticker del nombre del parquet: .../WALMEX.MX.parquet → WALMEX.MX.

    Spark entrega la ruta como dirección web (^MXX llega como %5EMXX, un espacio
    como %20) y aquí se traduce de regreso. El + se protege antes porque
    url_decode lo convertiría en espacio.
    """
    from pyspark.sql import functions as F

    nombre = F.regexp_extract(F.input_file_name(), r"([^/]+)\.parquet$", 1)
    return F.url_decode(F.regexp_replace(nombre, r"\+", "%2B"))


# ── errores con mensaje claro ────────────────────────────────────────────────

def _ruta_legible(uri: str) -> str:
    ruta = unquote(re.sub(r"^file:/*", "/", uri))
    try:
        return str(Path(ruta).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return ruta


def error_claro(e: Exception) -> str | None:
    """Mensaje corto para los errores conocidos; None si conviene ver la traza completa."""
    if isinstance(e, ErrorETL):
        return str(e)
    texto = str(e)
    archivo = re.search(r"(file:/\S+?\.csv)", texto)
    ruta = _ruta_legible(archivo.group(1)) if archivo else "de esta capa"
    if "CSV header does not conform to the schema" in texto:
        m = re.search(r"Expected: (\S+) but found: (\S+)", texto)
        detalle = f": se esperaba la columna {m.group(1)} y viene {m.group(2)}" if m else ""
        return (f"El CSV {ruta} no tiene las columnas que espera esta capa{detalle}. "
                "Vuelve a correr la capa anterior para regenerarlo.")
    if "Malformed records are detected" in texto:
        m = re.search(r'For input string: "([^"]*)"', texto)
        detalle = f' ("{m.group(1)}")' if m else ""
        return (f"El CSV {ruta} trae un valor que no es del tipo esperado{detalle}. "
                "Vuelve a correr la capa anterior para regenerarlo.")
    return None


# ── diccionario de datos ─────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _tabla_md(capa: str, muestra: pd.DataFrame) -> list[str]:
    lineas = ["| Columna | Tipo | Descripción |", "|---|---|---|"]
    lineas += [f"| `{c}` | {t} | {d} |" for c, t, d in ESQUEMA[capa]]
    lineas += ["", f"Muestra: {muestra['ticker'].iloc[0]}", "",
               "| " + " | ".join(muestra.columns) + " |",
               "|" + "---|" * len(muestra.columns)]
    lineas += ["| " + " | ".join(_fmt(v) for v in r) + " |" for _, r in muestra.iterrows()]
    return lineas


def escribir_diccionario(capa: str, muestra: pd.DataFrame, n_archivos: int, script: str,
                         extra: tuple[str, pd.DataFrame, str] | None = None) -> None:
    dominio, nivel = capa.split("/")
    lineas = [
                 f"# {dominio} / {nivel}",
                 "",
                 f"{n_archivos} archivos CSV, uno por {'símbolo' if dominio == 'market' else 'emisora'}. "
                 f"Los genera `python -m {script}` (PySpark).",
                 "",
             ] + _tabla_md(capa, muestra)
    if extra:
        capa2, muestra2, sub = extra
        lineas += ["", f"## Subcarpeta `{sub}`: resumen por día", "",
                   f"{n_archivos} archivos CSV más, un día por fila.", ""]
        lineas += _tabla_md(capa2, muestra2)
    (carpeta(dominio, nivel) / "_diccionario.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def muestra(df, orden: str, n: int = 5, ticker: str = "WALMEX.MX", desde_inicio: bool = False) -> pd.DataFrame:
    """Unas filas de un ticker para el diccionario (WALMEX si existe)."""
    from pyspark.sql import functions as F

    sub = df.filter(F.col("ticker") == ticker)
    if sub.limit(1).count() == 0:
        sub = df.filter(F.col("ticker") == df.select("ticker").first()[0])
    sub = sub.orderBy(F.col(orden).asc() if desde_inicio else F.col(orden).desc()).limit(n)
    pdf = sub.toPandas()
    return pdf if desde_inicio else pdf.iloc[::-1].reset_index(drop=True)


# ── base SQLite para el agente ───────────────────────────────────────────────

def construir_db(precios: pd.DataFrame, intradia_diario: pd.DataFrame | None,
                 db_path: Path = DB_PATH) -> Path:
    """Base del agente: gold diario, tipo de cambio y resumen intradía por día.

    Las barras de 5 min no entran: son 1.1 millones de filas que el agente no
    necesita para responder; ML las lee de los CSV de data/intraday/gold/.
    """
    # Se insertan ordenadas por ticker y fecha: Spark no garantiza el orden de las filas y así la
    # base sale idéntica corra gold completo o con --solo (que lee del gold en disco).
    precios = precios.sort_values(["ticker", "fecha"], ignore_index=True)
    if intradia_diario is not None:
        intradia_diario = intradia_diario.sort_values(["ticker", "fecha"], ignore_index=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(tmp) as conn:
        conn.executescript("""
                           CREATE TABLE precios_historicos
                           (
                               ticker          TEXT NOT NULL,
                               fecha           TEXT NOT NULL,
                               open            REAL,
                               high            REAL,
                               low             REAL,
                               close           REAL,
                               volume          INTEGER,
                               moneda          TEXT,
                               usdmxn_close    REAL,
                               close_usd       REAL,
                               ret_1d          REAL,
                               target_next_day REAL,
                               es_valido       INTEGER,
                               PRIMARY KEY (ticker, fecha)
                           );
                           CREATE TABLE tipo_cambio
                           (
                               fecha TEXT PRIMARY KEY,
                               close REAL
                           );
                           CREATE TABLE intradia_diario
                           (
                               ticker         TEXT NOT NULL,
                               fecha          TEXT NOT NULL,
                               n_barras       INTEGER,
                               precio_primero REAL,
                               precio_ultimo  REAL,
                               precio_max     REAL,
                               precio_min     REAL,
                               ret_intradia   REAL,
                               volatilidad_5m REAL,
                               dia_completo   INTEGER,
                               dia_plano      INTEGER,
                               PRIMARY KEY (ticker, fecha)
                           );
                           """)
        precios.to_sql("precios_historicos", conn, if_exists="append", index=False)
        fx = precios.loc[(precios["ticker"] == FX) & (precios["close"] > 0), ["fecha", "close"]]
        fx.to_sql("tipo_cambio", conn, if_exists="append", index=False)
        if intradia_diario is not None:
            intradia_diario.to_sql("intradia_diario", conn, if_exists="append", index=False)
        conn.execute("CREATE INDEX idx_precios_fecha ON precios_historicos (fecha)")
    # El "with" de sqlite3 guarda los cambios pero no cierra la conexión, y Windows no deja
    # reemplazar un archivo abierto (PermissionError WinError 32): se cierra antes del replace.
    conn.close()
    tmp.replace(db_path)
    return db_path
