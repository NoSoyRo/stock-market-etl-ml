# Noticias — PySpark

Implementación PySpark de las capas solicitadas:

| Capa | Código | Función |
| --- | --- | --- |
| Bronce | `bronce/news/noticias.py` | Lee capturas JSON, conserva registros originales y trazabilidad de lote. |
| Silver | `silver/news/noticias.py` | Limpia HTML, normaliza fechas, identifica empresas en cables DataBursátil y deduplica dentro del lote. |
| Golden | `golden/news/noticias.py` | Deduplica entre lotes y agrega noticias por ticker y tema a una fecha de corte. |

Desde la raíz del repositorio, con Java 17 y el entorno del proyecto activo:

```bash
pip install -r requirements-spark.txt
python -m src.etl.ejecutar_noticias_pyspark \
  --as-of 2026-10-31T23:59:59Z --days 7
```

El origen predeterminado es `data/news/bronze/<fuente>/<fecha>/<lote>.json`,
con los sobres creados por el pipeline actual (`schema_version`, `batch_id`,
`source`, `fetched_at`, `representation`, `payload`). No consulta APIs ni necesita
credenciales. Primero deben existir capturas; se falla explícitamente si no hay.
Admite noticias normalizadas y snapshots DataBursátil con `noticias` y `cables`.

Las salidas son directorios Parquet de Spark en `data/news/noticias/bronce`,
`silver` y `golden`. Cada ejecución reconstruye las tres salidas completas;
golden es el resultado del corte solicitado, no un historial de cortes.
Las capturas originales y las salidas del pipeline pandas se conservan.
Se pueden cambiar las rutas mediante `--capturas` y `--salida`.

Golden incluye conteo, media, dispersión poblacional y variación del sentimiento
frente a la ventana anterior, relevancia media, fuentes y IDs de evidencia.
Usa el diccionario de sentimiento existente con medias sin ponderación;
sus columnas no sustituyen las features ML con ponderación temporal y TF-IDF.
La disponibilidad es el máximo entre publicación y captura; DataBursátil usa
la captura porque no aporta evidencia fiable de publicación. Solo intervienen
registros disponibles en el corte y eventos de las dos últimas ventanas.

Validación:

```bash
python -m pytest tests/test_news_pyspark.py -q
```
