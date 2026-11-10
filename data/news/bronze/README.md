# Bronce: todas las noticias en Parquet

**Archivo principal:** `news.parquet`.

Cada fila representa una noticia, fuente y captura. Conserva también los registros repetidos y los valores sin limpiar. La clave lógica es `(batch_id, record_index)`.

Contiene `source`, `fetched_at`, `title_raw`, `text_raw`, `url_raw`, `published_at_raw` y `raw_payload` (registro JSON original completo), además de identificadores y metadatos de procedencia. Los cables DataBursátil incluyen su categoría. No se asignan emisoras en esta capa.

Las ingestas actualizan el Parquet automáticamente. Silver consume sus registros para limpiar y normalizar artículos.

## Respaldo y reconstrucción

Los archivos `<fuente>/YYYY-MM-DD/<batch_id>.json` conservan los sobres completos de captura. No se reemplazan al construir el Parquet.

Desde la raíz del proyecto, para cargar las capturas disponibles sin descargar ni recalcular silver/gold:

```bash
venv/bin/python scripts/build_news_medallion.py --bronze-only
```

Repetir el comando no duplica las posiciones de un lote. Una descarga nueva sí conserva su propia observación de cada noticia.

El Parquet y los JSON se versionan en Git. Ver [arquitectura completa](../../../docs/NEWS_MEDALLION.md).
