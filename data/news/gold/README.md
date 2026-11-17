# Gold: tabla histórica y tabla del día

| Tabla | Archivo | Grano |
|---|---|---|
| Histórica | `historical_features.parquet` | Una fila por fecha UTC, emisora y tema con evidencia disponible. |
| Del día | `today_features.parquet` | Una fila por emisora y tema para la fecha UTC actual. |

Ambas contienen sentimiento, volumen ponderado, cambio de sentimiento, dispersión, relevancia, sentimiento ESG y trazabilidad (`article_ids`, `batch_ids`, fuentes, fecha de corte y ventana).

La histórica conserva fechas anteriores y actualiza las filas de la fecha recalculada. La tabla del día se reemplaza al calcular hoy. Calcular un día histórico no sustituye los resultados actuales. La próxima ejecución después de un cambio de día elimina de la tabla del día los registros de ayer.

**Del día significa calculado hoy**, con ventana de siete días por defecto, no únicamente artículos publicados hoy. Las fechas se determinan en UTC.

Salidas compatibles:

- `daily_features.parquet`: copia de la tabla histórica para consumidores anteriores.
- `latest.json`: exportación de la tabla del día; puede ser una lista vacía.

Desde la raíz del proyecto:

```bash
# Actualizar los resultados con los datos disponibles.
venv/bin/python scripts/build_news_medallion.py

# Reconstruir los cortes diarios desde la primera disponibilidad conocida.
venv/bin/python scripts/build_news_medallion.py --rebuild-gold-history
```

Los datos se versionan en Git. Ver [arquitectura completa](../../../docs/NEWS_MEDALLION.md).
