# Datos de noticias

La arquitectura medallion se organiza en tres capas:

| Capa | Contenido | Formato generado |
|---|---|---|
| [Bronce](bronze/README.md) | Una noticia por fuente y captura en `news.parquet`, con JSON de respaldo. | Parquet y JSON |
| [Silver](silver/README.md) | Artículos normalizados y observaciones auditables. | Parquet |
| [Gold](gold/README.md) | Tabla histórica (`historical_features.parquet`) y del día (`today_features.parquet`), con indicadores por emisora o tema. | Parquet y JSON |

Los datos de estas capas se generan localmente y se versionan: las capturas JSON, el CSV de Trump y los Parquet están disponibles en GitHub junto con los README. `history/` y los JSON anteriores se conservan por compatibilidad; no representan las nuevas capas.

## Ejecución

Desde la raíz del proyecto:

```bash
venv/bin/python scripts/databursatil_news_snapshot.py
venv/bin/python scripts/fetch_news.py --universe bmv
venv/bin/python scripts/ingesta_trump.py
```

Para importar archivos anteriores y calcular gold sin descargar noticias:

```bash
venv/bin/python scripts/build_news_medallion.py --migrate
```

Ver [documentación completa](../../docs/NEWS_MEDALLION.md).
