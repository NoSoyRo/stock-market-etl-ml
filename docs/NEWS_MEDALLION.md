# Arquitectura medallion de noticias

El proyecto mantiene un lago local en `data/news/bronze`, `silver` y `gold`. Su implementación está en `src/data/news_medallion.py`. Los JSON, CSV y Parquet anteriores se conservan para los consumidores existentes. Este cambio no ejecuta cargas en PostgreSQL ni altera las tablas `lab`.

Las consultas de negocio de solo lectura sobre gold/silver están descritas en
[NEWS_SEMANTIC.md](NEWS_SEMANTIC.md).

## Estructura física

```text
data/news/
├── bronze/
│   ├── news.parquet                # Una noticia por fuente y captura
│   ├── databursatil/YYYY-MM-DD/<batch_id>.json
│   ├── trump/YYYY-MM-DD/<batch_id>.json
│   ├── google_news/YYYY-MM-DD/<batch_id>.json
│   ├── gdelt/YYYY-MM-DD/<batch_id>.json
│   ├── <otra_fuente>/YYYY-MM-DD/<batch_id>.json
│   └── _imports/<hash>.json
├── silver/
│   └── articles.parquet
├── gold/
│   ├── historical_features.parquet # Tabla histórica
│   ├── today_features.parquet      # Tabla del día UTC
│   ├── daily_features.parquet      # Alias compatible de la histórica
│   └── latest.json                # Exportación JSON de la tabla del día
├── history/                       # Datos anteriores, conservados
├── noticias_trump.csv              # Exportación compatible de la ingesta Trump
└── latest.json                     # Indicadores del flujo anterior
```

Los JSON, CSV y Parquet de las tres capas, el CSV temático de Trump y los registros de importación se versionan en Git. El bloqueo de escritura permanece excluido. No se incluyen credenciales en los sobres de captura. La fecha de partición bronce es la fecha UTC de captura, no la fecha de publicación.

## Bronce: todas las noticias en un Parquet

**Tabla principal:** `data/news/bronze/news.parquet`.

**Grano:** una noticia recibida por fuente y captura. La clave lógica es `(batch_id, record_index)`; `record_id` es su SHA-256. Se conservan noticias repetidas dentro de una descarga y entre descargas, registros sin titular y fechas no interpretables. Bronce no limpia HTML, interpreta fechas de publicación, deduplica artículos ni expande asociaciones con emisoras.

Las ingestas actualizan esta tabla automáticamente después de guardar su captura JSON. Silver lee los registros de este Parquet y aplica las transformaciones posteriores. Los JSON siguen disponibles como respaldo del sobre completo, incluidos metadatos que no pertenecen a una noticia individual.

| Columna | Tipo lógico | Significado |
|---|---|---|
| `record_id` | texto | Identificador estable de la posición dentro del lote. |
| `batch_id` | texto | Identificador de captura. |
| `record_index` | entero | Posición de la noticia dentro de la captura, desde cero. |
| `source` | texto | Fuente declarada en el registro o, si falta, fuente del conector. |
| `capture_source` | texto | Fuente del sobre de captura. |
| `fetched_at` | timestamp UTC | Fecha de captura, conservada al reprocesar. |
| `bronze_path` | texto | Ruta relativa al JSON de respaldo. |
| `representation` | texto | Naturaleza del contenido: respuesta API, extracción o importación. |
| `origin` | texto nullable | Archivo anterior de procedencia, si fue importado. |
| `record_kind` | texto | `article`, `noticia` o `cable`. |
| `category` | texto | Categoría de los cables DataBursátil; vacío en los demás casos. |
| `source_record_id` | texto nullable | Identificador original `id` o `guid`, cuando existe. |
| `title_raw` | texto nullable | Titular tal como está disponible en la captura. |
| `text_raw` | texto nullable | Texto o extracto disponible sin limpieza adicional. |
| `url_raw` | texto nullable | Enlace recibido. |
| `published_at_raw` | texto nullable | Fecha de publicación sin interpretar; los valores numéricos se representan como texto. |
| `raw_payload` | texto JSON | Registro original completo, incluidos campos no proyectados en otras columnas. |

Los campos terminados en `_raw` preservan lo disponible en la captura; si un conector ya lo había limpiado, el Parquet no puede recuperar la versión anterior. `raw_payload` conserva también los tipos originales, mientras las proyecciones comunes son textuales para admitir todas las fuentes.

En los snapshots DataBursátil, cada noticia y cada cable producen una fila antes de buscar coincidencias con emisoras. Una fila bronce puede originar varias asociaciones silver. Los contadores de créditos y las listas vacías no se convierten en noticias; siguen conservados en el JSON de respaldo. Una captura vacía no agrega filas.

### Consultar y reconstruir la tabla

```python
import pandas as pd

bronze = pd.read_parquet("data/news/bronze/news.parquet")
trump = bronze[bronze["source"] == "trump"]
print(trump[["title_raw", "published_at_raw", "fetched_at"]].head())
```

Para cargar las capturas JSON existentes o recuperar el Parquet sin descargar noticias ni recalcular silver/gold:

```bash
venv/bin/python scripts/build_news_medallion.py --bronze-only
```

Reprocesar una captura reemplaza únicamente sus posiciones en la tabla, sin agregar duplicados del mismo lote. Si una noticia aparece en un lote nuevo, se conserva como una observación nueva. La escritura usa bloqueo y reemplazo atómico. El Parquet completo se reescribe al actualizarse; se mantiene la limitación de volumen local descrita al final.

### Capturas JSON de respaldo

Cada ingesta escribe un archivo nuevo con identificador aleatorio. Repetir una descarga conserva ambas capturas, incluso si son iguales. No se sobrescribe una captura anterior.

| Campo del sobre | Contenido |
|---|---|
| `schema_version` | Versión del sobre: `1`. |
| `batch_id` | Identificador único de captura. |
| `source` | Fuente o conector. |
| `fetched_at` | Fecha y hora UTC de captura o de importación. |
| `representation` | Naturaleza del contenido preservado. |
| `origin` | Ruta de un archivo anterior, si es una importación; de otro modo null. |
| `payload` | Datos capturados. |

Las representaciones distinguen `api_payload` (snapshot de noticias/cables de DataBursátil), `rss_extract` (campos seleccionados del RSS Trump), `normalized_extract` (salida de los conectores existentes, antes de deduplicar en `news.py`) y `legacy_import` (archivo anterior). **No todas las capturas son respuestas HTTP completas**: los conectores existentes ya seleccionan campos. Bronce conserva exactamente el contenido recibido en esta frontera, sin presentarlo como XML o JSON HTTP original cuando no lo es.

## Silver: observaciones normalizadas

`silver/articles.parquet` contiene una fila por artículo, asociación y captura. Conserva observaciones repetidas entre capturas para auditar qué datos se conocían en cada momento. Dentro de la misma captura elimina duplicados exactos de la clave `(article_id, ticker, topic, batch_id)`.

| Columna | Tipo lógico | Significado |
|---|---|---|
| `article_id` | texto | SHA-256 de fuente y URL; sin URL, usa el titular normalizado. |
| `batch_id` | texto | Captura de origen. |
| `bronze_path` | texto | Ruta relativa al JSON bronce. |
| `source` | texto | Fuente del artículo. |
| `ticker` | texto | Emisora asociada; vacío si no existe asociación. |
| `topic` | texto | Tema, como `trump` o `market`; puede estar vacío. |
| `title` | texto | Titular sin espacios exteriores. |
| `text` | texto nullable | Extracto limpio distinto del titular; nulo si falta o repite `title`. |
| `text_kind` | texto | `headline` o `excerpt`; no implica cuerpo completo. |
| `url` | texto | Enlace disponible, incluido el enlace intermediario de Google News. |
| `published_at` | timestamp UTC nullable | Publicación, si está disponible y es interpretable. |
| `fetched_at` | timestamp UTC | Momento de observación o importación. |
| `available_at` | timestamp UTC | Máximo entre publicación y captura; sin publicación, usa captura. |
| `sentiment` | decimal nullable | Puntuación proporcionada por la fuente; nulo si no fue evaluada. Cero significa una evaluación neutral. |
| `relevance_score` | decimal | Relevancia recibida; por defecto uno. |

Los textos se comparan después de limpiar HTML, entidades y espacios, sin distinguir mayúsculas, para evitar almacenar el titular dos veces. Los valores nulos de `text` no se convierten en la cadena "None" ni "nan" al analizar. Los ceros que `Article` añadía por defecto antes de evaluar las capturas antiguas se reconocen como no evaluados y se normalizan a nulo. Se preservan los ceros realmente proporcionados por una fuente. Gold calcula su propio sentimiento por palabras clave.

Las filas sin titular se descartan. Una misma noticia puede tener asociaciones con varias emisoras. La coincidencia por URL no intenta resolver redirecciones ni identificar que dos medios describen el mismo evento.

### DataBursátil

Los snapshots completos se vinculan a emisoras mediante coincidencias de alias con límites de palabra. Las noticias sin coincidencias se conservan con tema `market`. Esta asociación es léxica; no demuestra exposición económica.

`published_at` se mantiene nulo para DataBursátil, incluso si su conector anterior había sintetizado una hora. No se presenta la captura como una publicación real.

### Trump

`scripts/ingesta_trump.py` conserva la exportación CSV y añade capturas bronce y filas silver. La correspondencia es `titulo → title`, `fecha → published_at`, `texto → text`; la fuente es `trump` y el tema `trump`. El CSV público contiene solo `titulo`, `fecha`, `url` y `fuente`. El resumen RSS limpio se conserva en el payload bronce y en `silver.text`, no en el CSV. `text_kind` es `excerpt` cuando el resumen aporta un texto distinto al titular; sin resumen distinto, `text` queda nulo y se marca `headline`. El titular sigue disponible en `title` y se usa directamente al calcular indicadores. Los HTML y las entidades se limpian con una función común. Las importaciones de CSV de cuatro columnas conservan el titular en `title` y dejan `text` nulo; los CSV anteriores de cinco columnas siguen siendo compatibles.

El flujo medallion conserva las noticias de Trump como tema sin asignar automáticamente emisoras. Los scripts experimentales en `src/etl/shared/` son salidas separadas: `mapeo_trump.py` usa reglas heurísticas y `integrar_tfidf.py` combina noticias con WALMEX como ejemplo, sin demostrar exposición económica. Este último recupera los extractos por URL desde silver y admite el CSV sin `texto`.

Las noticias temáticas no se asignan automáticamente a emisoras. Sus indicadores gold se calculan para `ticker=""`, `topic="trump"`.

## Gold: indicadores con evidencia

Gold tiene dos tablas principales con el mismo esquema:

| Tabla | Archivo | Comportamiento |
|---|---|---|
| Histórica | `gold/historical_features.parquet` | Conserva los resultados de cada fecha UTC. Clave lógica `(date, ticker, topic)`. Un recálculo sustituye las filas de su fecha y conserva las demás. |
| Del día | `gold/today_features.parquet` | Contiene exclusivamente los resultados del día UTC actual. Se reemplaza al actualizar hoy; una consulta histórica no la sustituye. |

`gold/daily_features.parquet` se mantiene como exportación compatible de la histórica. Si todavía no existe la nueva tabla histórica, se inicializa con este archivo anterior. `gold/latest.json` exporta la tabla del día, no el último corte histórico solicitado.

La tabla del día contiene **indicadores calculados hoy**, con la ventana de siete días por defecto; no se limita a noticias publicadas hoy. La fecha de hoy se determina en UTC, como el resto del lago. Al cambiar de fecha, la siguiente ejecución descarta filas del día anterior de la tabla del día. Si solo se calcula un corte histórico y todavía no hay resultados para hoy, esta queda vacía. No hay un proceso permanente que actualice los archivos a medianoche.

Para reconstruir la histórica se calcula cada corte diario desde la primera fecha de disponibilidad conocida en silver hasta el corte solicitado, usando el fin de cada día y el corte exacto para el último. No se anticipa información importada después ni se inventan artículos en días sin evidencia:

```bash
venv/bin/python scripts/build_news_medallion.py --rebuild-gold-history
# También admite --as-of y --days.
```

Los grupos sin noticias en las ventanas consideradas no generan filas, por lo que la histórica no garantiza una cuadrícula completa para todos los días y emisoras.

Antes de calcular, se excluyen observaciones con `available_at` posterior al corte y se toma la última observación disponible de cada artículo/asociación. Se consideran hasta dos ventanas para comparar sentimiento reciente y anterior. Así, las capturas repetidas no multiplican el número de noticias.

| Campo | Significado |
|---|---|
| `date`, `as_of` | Fecha UTC de la fila y corte exacto utilizado. |
| `ticker`, `topic` | Asociación empresarial o temática. |
| `window_days` | Tamaño de la ventana reciente; siete por defecto. |
| `scoring_method` | `keywords`: sentimiento ES/EN reproducible, sin llamadas a Gemini. |
| `article_ids`, `batch_ids` | Listas JSON de identificadores de los artículos y observaciones empleados, incluidas ambas ventanas. |
| `publication_date_missing` | Cantidad de artículos considerados sin publicación conocida. |
| `sources` | Fuentes de los artículos de la ventana reciente. |
| `n_articles` | Número de artículos de la ventana reciente. |
| `news_sent_score` | Sentimiento reciente ponderado. |
| `news_count_7d` | Logaritmo `log(1 + suma de pesos)`; no es el conteo bruto. El nombre se conserva aunque `window_days` sea distinto de siete. |
| `news_sent_delta` | Sentimiento reciente menos sentimiento de la ventana previa. |
| `news_sent_vol` | Desviación ponderada del sentimiento. |
| `news_relevance_avg` | Relevancia media ponderada. |
| `news_esg_score` | Sentimiento de titulares identificados como ESG. |

Cuando falta publicación, la ventana usa el momento de captura y se registra la ausencia. No se reconstruyen fechas de publicación inexistentes. Los grupos sin observaciones en ambas ventanas no generan filas; un resultado vacío no implica que la descarga haya sido exitosa ni que todo el mercado sea neutral.

Gold se reconstruye desde silver y aplica el cálculo común de `news.py` con ponderación temporal y de similitud. Puede diferir del JSON anterior cuando este utiliza Gemini, sentimiento de Alpha Vantage o una cobertura distinta. No reemplaza silenciosamente el histórico anterior de backtesting.

## Integración y ejecución

Se conectaron los siguientes puntos:

- `news.py`: conserva los artículos de todas las fuentes antes de su deduplicación local. Al guardar un lote con caché, también construye gold.
- `scripts/fetch_news.py`: al guardar resultados, construye gold desde silver.
- `databursatil_news.save_daily_snapshot`: conserva el snapshot en bronce y normaliza silver. El script `databursatil_news_snapshot.py` también construye gold.
- `news_history.py`: conserva los nuevos resultados de consultas históricas GDELT. `build_news_history.py` construye gold al terminar, con corte actual; no inventa disponibilidad histórica.
- `scripts/ingesta_trump.py`: conserva la extracción temática, normaliza silver y construye gold cuando hay noticias.

Desde la raíz del proyecto:

```bash
# Importar datos locales anteriores y construir gold, sin descargar noticias.
venv/bin/python scripts/build_news_medallion.py --migrate

# Reconstruir silver desde bronce y recalcular gold.
venv/bin/python scripts/build_news_medallion.py --replay

# Consultar un corte reproducible con una ventana explícita.
venv/bin/python scripts/build_news_medallion.py --as-of 2026-10-29T23:59:59Z --days 7

# Ingestas habituales: generan las nuevas capas y las salidas anteriores.
venv/bin/python scripts/fetch_news.py --universe bmv
venv/bin/python scripts/databursatil_news_snapshot.py
venv/bin/python scripts/ingesta_trump.py
```

`--replay` vuelve a materializar cada captura en `bronze/news.parquet` y silver mediante un upsert por lote. Permite recuperar silver después de perder su Parquet. No elimina observaciones de lotes que hayan sido borrados manualmente de bronce.

### Migración conservadora

`--migrate` importa snapshots fechados de DataBursátil, `history/articles.parquet` y el CSV Trump si existe. Usa un registro de hashes para no duplicar importaciones iguales. Conserva los archivos originales. Si cambia un archivo, crea una nueva captura.

Los snapshots con `fetched_at` conservan ese momento. Cuando un archivo anterior no registra su fecha de captura, se utiliza la fecha de importación, sin afirmar que el sistema ya lo conocía en su fecha de publicación. Por ello un corte antiguo puede no tener resultados para artículos importados después.

Los indicadores agregados anteriores no se convierten en artículos ni se mezclan automáticamente con gold: no permiten reconstruir su evidencia. Si los artículos disponibles son antiguos, gold reciente puede estar vacío hasta ejecutar nuevas ingestas.

## Consistencia y alcance

Bronce Parquet, silver y gold usan escritura temporal y reemplazo atómico; un bloqueo local serializa sus modificaciones concurrentes. Bronce es la evidencia persistida y permite repetir la normalización tras una interrupción. No hay transacción única entre capas.

Esta versión usa Parquet local completo: cada actualización de silver lee y escribe su conjunto. Es adecuada para el volumen actual; volúmenes mayores requerirán particiones y un catálogo o motor transaccional. No incorpora un servidor MCP, una programación de tareas ni cambios en las tablas SQL.
