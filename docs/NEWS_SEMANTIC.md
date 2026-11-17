# Capa semántica de noticias

La capa semántica consulta las tablas gold y silver existentes en `data/news/` sin
activar ingestas, escribir archivos ni requerir red. El servicio está en
`src/semantic/service.py`; los contratos de métricas versionados están en
`src/semantic/metrics.json`.

## Consultas

Desde la raíz del proyecto:

```bash
venv/bin/python scripts/query_news_semantic.py empresa --ticker WALMEX
venv/bin/python scripts/query_news_semantic.py tema --topic trump
venv/bin/python scripts/query_news_semantic.py historial --ticker WALMEX --inicio 2026-10-30 --fin 2026-10-31
venv/bin/python scripts/query_news_semantic.py evidencia --topic trump --fecha 2026-10-31 --limit 10
venv/bin/python scripts/query_news_semantic.py cobertura
```

Para consultar un Parquet sintético o una raíz alternativa, pase
`--news-root /ruta/a/news` antes del subcomando. También hay opciones para limitar
la ventana, establecer un corte ISO-8601 y paginar evidencia. `comparar` compara
dos fechas guardadas sin rellenar días ausentes.

La salida de cada comando es un objeto JSON con `schema_version`, `status`,
`query`, `data`, `warnings` y `provenance`. Códigos de salida: `0` para consultas
resueltas o ausencia normal de datos, `2` para entradas inválidas y `3` para
archivos/esquemas no disponibles o inconsistentes.

## Reglas relevantes

- Empresa selecciona `(ticker normalizado, topic="")`; tema selecciona
  `(ticker="", topic normalizado)`. No se atribuye un tema a una empresa.
- `hoy` es el día UTC. Gold del día desfasado, o un cálculo demasiado antiguo,
  produce estado `stale`. Una fecha histórica busca una coincidencia exacta.
- Las respuestas muestran `window_days` junto al volumen informativo ponderado.
  Los conteos de días distintos no se suman, porque sus ventanas pueden
  superponerse.
- Un resultado con cero artículos recientes no se describe como neutralidad. Un score nulo tampoco se interpreta como neutral: indica sentimiento no disponible.
  Los scores ESG se mantienen nulos hasta verificar cobertura en los titulares.
- La evidencia filtra por asociación, IDs y disponibilidad al corte, y selecciona
  la última observación por artículo entre los lotes declarados por gold. Nunca sustituye evidencia con capturas ajenas. Informa explícitamente IDs/lotes ausentes.
- `news_sent_delta` conserva la fórmula de gold; la respuesta advierte cuando no
  se verificó la evidencia de ambas ventanas. No es la diferencia entre filas
  diarias.
- El modo `semantica=true` de `ver_noticias` es explícito y opt-in. Su modo
  anterior (señal guardada) y `en_vivo=true` conservan sus contratos.

## Ejemplos reales verificados

Estos ejemplos se generaron contra los Parquet presentes en el repositorio; son
una instantánea, no valores garantizados en ejecuciones futuras.

Consulta empresarial:

```bash
venv/bin/python scripts/query_news_semantic.py empresa --ticker WALMEX
```

Al consultar el gold del 2026-10-31, la respuesta encontró 7 artículos recientes
para WALMEX, con ventana de 7 días, método `keywords` y fuentes `databursatil`,
`google_news` y `expansion`. El score fue `-0.0053`; la interpretación comunica el
signo según palabras clave, no una recomendación ni una predicción de precio.

Consulta de evidencia:

```bash
venv/bin/python scripts/query_news_semantic.py evidencia --ticker WALMEX --fecha 2026-10-31 --limit 3
```

La consulta resolvió completamente los IDs y lotes declarados por gold. Entre los
tres primeros titulares, el artículo “07:45 Walmart México principales puntos
conferencia tercer trimestre” no tenía fecha de publicación ni extracto; la
respuesta conservó ambos como nulos y mostró su captura y ruta relativa de bronce.

Consulta temática:

```bash
venv/bin/python scripts/query_news_semantic.py tema --topic trump
```

En el mismo corte UTC 2026-10-31, la respuesta devolvió `topic="trump"` con 60
artículos recientes, fuente `trump` y score `-0.0049`. Su clave gold conserva
`ticker=""`; no se atribuye el resultado a WALMEX ni a `market`.

Consulta histórica:

```bash
venv/bin/python scripts/query_news_semantic.py historial --ticker WALMEX --inicio 2026-10-30 --fin 2026-10-31
```

La respuesta incluyó las filas observadas en ambas fechas y 7 artículos por fila.
Los conteos se presentan por día y no se suman, porque las ventanas de siete días
se superponen.

## Conteo histórico único

`consultar_historial(..., incluir_conteo_unico=True)` añade `data.unique_recent_articles`.
Resuelve la evidencia de cada fila histórica y cuenta una sola vez cada `article_id`
presente en sus ventanas recientes. Comprueba que el número resuelto por fila
coincida con `n_articles`. No suma conteos diarios.

El total comprende las ventanas recientes de las fechas observadas: puede incluir
publicaciones anteriores al inicio del rango consultado. No representa únicamente
artículos publicados dentro de ese rango. Si falta silver, la evidencia está
incompleta o los conteos no coinciden, el total es `null` y se incluye una
advertencia; las métricas históricas siguen disponibles.
