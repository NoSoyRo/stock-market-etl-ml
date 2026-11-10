# Guía sencilla: capa semántica de noticias

## ¿Qué es la capa semántica?

Piensa en el medallion como una bodega con cajas:

- **Bronze** conserva lo que se recibió de las fuentes.
- **Silver** organiza las observaciones de artículos: titular, fecha, fuente y
  asociación con una empresa o un tema.
- **Gold** guarda los indicadores calculados para una empresa o tema en una
  fecha.

La capa semántica es como una persona que sabe encontrar la caja correcta,
comprobar que sus etiquetas estén bien y explicar su contenido con nombres más
entendibles. No vuelve a calcular ni descarga noticias.

Por ejemplo, en vez de pedirle a una persona que busque manualmente en Parquet
una fila con `ticker="WALMEX"` y `topic=""`, se puede preguntar:

```bash
venv/bin/python scripts/query_news_semantic.py empresa --ticker WALMEX
```

La capa aplica esa selección exacta, devuelve las métricas guardadas y aclara
qué fecha, ventana, método y fuentes corresponden al resultado. Si se piden los
artículos, los busca en silver y devuelve los titulares que respaldan el
resultado.

**Importante:** la capa lee gold y silver. No descarga noticias, no modifica los
Parquet y no convierte un resultado de sentimiento en una recomendación de
compra o en una predicción del precio.

## Cómo funciona una consulta, paso a paso

Cuando se ejecuta una consulta empresarial, por ejemplo WALMEX, ocurre esto:

1. **Valida la pregunta.** Limpia espacios y normaliza el ticker a mayúsculas.
   Rechaza una fecha mal escrita o parámetros fuera de las reglas.
2. **Escoge el archivo correcto.** Sin `--fecha`, lee
   `data/news/gold/today_features.parquet`. Con `--fecha`, lee
   `data/news/gold/historical_features.parquet`.
3. **Busca la asociación exacta.** Para WALMEX busca `ticker="WALMEX"` y
   `topic=""`. Para el tema Trump busca `ticker=""` y `topic="trump"`. Así, las
   noticias de un tema no se atribuyen automáticamente a una empresa.
4. **Comprueba que el resultado sea utilizable.** Revisa que la fila exista,
   que no haya claves repetidas, que las métricas tengan valores válidos y que
   el cálculo corresponda a la fecha/corte solicitado.
5. **Traduce los campos.** Convierte nombres técnicos como `news_sent_score` a
   nombres claros como `sentimiento_general`, y agrega la ventana, las fuentes,
   la fecha del cálculo y advertencias.
6. **Devuelve JSON.** La misma estructura permite usar la consulta desde la CLI,
   un script o el agente.

El esquema resumido es:

```text
Pregunta
   ↓
Validación de parámetros
   ↓
Selección exacta de gold (empresa/tema + fecha)
   ↓
Validación de datos y antigüedad
   ↓
Traducción a nombres de negocio
   ↓
Respuesta JSON + advertencias + procedencia
```

Las piezas del código tienen una responsabilidad cada una:

- `models.py`: forma estándar de respuesta y validación de entradas.
- `repository.py`: lee Parquet y valida esquemas, fechas, duplicados y métricas.
- `service.py`: decide qué fila usar y aplica las reglas de las consultas.
- `interpretation.py` y `metrics.json`: convierten los nombres técnicos a
  términos de negocio y limitan lo que se puede concluir.

## ¿Cómo encuentra los artículos que respaldan gold?

Gold contiene los indicadores y listas de identificadores de artículos y
capturas que se consideraron para calcularlos. No guarda los titulares completos;
esos detalles están en silver.

Al consultar evidencia, la capa:

1. encuentra el resultado gold de la asociación y fecha;
2. toma sus `article_ids` y el corte `as_of`;
3. busca en silver observaciones de esos artículos para la misma asociación que
   ya estuvieran disponibles al corte;
4. limita los eventos a las dos ventanas usadas por el cálculo;
5. si un artículo aparece en varios lotes, conserva su última observación
   disponible al corte;
6. muestra titular, extracto si es distinto, fuente, enlace y fechas; y
   advierte si no pudo resolver las referencias esperadas.

Gold conserva por separado listas de artículos y de lotes, no parejas exactas
artículo-lote. Por eso la capa no combina esas listas como si cada artículo
perteneciera a cada lote. El lote que se presenta es el de la observación silver
seleccionada.

Si `published_at` falta, se mantiene como `null`; el evento se ubica con la fecha
de captura usada por el proceso del medallion. Si el resumen repite el titular,
la respuesta no lo muestra dos veces.

## Fechas y ventanas: un ejemplo sencillo

Una fila de gold para el **31 de octubre** significa: “indicadores calculados
para la fecha UTC 31 de octubre”. No significa que todas las noticias se hayan
publicado ese día. Normalmente usa una ventana reciente de siete días y una
ventana previa de siete días.

Gold guarda una versión por fecha, no todas las versiones intradía. Por eso:

- `empresa --ticker WALMEX` busca la fecha UTC actual;
- `empresa --ticker WALMEX --fecha 2026-10-30` busca exactamente esa fecha
  histórica;
- si se pide un corte anterior al `as_of` guardado, no se usa una fila calculada
  después de ese corte;
- si la tabla de hoy contiene una fecha distinta de hoy UTC, la respuesta avisa
  que está desactualizada;
- si se pide una ventana distinta a la guardada, la capa no recalcula en
  silencio: responde que las métricas son incompatibles.

El historial presenta los conteos de cada fecha por separado. No los suma,
porque dos ventanas de siete días en fechas consecutivas comparten muchos de los
mismos artículos.

## Qué puede y qué no puede concluir

La capa ayuda a contestar “¿qué indicador está guardado?” y “¿qué titulares lo
respaldan?”. No determina por qué ocurrió un evento ni prueba una relación
causal.

- Un score positivo significa balance positivo **según las palabras clave del
  cálculo**, no que la acción vaya a subir.
- Un score negativo tampoco es una predicción de caída.
- Un score cero con artículos es numéricamente neutral según el cálculo, no
  certeza de neutralidad económica.
- Sin artículos recientes, el sentimiento se presenta como desconocido; no se
  inventa un cero neutral.
- El volumen informativo ponderado no es el número de artículos. Para eso está
  `cantidad_articulos`.
- La dispersión del sentimiento no es volatilidad del precio.
- La cobertura ESG permanece desconocida hasta revisar los titulares.

## Cambios realizados

| Archivo | En palabras sencillas |
|---|---|
| `src/semantic/models.py` | Define el formato de las respuestas y valida nombres y fechas. |
| `src/semantic/metrics.json` | Explica los nombres de negocio y el significado de cada métrica. |
| `src/semantic/repository.py` | Lee los Parquet y revisa que sus datos y formatos sean válidos. |
| `src/semantic/service.py` | Realiza las consultas y aplica las reglas de fechas, empresas, temas y evidencia. |
| `src/semantic/interpretation.py` | Explica el resultado de sentimiento sin atribuirle significado financiero que no tiene. |
| `scripts/query_news_semantic.py` | Permite probar las consultas desde una terminal, sin usar un agente. |
| `tests/test_news_semantic.py` | Comprueba las consultas con datos de prueba, sin conectarse a internet. |
| `src/agent/tools.py` | Agrega una opción semántica explícita a `ver_noticias` y conserva sus opciones anteriores. |
| `docs/NEWS_SEMANTIC.md` | Documenta el contrato y ejemplos de consultas. |
| `docs/NEWS_MEDALLION.md` | Enlaza la documentación de la capa semántica. |

No se agregó un servidor MCP: el documento de implementación lo deja como una
posible integración futura, no como requisito de esta primera versión.

## ¿Qué significa cada dato?

- **Sentimiento general:** balance de palabras positivas o negativas encontradas
  en las noticias. No es una probabilidad de que suba o baje la acción.
- **Cantidad de artículos:** artículos distintos en la ventana reciente.
- **Volumen informativo ponderado:** indicador transformado de volumen; no es
  el número de artículos. Por eso se muestra junto a `window_days`, el tamaño
  real de la ventana.
- **Cambio de sentimiento:** diferencia entre ventanas dentro del cálculo gold.
  No es la diferencia entre el resultado de hoy y el de ayer.
- **Dispersión:** variación del sentimiento entre noticias; no es volatilidad
  del precio.
- **Sentimiento ESG:** se informa como desconocido hasta revisar los titulares.
  Un cero no demuestra que la empresa sea neutral en temas ESG.

Si no hay artículos recientes, la respuesta no inventa un sentimiento neutral.
Si falta un titular o lote que gold esperaba, la consulta de evidencia lo
advierte.

## Cómo probarlo

Ejecuta estos comandos desde la carpeta principal del proyecto.

### 1. Ejecutar las pruebas automáticas

```bash
venv/bin/python -m pytest tests/test_news_semantic.py tests/test_news_medallion.py -q
```

La última ejecución de validación también incluyó las pruebas del agente:

```bash
venv/bin/python -m pytest tests/test_news_semantic.py tests/test_news_medallion.py tests/test_investor_agent.py -q
```

Las pruebas usan archivos temporales y un reloj controlado. Comprueban, entre
otras cosas, que no se mezclen empresas y temas, que no se usen datos posteriores
al corte, que los duplicados se reporten y que los Parquet no se modifiquen.
En esta implementación, las 51 pruebas de esos tres archivos pasaron.

### 2. Consultar una empresa

```bash
venv/bin/python scripts/query_news_semantic.py empresa --ticker WALMEX
```

### 3. Consultar un tema

```bash
venv/bin/python scripts/query_news_semantic.py tema --topic trump
```

La respuesta del tema se refiere a `trump`; no se atribuye automáticamente a
WALMEX ni a otra empresa.

### 4. Consultar un rango de fechas

Usa fechas disponibles en tus datos, con formato `AAAA-MM-DD`:

```bash
venv/bin/python scripts/query_news_semantic.py historial --ticker WALMEX --inicio 2026-10-30 --fin 2026-10-31
```

El historial devuelve los resultados por fecha. No suma los conteos de días
distintos, porque sus ventanas de noticias pueden superponerse.

### 5. Consultar los artículos que respaldan un resultado

```bash
venv/bin/python scripts/query_news_semantic.py evidencia --ticker WALMEX --fecha 2026-10-31 --limit 10
```

Para un tema, cambia `--ticker WALMEX` por `--topic trump`. El número después de
`--limit` controla cuántos artículos se muestran. Puedes pedir la página
siguiente con `--offset`, por ejemplo:

```bash
venv/bin/python scripts/query_news_semantic.py evidencia --ticker WALMEX --fecha 2026-10-31 --limit 10 --offset 10
```

### 6. Revisar la cobertura disponible

```bash
venv/bin/python scripts/query_news_semantic.py cobertura
```

Esto muestra fechas, empresas, temas, ventanas y fuentes que aparecen en los
datos guardados. Que haya una fecha inicial y una final no significa que estén
guardados todos los días intermedios ni que todas las fuentes hayan funcionado.

## Cómo leer la respuesta

La CLI imprime JSON. Los campos más útiles son:

- `status`: resultado de la consulta.
- `query`: qué se pidió.
- `data`: métricas, artículos o cobertura.
- `warnings`: advertencias que conviene leer antes de interpretar el resultado.
- `provenance`: archivo y clave del dato de origen.

Estados frecuentes:

| Estado | Significado |
|---|---|
| `ok` | Hay datos que cumplen la consulta. |
| `not_found` | No hay un resultado guardado para esa empresa, tema o fecha. |
| `no_recent_articles` | Existe la fila gold, pero no tiene artículos recientes. |
| `stale` | La tabla del día no corresponde al día UTC actual o el cálculo está viejo. |
| `incompatible_metrics` | La ventana o el método no coincide con lo que se pidió comparar. |
| `invalid_input` | Hay un ticker, fecha u otro parámetro mal formado. |
| `data_unavailable` | Falta el archivo o no se puede leer. |
| `data_integrity_error` | El archivo contiene datos inconsistentes o duplicados. |

Una ausencia normal (`not_found`) no es un error de ejecución. La CLI devuelve
código `0` para consultas y ausencias normales, `2` para entradas inválidas y
`3` para archivos o datos inconsistentes.

## Opciones adicionales

Pide una ventana concreta sin recalcular los datos:

```bash
venv/bin/python scripts/query_news_semantic.py empresa --ticker WALMEX --window-days 7
```

También se puede pedir un corte de fecha y hora ISO-8601 con zona horaria:

```bash
venv/bin/python scripts/query_news_semantic.py empresa --ticker WALMEX --fecha 2026-10-31 --corte 2026-10-31T12:00:00Z
```

Gold guarda una versión por fecha. Si esa versión se calculó después del corte,
la consulta no usa información futura; puede responder que no hay una versión
disponible para ese corte.

Para consultar otra carpeta de datos, coloca `--news-root` antes del subcomando:

```bash
venv/bin/python scripts/query_news_semantic.py --news-root /ruta/a/news empresa --ticker WALMEX
```

## Uso desde el agente

La herramienta existente `ver_noticias` conserva su comportamiento anterior.
Para pedir la lectura semántica, se debe solicitar explícitamente:

```json
{"ticker": "WALMEX", "semantica": true}
```

Para incluir artículos de respaldo:

```json
{"ticker": "WALMEX", "semantica": true, "con_evidencia": true}
```

`en_vivo=true` es otra operación: consulta noticias en vivo. No se debe mezclar
con `semantica=true`; la consulta semántica usa únicamente los archivos locales.
