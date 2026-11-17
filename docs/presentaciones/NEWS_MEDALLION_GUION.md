# Guion de exposición: arquitectura medallion de noticias

Duración sugerida: 10–15 minutos. Fuente: [NEWS_MEDALLION.md](../NEWS_MEDALLION.md).

## 1. De noticias dispersas a indicadores con evidencia

Presenta el objetivo: ordenar noticias de múltiples fuentes y poder explicar de dónde sale cada indicador. La arquitectura ya está implementada localmente. Medallion organiza responsabilidades; los formatos de archivo no determinan las capas.

## 2. Tres capas, tres preguntas

Bronce responde qué recibimos; silver qué estructura común tienen los artículos; gold qué señales medimos. Los datos siguen siendo locales y las salidas anteriores se conservan para no romper consumidores. No se modificaron tablas SQL.

## 3. Cómo se conectan fuentes y capas

Las fuentes incluyen DataBursátil, Google News, GDELT y otros conectores. Trump es una extracción temática de Google News RSS. Los módulos existentes alimentan la nueva arquitectura. Se conserva compatibilidad con archivos anteriores.

## 4. Bronce: conservar cada captura

Cada descarga genera un batch_id y un archivo nuevo. La partición usa fecha UTC de captura. El sobre describe source, fetched_at, representation, origin y payload. No todas las capturas son respuestas HTTP completas: algunas son extracciones seleccionadas por conectores. Las credenciales no forman parte del sobre.

## 5. Silver: una estructura común para los artículos

El grano es artículo más asociación más captura. article_id se calcula con SHA-256 de fuente y URL, o titular normalizado si falta URL. Dentro de la captura se deduplica por article_id, ticker, topic y batch_id; entre capturas se conservan observaciones. No se resuelven redirecciones ni se agrupan automáticamente eventos semánticos.

## 6. La fecha de publicación no es la fecha de captura

published_at representa publicación; fetched_at captura; available_at es el máximo de ambas, o captura cuando falta publicación. Gold solo incluye observaciones disponibles antes del corte. DataBursátil conserva publicación nula si no existe evidencia. Los archivos importados sin fecha de captura quedan disponibles desde su importación, no retrospectivamente.

## 7. Gold: indicadores diarios con trazabilidad

La clave lógica es date, ticker y topic. Gold selecciona la última observación conocida de cada artículo y asociación, evitando contar capturas repetidas. Usa una ventana reciente y otra anterior. El sentimiento se calcula con keywords ES/EN y ponderaciones temporales y de similitud. news_count_7d es log(1+suma de pesos), no un conteo bruto. Puede diferir del flujo anterior con Gemini o Alpha Vantage.

## 8. Trump y DataBursátil: reglas diferentes

Trump entra desde scripts/ingesta_trump.py. titulo se transforma en title, fecha en published_at y texto en text. El CSV tiene cuatro columnas y omite texto; los resúmenes RSS se conservan en bronce y silver. Se marca excerpt si el resumen limpio difiere del titular, o headline si solo hay titular. No se asigna automáticamente una emisora. DataBursátil se vincula por alias con límites de palabra; sin coincidencias se conserva como market. Esa relación es léxica y no prueba exposición económica.

## 9. Dónde vive cada conjunto de datos

Explica el árbol sin confundir el latest anterior con el nuevo. gold/latest.json contiene la última invocación, incluso si tuvo un corte histórico. history y los JSON anteriores permanecen para compatibilidad. Las capas generadas están excluidas de Git.

## 10. Cómo ejecutar las noticias

Ejecutar desde la raíz con el entorno virtual. Los comandos descargan datos y actualizan las nuevas capas, además de mantener las salidas anteriores. El token DataBursátil se carga desde el entorno. Recalcular medallion sin opciones no descarga noticias. Una ventana vacía no equivale a éxito de ingesta ni a neutralidad del mercado.

## 11. Migrar, reconstruir y auditar

--migrate importa snapshots DataBursátil, articles.parquet histórico y Trump CSV si existe, sin borrar originales. Los hashes evitan importar dos veces el mismo archivo. --replay hace upsert por lote desde bronce. La trazabilidad permite ir de gold a silver y luego al archivo bronce. Borrar manualmente un lote no elimina sus observaciones silver con replay.

## 12. Qué garantiza y cuál es su alcance

Silver y gold se escriben mediante archivos temporales y reemplazo atómico con bloqueo local. No hay transacción única entre capas; bronce permite reprocesar. El Parquet completo se reescribe al actualizar, adecuado al volumen actual. No incluye un servidor MCP, scheduler ni cambios SQL. El siguiente paso de consumo puede ser un agente que consulte silver para evidencia y gold para indicadores, pero no se presenta como implementado.

