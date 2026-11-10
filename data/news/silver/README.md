# Silver: artículos normalizados

Archivo generado: `articles.parquet`.

Cada fila representa un artículo asociado a una emisora o tema en una captura. Incluye identificadores, titular, texto disponible, URL, fuente, fechas UTC y referencia al archivo bronce.

Se eliminan duplicados dentro de cada captura y se conservan observaciones de capturas diferentes para auditoría. Las noticias de Trump se clasifican con tema `trump`, sin asignarlas automáticamente a emisoras.

Reglas de contenido:

- `title` conserva siempre el titular.
- `text` solo contiene un extracto distinto. Si falta o repite el titular después de limpiar HTML y espacios, queda nulo y `text_kind` es `headline`.
- `sentiment` queda nulo cuando no fue evaluado. Un cero proporcionado por la fuente se conserva como neutral; no se asigna cero por ausencia de evaluación.
- Gold utiliza el titular y el extracto disponible, y calcula su propio sentimiento.

Los datos se generan localmente y se versionan en Git junto con este README.

Ver [arquitectura completa](../../../docs/NEWS_MEDALLION.md).
