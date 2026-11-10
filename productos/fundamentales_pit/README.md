# Producto de datos: Fundamentales PIT (Equipo Rojo)

> **Fundamentales** = la salud financiera de una empresa, sacada de sus reportes
> trimestrales: cuánto vendió, cuánto le quedó limpio, si creció.
> **PIT (point-in-time)** = «lo que se sabía en ese momento».

Una empresa cierra el trimestre el 31 de marzo, pero lo publica semanas
después. Si un modelo usa esas cifras el 31 de marzo, está viendo el futuro y
sus resultados salen mejores de lo que serían en la vida real. Es como el
estado de cuenta de una tarjeta: el **día de corte** cierra el periodo, pero tú
no ves el estado de cuenta hasta días después. Aquí, `quarter_end` es el día de
corte y `available_date` el día en que ya se puede ver.

**Regla del producto:** una fila solo existe para quien consulta en una fecha
`>= available_date`. Vive en un solo lugar: `pit.fundamentales_al()`.

## Cómo correrlo

```bash
python -m productos.fundamentales_pit.capas todo            # silver + golden + calidad
python -m productos.fundamentales_pit.capas todo --bronze   # además baja la API (necesita token)
python build_dashboard.py                                   # dashboard desde golden
python -m pytest tests/test_fundamentales_pit.py tests/test_agents_mcp_fundamentales.py -q   # 28 tests
```

Sin token, `todo` construye silver desde `data/fundamentals.parquet`, que solo
trae ratios. Con token, `--bronze` guarda las respuestas crudas y silver sale de ahí.

## Las capas

| Capa | Archivo | Qué guarda |
|---|---|---|
| Bronze | `data/fundamentals/bronze/<ticker>/<financieros>/<periodo>.json` | La respuesta de la API **tal cual**, con la fecha de descarga. Nunca el token. Sirve para rehacer silver sin volver a llamar a la API. |
| Silver | `data/fundamentals/silver/fundamentales_trimestre.parquet` | Un trimestre por fila, **con montos** (ingresos, utilidades) y de dónde salió cada cifra. |
| Golden | `data/fundamentals/golden/fundamentales_pit.parquet` | Lo que consumen dashboard, agente y modelos. Columnas `fund_*` compatibles con `src/ml/features.py`. |
| Golden | `data/fundamentals/golden/dividendos_eventos.parquet` | Dividendos con sus tres fechas. Hoy vacío (ver Pendientes). |
| Calidad | `data/fundamentals/golden/calidad.md` | Cada regla del contrato revisada contra los datos. |

Todo `data/fundamentals/` está en `.gitignore`: se regenera con un comando.

## Archivos del producto

| Archivo | Qué es |
|---|---|
| `contrato.yaml` | El contrato de datos: columnas, tipos, llave, regla PIT, SLO, dueño y consumidores. Lo lee `calidad.py`, así que no es decorativo. |
| `capa_semantica.yaml` | Las definiciones de negocio (margen neto, YoY, UPA, dividendo), las reglas y las consultas de ejemplo. Se la pasa la herramienta al agente. |
| `pit.py` | `fundamentales_al(fecha, ticker)` y `dividendos_al(...)`: el único lugar donde vive la regla PIT. |
| `capas.py` | Orquestador: corre bronze → silver → golden → calidad. El código de cada capa está en `src/etl/` (ver abajo). |
| `../../src/etl/bronze/fundamentales_pit/capa.py` | Bronze: baja la API y guarda cada respuesta tal cual. |
| `../../src/etl/silver/fundamentales_pit/capa.py` | Silver: una fila por emisora y trimestre; gana la cifra original. |
| `../../src/etl/golden/fundamentales_pit/capa.py` | Golden: available_date, márgenes, YoY por fecha y UPA. |
| `calidad.py` | Revisión de golden contra el contrato. |
| `../../src/agent/fundamentales.py` | Herramienta del agente `ver_fundamentales`. |
| `../../sql/04_fundamentales_schema.sql` | Las mismas tablas para SQL Server, más la función `golden.fn_fundamentales_al(@fecha)`. |

## La herramienta del agente (`ver_fundamentales`)

Registrada en `build_tools()` (`src/agent/tools.py`). Tiene dos modos:

- **Resumen:** `{"ticker": "WALMEX", "fecha_consulta": "2025-01-15"}` devuelve los
  últimos trimestres **ya publicados** a esa fecha.
- **SQL:** `{"sql": "SELECT ...", "fecha_consulta": "..."}` corre un SELECT sobre
  `fundamentales_pit`, `dividendos_eventos` y `mapa_tickers`.

La regla PIT **no depende de que el modelo la respete**. La herramienta arma
una base SQLite en memoria que solo contiene las filas con
`available_date <= fecha_consulta`: los trimestres futuros **no existen** en esa
base. Un autorizador de SQLite rechaza todo lo que no sea leer (INSERT, DELETE,
DROP, ATTACH, PRAGMA…), las consultas que tardan más de 5 s se cortan, y una
`fecha_consulta` futura se recorta a hoy.

## Hallazgos (medidos el 2026-10-26)

1. **YoY mal calculado en el dato más reciente.** `src/api/fundamentals.py`
   calcula el crecimiento como `pct_change(4)`, o sea, contra la fila de 4
   posiciones atrás. Si falta un trimestre, compara contra el equivocado. En
   `data/fundamentals.parquet` pasa en **35 de 833** filas, incluidas **las 28 de
   2026T1** (todas las emisoras): comparaban contra 2024T2, no contra 2025T1.
   (Medido el 26-oct. Con el refresh del 6-nov del archivo del repo son **63 de
   1,146**: 31 en 2026T1, 31 en 2026T2 y 1 en 2026T3; todas con hueco real.)
   Golden lo calcula **por fecha** y deja NaN si no existe el mismo trimestre
   del año anterior. Nunca inventa un valor.
2. **Posible fuga por reexpresión.** Para ahorrar llamadas, la ingesta original
   pide solo años pares y saca los impares del comparativo del año siguiente.
   Ese comparativo se publica un año después y puede venir corregido.
   `capas.py` pide todos los años (`--no-estricto` para el modo viejo) y, si
   existen las dos versiones, gana la original. `reexpresado = True` marca
   cuando cambió.
3. **Huecos.** 30 huecos de trimestres; en casi todas las emisoras faltan
   2025T2–T4 porque solo llegan como comparativo de reportes que aún no existían.
   **Resuelto:** ahora se pide cada año directo (ver `docs/FUNDAMENTALES_INGESTA.md`, hallazgo 2).
4. **Bancos y aseguradoras.** GFNORTEO, GENTERA, GFINBURO y Q fallan siempre.
   Hipótesis, **sin verificar**: no reportan `revenue`/`grossprofit` (los bancos
   reportan intereses y margen financiero), así que todos sus márgenes salen NaN
   y la emisora se descarta. Silver ya no las descarta: las marca
   `formato = sin_ingresos_corporativos`. **Verificado con bronze (26-oct):** la
   hipótesis era falsa. La API responde 400 ("emisora desconocida") y en
   `/v2/emisoras` su campo `financieros` viene vacío: la API no ofrece sus estados
   financieros (hallazgo 4 de `docs/FUNDAMENTALES_INGESTA.md`).
5. **yfinance no es PIT.** `pe_ratio`, `market_cap_log`, `dividend_yield` y
   `profit_margin` (`src/ml/features.py`) son un dato de hoy repetido en toda la
   historia. **No están en este producto.** Le toca al equipo Rosa decidir si
   los quita de sus modelos.
6. **Recorte en ±5.** 38 ratios quedaron exactamente en ±5 por el `clip(-5, 5)`
   original, y su valor real se perdió. Ejemplo: margen operativo de HOTEL
   2020T2, en plena pandemia.
7. **El rezago de 70 días es fijo y conservador.** Según la corrida del equipo
   (no la reproduje: no tengo el token), GAPB entregó su 2026T3 el 2026-10-24,
   24 días después del cierre. No hay fuga (el dato se usa hasta el
   2026-12-09), pero llega tarde. Al 6-nov ya son 31 de 32 emisoras con 2026T3
   (la que falta es ALFAA, que no pasa de 2025T3).
8. **Cruce con Mercado.** Su ticker de Yahoo **no es** `ticker || '.MX'` en 4
   emisoras: TLEVISA → TLEVISACPO.MX, AMXL → AMXB.MX, VISTA → VISTAA.MX,
   POSADAS → POSADASA.MX. La tabla `mapa_tickers` lo resuelve.
9. **El agente no corre en Windows.** `src/ml/news_sources.py` importa `fcntl`
   (solo Linux/Mac), y `src/agent/__init__.py` lo arrastra. En Cloud Run
   funciona. Los tests ponen un sustituto vacío solo en Windows.

## Qué está verificado y qué no

| Pieza | Estado |
|---|---|
| Capas, contrato, capa semántica, herramienta | **Probado**: 28 tests (25 del producto y 3 de la herramienta del agente de dominio); los del producto detectan 5 bugs metidos a propósito |
| Tests del profe con la herramienta registrada | **Probado**: `tests/test_investor_agent.py` sigue pasando (8 de 8) |
| Golden y dashboard con datos reales | **Probado** con la ingesta real por capas: 1,368 filas, 32 de 36 emisoras (6-nov) |
| Bronze contra la API real | **Probado**: 1,440 llamadas (26-oct), reintento de trimestres recientes no publicados (30-oct, 67 llamadas) y refresh del 6-nov (67 llamadas, 0 errores, +11 filas) |
| Paso diario `fundamentales_pit` en `scripts/run_all_ingests.py` | **Probado** con datos de ejemplo y a mano con la API real. **No corrido en GitHub Actions**: su entorno empieza vacío (sin bronze) y no versiona datos |
| `query_fundamentales` del agente de dominio (`src/agents_mcp`) | **Probado** con golden de ejemplo (3 tests). **No probado de punta a punta** con el orquestador (requiere `ANTHROPIC_API_KEY`) |
| UPA desde la API | **Verificado**: viene como `basicearningslosspershare`; presente en el 100% de las filas |
| `sql/04_fundamentales_schema.sql` | Revisión de sintaxis; **no ejecutado** en SQL Server |

## Pendientes

- [x] Con token: `capas todo --bronze`, revisar el JSON de un banco y confirmar el hallazgo 4.
- [ ] Ver qué otros `financieros` ofrece la API (¿balance, acciones en circulación, dividendos?) y agregarlos a `CAMPOS`.
- [ ] Fuente de dividendos con fecha de anuncio. Mientras: `dividendos_plantilla.csv` → `data/fundamentals/dividendos_manual.csv`.
- [ ] Confirmar con el profe dónde vive golden (SQL Server o Postgres) y qué significa «webfinals».
- [ ] Avisar al equipo Rosa de los hallazgos 1 y 5, y al Negro del registro en `tools.py`.
