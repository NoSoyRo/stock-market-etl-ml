# intraday / silver

36 archivos CSV, uno por emisora. Los genera `python -m src.etl.silver.mercado.etl_silver` (PySpark).

| Columna | Tipo | Descripción |
|---|---|---|
| `ticker` | texto | Emisora |
| `fecha_hora` | fecha y hora | Marca de la barra, YYYY-MM-DD HH:MM:SS |
| `fecha` | fecha | Día de la barra |
| `precio` | decimal | Último precio negociado en esa marca |
| `flag_festivo` | 0/1 | 1 si ese día no hubo sesión en el diario de la BMV y todo el día trae el mismo precio (barras de relleno) |
| `flag_falla_fuente` | 0/1 | 1 si ese día casi todas las emisoras traen un solo precio a la vez: la fuente no mandó datos y rellenó |
| `flag_dia_plano` | 0/1 | 1 si la emisora tuvo un solo precio todo el día (en emisoras poco líquidas puede ser real: no hubo operaciones) |
| `flag_dia_incompleto` | 0/1 | 1 si el día tiene menos de 79 barras |

Muestra: WALMEX

| ticker | fecha_hora | fecha | precio | flag_festivo | flag_falla_fuente | flag_dia_plano | flag_dia_incompleto |
|---|---|---|---|---|---|---|---|
| WALMEX | 2026-10-30 13:40:00 | 2026-10-30 | 45.68 | 0 | 1 | 1 | 0 |
| WALMEX | 2026-10-30 13:45:00 | 2026-10-30 | 45.68 | 0 | 1 | 1 | 0 |
| WALMEX | 2026-10-30 13:50:00 | 2026-10-30 | 45.68 | 0 | 1 | 1 | 0 |
| WALMEX | 2026-10-30 13:55:00 | 2026-10-30 | 45.68 | 0 | 1 | 1 | 0 |
| WALMEX | 2026-10-30 14:00:00 | 2026-10-30 | 45.68 | 0 | 1 | 1 | 0 |
