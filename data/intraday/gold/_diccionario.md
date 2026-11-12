# intraday / gold

36 archivos CSV, uno por emisora. Los genera `python -m src.etl.gold.mercado.etl_gold` (PySpark).

| Columna | Tipo | Descripción |
|---|---|---|
| `ticker` | texto | Emisora |
| `fecha_hora` | fecha y hora | Marca de la barra |
| `fecha` | fecha | Día de la barra |
| `hora` | entero | Hora de la marca (7 a 14) |
| `minuto` | entero | Minuto de la marca (0, 5, ..., 55) |
| `precio` | decimal | Último precio negociado en esa marca |
| `ret_5m` | decimal | Rendimiento contra la barra anterior del MISMO día, en porcentaje. Vacío en la primera barra de cada día |
| `target_next_5m` | decimal | Rendimiento en porcentaje de la siguiente barra del mismo día. Vacío en la última barra. Etiqueta para ML |

Muestra: WALMEX

| ticker | fecha_hora | fecha | hora | minuto | precio | ret_5m | target_next_5m |
|---|---|---|---|---|---|---|---|
| WALMEX | 2026-10-02 13:40:00 | 2026-10-02 | 13 | 40 | 45.68 | 0 | 0 |
| WALMEX | 2026-10-02 13:45:00 | 2026-10-02 | 13 | 45 | 45.68 | 0 | 0 |
| WALMEX | 2026-10-02 13:50:00 | 2026-10-02 | 13 | 50 | 45.68 | 0 | 0 |
| WALMEX | 2026-10-02 13:55:00 | 2026-10-02 | 13 | 55 | 45.68 | 0 | 0 |
| WALMEX | 2026-10-02 14:00:00 | 2026-10-02 | 14 | 0 | 45.68 | 0 |  |

## Subcarpeta `diario/`: resumen por día

36 archivos CSV más, un día por fila.

| Columna | Tipo | Descripción |
|---|---|---|
| `ticker` | texto | Emisora |
| `fecha` | fecha | Día de la sesión |
| `n_barras` | entero | Barras de 5 minutos ese día (79 = día completo) |
| `precio_primero` | decimal | Precio de la primera barra |
| `precio_ultimo` | decimal | Precio de la última barra |
| `precio_max` | decimal | Precio máximo entre las barras (puede quedarse corto contra el High diario) |
| `precio_min` | decimal | Precio mínimo entre las barras |
| `ret_intradia` | decimal | Rendimiento de la primera a la última barra, en porcentaje |
| `volatilidad_5m` | decimal | Desviación estándar de ret_5m del día, en porcentaje |
| `dia_completo` | 0/1 | 1 si n_barras = 79 |
| `dia_plano` | 0/1 | 1 si todo el día hubo un solo precio |

Muestra: WALMEX

| ticker | fecha | n_barras | precio_primero | precio_ultimo | precio_max | precio_min | ret_intradia | volatilidad_5m | dia_completo | dia_plano |
|---|---|---|---|---|---|---|---|---|---|---|
| WALMEX | 2026-09-15 | 79 | 45.96 | 45.49 | 46.04 | 45.44 | -1.02263 | 0.147296 | 1 | 0 |
| WALMEX | 2026-09-17 | 79 | 45.49 | 46.27 | 46.28 | 45.49 | 1.71466 | 0.173798 | 1 | 0 |
| WALMEX | 2026-09-18 | 79 | 46.27 | 46.61 | 46.65 | 46.23 | 0.734817 | 0.0919358 | 1 | 0 |
| WALMEX | 2026-10-01 | 79 | 46.18 | 46.57 | 46.73 | 45.82 | 0.844521 | 0.181111 | 1 | 0 |
| WALMEX | 2026-10-02 | 79 | 46.57 | 45.68 | 46.57 | 45.2 | -1.9111 | 0.288626 | 1 | 0 |
