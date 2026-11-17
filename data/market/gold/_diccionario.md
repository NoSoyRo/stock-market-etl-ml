# market / gold

94 archivos CSV, uno por símbolo. Los genera `python -m src.etl.gold.mercado.etl_gold` (PySpark).

| Columna | Tipo | Descripción |
|---|---|---|
| `ticker` | texto | Símbolo de Yahoo Finance |
| `fecha` | fecha | Día de la sesión, YYYY-MM-DD |
| `open` | decimal | Precio de apertura, en la moneda del símbolo |
| `high` | decimal | Precio máximo del día |
| `low` | decimal | Precio mínimo del día |
| `close` | decimal | Precio de cierre |
| `volume` | entero | Títulos negociados |
| `moneda` | texto | MXN o USD |
| `usdmxn_close` | decimal | Pesos por dólar al cierre de ese día (MXN=X). Si ese día no hubo dato, el último de hasta 5 días antes |
| `close_usd` | decimal | Cierre en dólares: close / usdmxn_close si moneda = MXN; close si moneda = USD. Vacío para MXN=X |
| `ret_1d` | decimal | Rendimiento del día contra el cierre anterior, en porcentaje (1.2 = +1.2 %). Vacío en la primera fila |
| `target_next_day` | decimal | Rendimiento en porcentaje de la siguiente sesión (el ret_1d de mañana). Vacío en la última fila. Es la etiqueta para ML: nunca usarla como variable de entrada |
| `es_valido` | 0/1 | 1 si la fila no tiene precio <= 0, ni OHLC incoherente, ni volumen 0 (MXN=X queda exento del volumen) |

Muestra: WALMEX.MX

| ticker | fecha | open | high | low | close | volume | moneda | usdmxn_close | close_usd | ret_1d | target_next_day | es_valido |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WALMEX.MX | 2026-11-05 | 52.5 | 53.7 | 52.23 | 53.68 | 13164400 | MXN | 18.6077 | 2.88483 | 2.79586 | 0.968704 | 1 |
| WALMEX.MX | 2026-11-06 | 53.3 | 55.04 | 52.81 | 54.2 | 18893900 | MXN | 18.7809 | 2.88591 | 0.968704 | 0.4428 | 1 |
| WALMEX.MX | 2026-11-09 | 54.24 | 54.74 | 53.52 | 54.44 | 16706444 | MXN | 18.6374 | 2.921 | 0.4428 | 0.0183728 | 1 |
| WALMEX.MX | 2026-11-10 | 54.31 | 55 | 53.89 | 54.45 | 32103398 | MXN | 18.5334 | 2.93794 | 0.0183728 | -1.23049 | 1 |
| WALMEX.MX | 2026-11-11 | 54.22 | 55.26 | 53.6 | 53.78 | 21708270 | MXN | 18.5992 | 2.89152 | -1.23049 |  | 1 |
