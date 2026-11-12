# market / silver

94 archivos CSV, uno por símbolo. Los genera `python -m src.etl.silver.mercado.etl_silver` (PySpark).

| Columna | Tipo | Descripción |
|---|---|---|
| `ticker` | texto | Símbolo de Yahoo Finance |
| `fecha` | fecha | Día de la sesión, YYYY-MM-DD (Date se renombra porque DATE es palabra reservada en SQL) |
| `open` | decimal | Precio de apertura |
| `high` | decimal | Precio máximo del día |
| `low` | decimal | Precio mínimo del día |
| `close` | decimal | Precio de cierre |
| `volume` | entero | Títulos negociados |
| `moneda` | texto | MXN si el símbolo termina en .MX o es MXN=X; USD en otro caso |
| `flag_volumen_cero` | 0/1 | 1 si volume = 0 (normal en MXN=X; en una acción indica que no se negoció) |
| `flag_precio_no_positivo` | 0/1 | 1 si algún precio es <= 0 |
| `flag_ohlc_incoherente` | 0/1 | 1 si high < max(open, close) o low > min(open, close) |
| `flag_salto_25` | 0/1 | 1 si el cierre cambió más de 25 % contra el día anterior |
| `flag_cierre_repetido` | 0/1 | 1 si el cierre es idéntico al del día anterior |

Muestra: WALMEX.MX

| ticker | fecha | open | high | low | close | volume | moneda | flag_volumen_cero | flag_precio_no_positivo | flag_ohlc_incoherente | flag_salto_25 | flag_cierre_repetido |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WALMEX.MX | 2026-11-05 | 52.5 | 53.7 | 52.23 | 53.68 | 13164400 | MXN | 0 | 0 | 0 | 0 | 0 |
| WALMEX.MX | 2026-11-06 | 53.3 | 55.04 | 52.81 | 54.2 | 18893900 | MXN | 0 | 0 | 0 | 0 | 0 |
| WALMEX.MX | 2026-11-09 | 54.24 | 54.74 | 53.52 | 54.44 | 16706444 | MXN | 0 | 0 | 0 | 0 | 0 |
| WALMEX.MX | 2026-11-10 | 54.31 | 55 | 53.89 | 54.45 | 32103398 | MXN | 0 | 0 | 0 | 0 | 0 |
| WALMEX.MX | 2026-11-11 | 54.22 | 55.26 | 53.6 | 53.78 | 21708270 | MXN | 0 | 0 | 0 | 0 | 0 |
