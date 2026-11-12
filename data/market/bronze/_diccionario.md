# market / bronze

94 archivos CSV, uno por símbolo. Los genera `python -m src.etl.bronze.mercado.etl_bronze` (PySpark).

| Columna | Tipo | Descripción |
|---|---|---|
| `ticker` | texto | Símbolo de Yahoo Finance: WALMEX.MX (BMV, pesos), AAPL (USA, dólares), MXN=X (tipo de cambio) |
| `Date` | fecha | Día de la sesión, YYYY-MM-DD |
| `Open` | decimal | Precio de apertura |
| `High` | decimal | Precio máximo del día |
| `Low` | decimal | Precio mínimo del día |
| `Close` | decimal | Precio de cierre |
| `Volume` | entero | Títulos negociados. En MXN=X siempre es 0: el tipo de cambio no se negocia en una bolsa |

Muestra: WALMEX.MX

| ticker | Date | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|---|
| WALMEX.MX | 2003-01-01 | 5.7245 | 5.94657 | 5.7245 | 5.87995 | 0 |
| WALMEX.MX | 2003-01-02 | 5.7245 | 5.94657 | 5.7245 | 5.87995 | 8334330 |
| WALMEX.MX | 2003-01-03 | 5.87255 | 5.90956 | 5.85034 | 5.87255 | 18475096 |
| WALMEX.MX | 2003-01-06 | 5.87255 | 6.11683 | 5.87255 | 6.10449 | 16180223 |
| WALMEX.MX | 2003-01-07 | 6.10449 | 6.10449 | 5.95151 | 5.96384 | 15261074 |
