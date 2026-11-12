# intraday / bronze

36 archivos CSV, uno por emisora. Los genera `python -m src.etl.bronze.mercado.etl_bronze` (PySpark).

| Columna | Tipo | Descripción |
|---|---|---|
| `ticker` | texto | Emisora de DataBursátil sin serie: WALMEX, AMXL, FEMSAUBD |
| `datetime` | fecha y hora | Marca de la barra de 5 minutos, tal como la entrega DataBursátil (07:30 a 14:00) |
| `price` | decimal | Último precio negociado en esa marca. No hay open/high/low ni volumen por barra |

Muestra: WALMEX

| ticker | datetime | price |
|---|---|---|
| WALMEX | 2024-11-22 08:35:00 | 51.7 |
| WALMEX | 2024-11-22 08:40:00 | 52.03 |
| WALMEX | 2024-11-22 08:45:00 | 51.86 |
| WALMEX | 2024-11-22 08:50:00 | 51.95 |
| WALMEX | 2024-11-22 08:55:00 | 51.95 |
