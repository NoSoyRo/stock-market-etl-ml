# Calidad — fundamentales_pit

1368 filas · 32 emisoras · 2016-03-31 a 2026-09-30

| control | resultado | filas | detalle |
|---|---|---|---|
| esquema: columnas del contrato | OK | 0 | faltan=[] sobran=[] |
| esquema: tipos | OK | 0 | columnas con tipo distinto: [] |
| nulos solo donde el contrato los permite | OK | 0 | {} |
| llave única ['ticker', 'quarter_end'] | OK | 0 |  |
| available_date = quarter_end + 70 días | OK | 0 |  |
| quarter_end es fin de trimestre | OK | 0 |  |
| fund_gross_margin en [-5.0, 1.0] | OK | 0 |  |
| fund_oper_margin en [-5.0, 1.0] | FALLA | 1 |  |
| fund_net_margin en [-5.0, 5.0] | OK | 0 |  |
| completitud: >= 32 de 36 emisoras | OK | 0 | hay 32 emisoras |
| trimestres consecutivos (sin huecos dentro de cada emisora) | OK | 0 |  |
| informativo: YoY nulo por no tener el mismo trimestre del año anterior | OK | 101 |  |
| informativo: emisoras con formato no corporativo (bancos/seguros) | OK | 0 | [] |
| informativo: ratios recortados en ±5 (el valor real se perdió) | OK | 66 | la ingesta original aplica clip(-5, 5) |
| informativo: trimestres con cifra reexpresada después | OK | 187 |  |
