# Recon de datos nuevos (EPS, flujo por casa, fundamentales)

Fecha: 2026-11-09 · Rama `eval/oos-v2-recon` · Solo lectura y perfilado, sin modelos. Todo `[verificado]` con scripts propios sobre `data/`.

## 1. EPS reportados (`data/eps/reported_eps.json`)
- **Cobertura real menor a lo dicho:** 92 claves, pero **27 tienen lista vacía**. Quedan 65 tickers con datos, 37 MX (el handoff decía 92 y 40).
- **MX:** 2,299 anuncios desde 2004; ~133 por año desde 2018 (≈ 4 por ticker y año, consistente).
- **Calidad:** 11 duplicados (ticker, fecha), 49 anuncios en fin de semana, 40 `surprise` nulos (AMZN 9, CEMEXCPO 8, NVDA 5), 20 sorpresas MX con |valor| > 5 (revisar antes de usarlas). 118 de 2,262 huecos entre anuncios MX son > 130 días.
- **`surprise` = 0 exacto en 7% de las filas MX**: probable "sin consenso" disfrazado de cero. Tratar como NaN.
- **PIT (prueba de reacción de precio):** el pico de |retorno| / mediana 60d, por offset desde `announced`, es 1.0 (d−1), 1.19 (d), **1.63 (d+1)**, 1.21 (d+2). Los resultados MX se publican sobre todo **después del cierre**. Regla segura: la información es operable desde el **Close de d+1** (o el Open de d+2), no desde `announced`.

## 2. Flujo por casa (`data/microstructure/hechos/`)
- 533,805 filas, 31 casas, 4,986 códigos, sin nulos, 2025-06-30 → 2026-10-29.
- **Huecos grandes:** faltan **120 de los días hábiles** del rango (≈ 2 por mes hasta abril; desde mayo de 2026 faltan 21, 21, 23, 13, 14 y 11 días por mes). **Desde mayo casi no hay datos**, así que se traslapa muy poco con la ventana OOS (jun–oct).
- **52% de las filas con compra == venta** (cruce interno probable); ninguna con ambos en 0.
- **Los códigos no son los tickers del mercado** (solo 83 de 4,986 coinciden con nombres MX): falta una tabla de mapeo antes de cualquier uso.
- Conclusión: el flujo no sirve para entrenar ni evaluar en OOS hasta resolver huecos y mapeo. Útil solo como estudio exploratorio de ~10 meses densos (jul-2025 → abr-2026).

## 3. Fundamentales (`data/fundamentals.parquet`)
- 1,146 filas, **32 tickers**, quarter_end 2017-03-31 → 2026-09-30. Sin nulos salvo `fund_revenue_yoy` y `fund_earnings_yoy` (11%).
- `available_date` = quarter_end + 70 días **fijos** en todas las filas.
- Contra los anuncios reales de EPS (960 filas pareables): mediana **44 días después** del anuncio real. **97.1% es conservador, 2.9% cae antes del anuncio real** (fuga leve, de unos pocos días en algunos trimestres).
- 1 fila (GAPB, 2026-09-30) tiene `available_date` futura (2026-12-09): todo consumidor debe filtrar `available_date <= fecha`.

## 4. ¿Qué ve hoy el panel?
- `build_panel` (bmv): 35 tickers, 133,730 filas con máscara, 2008-03 → 2026-10-21. **No usa** EPS ni `hechos`. Las features de fundamentales son las 70-días fijos.

## 5. Lo que falta para usarlos honestamente
1. **EPS/PEAD:** limpiar duplicados y fines de semana, tratar `surprise == 0` como NaN, y fijar disponibilidad en el Close de d+1. Se puede hacer ya para los 37 MX.
2. **Fundamentales:** reemplazar +70 fijo por `anuncio real + 1 día hábil` cuando haya anuncio (cubre 960 de 1,146 filas), manteniendo +70 como respaldo.
3. **Flujo por casa:** tabla de mapeo código → ticker y decidir qué hacer con los huecos de mayo-octubre 2026 (reconstruir o declarar el flujo como exploratorio).
