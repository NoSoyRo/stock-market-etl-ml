# ETL con PySpark por capas

Cada equipo pone su código en una subcarpeta con su nombre dentro de cada capa:

```
src/etl/
├── bronze/<equipo>/   fuente cruda → campos definidos
├── silver/<equipo>/   limpieza, nombres y banderas de calidad
├── gold/<equipo>/     datos listos para el agente y ML
└── shared/            piezas comunes
```

## Equipo Mercado (Precios & FX)

| Capa   | Script                         | Lee                                                | Escribe                                                        |
|--------|--------------------------------|----------------------------------------------------|----------------------------------------------------------------|
| bronze | `bronze/mercado/etl_bronze.py` | `data/market/*.parquet`, `data/intraday/*.parquet` | `data/{market,intraday}/bronze/`                               |
| silver | `silver/mercado/etl_silver.py` | bronze                                             | `data/{market,intraday}/silver/`                               |
| gold   | `gold/mercado/etl_gold.py`     | silver                                             | `data/{market,intraday}/gold/` y `data/market/gold/mercado.db` |

Se corren en orden desde la raíz del repo:

```bash
python -m src.etl.bronze.mercado.etl_bronze
python -m src.etl.silver.mercado.etl_silver
python -m src.etl.gold.mercado.etl_gold
```

Necesitan 21 y `pyspark`. Los esquemas de cada capa están en `shared/precios_fx.py` (`ESQUEMA`) y en el
`_diccionario.md` de cada carpeta de datos. Detalle del producto: `productos/precios_fx/README.md`.

Los demás archivos de esta carpeta (`alsea/`, `walmex/`, `cargar_precios.py`, `shared/extractors`, `shared/loaders`...)
son el ETL anterior hacia SQL Server.
