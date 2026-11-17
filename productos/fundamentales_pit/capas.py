"""Bronze -> silver -> golden de Fundamentales PIT.

    python -m productos.fundamentales_pit.capas bronze      # necesita DATABURSATIL_API_TOKEN
    python -m productos.fundamentales_pit.capas silver
    python -m productos.fundamentales_pit.capas golden
    python -m productos.fundamentales_pit.capas todo        # silver + golden + calidad
    python -m productos.fundamentales_pit.capas todo --bronze  # además baja bronze primero

BRONZE  data/fundamentals/bronze/<ticker>/<financieros>/<periodo>.json
        La respuesta de la API tal cual, con la fecha de descarga. Nunca el token.
        Sirve para rehacer silver sin volver a llamar a la API.
SILVER  data/fundamentals/silver/fundamentales_trimestre.parquet
        Un trimestre por fila, con MONTOS (no solo ratios) y de dónde salió.
GOLDEN  data/fundamentals/golden/fundamentales_pit.parquet
        Lo que consumen dashboard, agente y modelos. Columnas fund_* compatibles
        con src/ml/features.py y build_dashboard.py.

El código de cada capa vive en src/etl, en la carpeta de su capa:

    src/etl/bronze/fundamentales_pit/capa.py
    src/etl/silver/fundamentales_pit/capa.py
    src/etl/golden/fundamentales_pit/capa.py

Este archivo solo las corre en orden (main) y reexporta sus funciones para que
los comandos, los tests y los demás módulos sigan funcionando igual.


Dos correcciones respecto a src/api/fundamentals.py (que no se modifica):

1. YoY por FECHA, no por posición. pct_change(4) compara contra "4 filas
   atrás"; si falta un trimestre, compara contra el trimestre equivocado.
   En el parquet del repo eso pasa en los 28 registros de 2026T1.
2. Cifra original antes que comparativo. La ingesta original pide solo años
   pares y saca los impares del comparativo del reporte del año siguiente,
   que se publica un año después y puede venir reexpresado. Aquí, si existe
   el reporte propio del trimestre, gana; el comparativo solo rellena huecos y
   queda marcado (origen = "comparativo").
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from productos.fundamentales_pit.pit import DIVIDENDOS_GOLDEN, GOLDEN
from src.etl.bronze.fundamentales_pit.capa import (  # noqa: F401
    API,
    BRONZE,
    VENTANA_REINTENTO_DIAS,
    _es_error_de_red,
    _fin_de_trimestre,
    _mapa_emisoras,
    _reintentar,
    bajar_bronze,
    periodos,
)
from src.etl.golden.fundamentales_pit.capa import (  # noqa: F401
    DIVIDENDOS_COLS,
    DIVIDENDOS_MANUAL,
    GOLDEN_COLS,
    VERSION_PRODUCTO,
    _lag_dias,
    _yoy_legacy_valido,
    _yoy_por_fecha,
    construir_dividendos,
    construir_golden,
)
from src.etl.silver.fundamentales_pit.capa import (  # noqa: F401
    CAMPOS,
    FUENTE,
    LEGACY,
    SILVER,
    SILVER_COLS,
    _elegir,
    _fin_de_periodo,
    _monto,
    _periodo_de,
    filas_de_bronze,
    silver_desde_bronze,
    silver_desde_legacy,
)

logger = logging.getLogger(__name__)


def _escribir(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paso", choices=["bronze", "silver", "golden", "calidad", "todo"])
    ap.add_argument("--bronze", action="store_true", help="con 'todo': bajar bronze primero")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--desde", type=int, default=2017)
    ap.add_argument("--no-estricto", action="store_true",
                    help="pedir solo años pares como la ingesta original (no PIT-estricto)")
    ap.add_argument("--forzar", action="store_true", help="volver a bajar aunque exista el JSON")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if a.paso == "bronze" or (a.paso == "todo" and a.bronze):
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ.get("DATABURSATIL_API_TOKEN")
        if not token:
            raise SystemExit("Falta DATABURSATIL_API_TOKEN en .env")
        r = bajar_bronze(token, a.tickers, a.desde, estricto=not a.no_estricto, forzar=a.forzar)
        print(f"[bronze] llamadas={r['llamadas']} ya_existian={r['omitidas_ya_existian']} "
              f"errores={r['errores']}")
        if a.paso == "bronze":
            return

    if a.paso in ("silver", "todo"):
        hay_bronze = any(BRONZE.glob("*/*/*.json"))
        silver = silver_desde_bronze() if hay_bronze else silver_desde_legacy()
        if not hay_bronze:
            print("[silver] sin bronze: se usa data/fundamentals.parquet (solo ratios)")
        print(f"[silver] {len(silver)} filas -> {_escribir(silver, SILVER)}")
    if a.paso in ("golden", "todo"):
        silver = pd.read_parquet(SILVER)
        golden = construir_golden(silver)
        print(f"[golden] {len(golden)} filas -> {_escribir(golden, GOLDEN)}")
        div = construir_dividendos()
        print(f"[golden] dividendos {len(div)} filas -> {_escribir(div, DIVIDENDOS_GOLDEN)}")
    if a.paso in ("calidad", "todo"):
        from productos.fundamentales_pit.calidad import escribir_reporte
        print(f"[calidad] {escribir_reporte()}")


if __name__ == "__main__":
    main()
