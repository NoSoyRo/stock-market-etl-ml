"""Refresca la fuente de bronze: data/market/*.parquet desde Yahoo Finance.

    python productos/precios_fx/actualizar_fuente.py
    python -m src.etl.bronze.mercado.etl_bronze     # después, para regenerar las capas
    python -m src.etl.silver.mercado.etl_silver
    python -m src.etl.gold.mercado.etl_gold

Usa src/data/store.update_all(): baja solo los días nuevos de cada símbolo y
el historial completo de los que no tienen archivo. Modifica los parquet de
data/market/, que están en git.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.store import update_all  # noqa: E402


def main() -> int:
    print("Actualizando data/market desde Yahoo Finance...")
    resultado = update_all(verbose=True)
    print(f"\n{len(resultado)} símbolos procesados. Siguiente paso: el ETL de src/etl (bronze → silver → gold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
