"""Demo de la herramienta consultar_mercado_sql: lo que haría Claude, paso a paso.

    python productos/precios_fx/demo_herramienta_sql.py

Necesita la base gold: python -m src.etl.gold.mercado.etl_gold (después de bronze y silver)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent.tools import MERCADO_DB, SEMANTIC_LAYER, AgenteMercadoSQL  # noqa: E402


def main() -> int:
    print("1. Capa semántica:", SEMANTIC_LAYER.relative_to(ROOT), "✅" if SEMANTIC_LAYER.exists() else "❌ no existe")
    print("2. Base gold:     ", MERCADO_DB.relative_to(ROOT),
          "✅" if MERCADO_DB.exists() else "❌ corre el ETL de src/etl")
    agente = AgenteMercadoSQL()

    schema = agente.tool_schema()
    print(f"\n3. Lo que Claude lee de la herramienta '{schema['name']}':")
    print("   " + schema["description"][:200].replace("\n", "\n   ") + "...")

    preguntas = [
        "SELECT fecha, close, close_usd, ret_1d FROM precios_historicos "
        "WHERE ticker = 'WALMEX.MX' ORDER BY fecha DESC LIMIT 3",
        "DROP TABLE precios_historicos",
    ]
    for i, sql in enumerate(preguntas, start=4):
        print(f"\n{i}. Claude manda: {sql}")
        r = agente.observe(sql_query=sql)
        if r["ok"]:
            for fila in r["datos"]:
                print("   ", fila)
        else:
            print(f"   bloqueado ({r['error_tipo']}): {r['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
