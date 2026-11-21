"""Ejecuta el agente de validacion y guarda sus resultados en JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.strategy_validator import StrategyValidator


def main():
    parser = argparse.ArgumentParser(
        description="Validacion automatica de estrategias de inversion"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/eval/backtest_topk.json",
        help="Archivo JSON con los resultados del backtest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/eval/strategy_validation.json",
        help="Ruta del reporte generado",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.30,
        help="Limite absoluto de drawdown (ejemplo: 0.30)",
    )

    parser.add_argument(
        "--ai",
        action="store_true",
        help="Generar preguntas adicionales utilizando Gemini",
    )
    parser.add_argument(
        "--ai-model",
        default=None,
        help="Modelo de Gemini para validar estrategias",
    )

    args = parser.parse_args()

    try:
        datos = json.loads(args.input.read_text(encoding="utf-8"))

        agente = StrategyValidator(
            max_drawdown_limit=args.max_drawdown
        )

        resultado = agente.act(datos)

        if args.ai:
            from src.agent.strategy_ai import StrategyAI

            ia = StrategyAI(model=args.ai_model)
            resultado["validacion_ia"] = ia.generate_questions(resultado)

        resultado["generated_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        resultado["source"] = str(args.input)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(resultado, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\n=== AGENTE DE VALIDACION DE ESTRATEGIAS ===\n")

    for i, pregunta in enumerate(resultado["preguntas"], 1):
        print(f'{i}. {pregunta["pregunta"]}')
        print(f'   Estado: {pregunta["estado"]}')
        print(f'   Evidencia: {pregunta["evidencia"]}')
        print()

    print("=== RESUMEN ===")
    for estado, cantidad in resultado["resumen"].items():
        print(f"{estado}: {cantidad}")

    if args.ai:
        validacion_ia = resultado["validacion_ia"]

        print("\n=== VALIDACION CON INTELIGENCIA ARTIFICIAL ===")
        print("Estado:", validacion_ia["estado"])
        print("Modelo:", validacion_ia["modelo"])

        for i, pregunta in enumerate(validacion_ia["preguntas"], 1):
            print(f'{i}. {pregunta["pregunta"]}')
            print(f'   Respuesta: {pregunta["respuesta"]}')
            print(f'   Estado: {pregunta["estado"]}')
            print(f'   Evidencia: {pregunta["evidencia"]}')
            print()

        if validacion_ia["estado"] == "error":
            print("Detalle:", validacion_ia.get("detalle"))

    print("\nEstrategia aprobada:", resultado["aprobada"])
    print("Reporte generado:", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
