"""Herramientas MCP del producto Gold de evaluacion."""

from claude_agent_sdk import tool

from src.agents_mcp.domains import evaluacion_semantica as sem


def ejecutar_consulta(funcion, *argumentos):

    try:

        resultado = funcion(*argumentos)

        return {
            "content": [
                {
                    "type": "text",
                    "text": sem.serializar(resultado)
                }
            ]
        }

    except (OSError, ValueError, KeyError, TypeError) as error:

        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": (
                        "No fue posible consultar el producto Gold: "
                        + str(error)
                    )
                }
            ]
        }


@tool(
    "resumen_estrategias",
    "Consulta el resumen Gold de las estrategias evaluadas, "
    "las aprobaciones, los mejores resultados y la decision global.",
    {}
)
async def resumen_estrategias(args):

    return ejecutar_consulta(
        sem.resumen_estrategias
    )


@tool(
    "consultar_estrategia",
    "Consulta las metricas financieras netas y brutas, "
    "Sharpe, drawdown, capital final, operaciones, configuracion, "
    "comparacion con benchmark y riesgos de una estrategia. "
    "Nombres disponibles mediante resumen_estrategias.",
    {"nombre": str}
)
async def consultar_estrategia(args):

    return ejecutar_consulta(
        sem.consultar_estrategia,
        args.get("nombre", "")
    )


@tool(
    "comparar_estrategias",
    "Compara las estrategias mediante un ranking. "
    "Criterios disponibles: rendimiento_neto, sharpe_neto "
    "y cumplimiento. Incluye el benchmark de referencia.",
    {"criterio": str}
)
async def comparar_estrategias(args):

    return ejecutar_consulta(
        sem.comparar_estrategias,
        args.get("criterio", "rendimiento_neto")
    )


@tool(
    "consultar_benchmark",
    "Consulta las metricas del benchmark NAFTRAC "
    "Buy and Hold, incluyendo rendimiento, Sharpe, "
    "drawdown, capital final y fuentes.",
    {}
)
async def consultar_benchmark(args):

    return ejecutar_consulta(
        sem.consultar_benchmark
    )


@tool(
    "consultar_validacion",
    "Consulta los criterios cumplidos, incumplidos "
    "y no aplicables de una estrategia, su aprobacion "
    "y sus motivos de rechazo.",
    {"nombre": str}
)
async def consultar_validacion(args):

    return ejecutar_consulta(
        sem.consultar_validacion,
        args.get("nombre", "")
    )


@tool(
    "consultar_calidad",
    "Consulta la calidad de los datos, trazabilidad "
    "Medallion, fuentes financieras, advertencias "
    "y limitaciones metodologicas.",
    {}
)
async def consultar_calidad(args):

    return ejecutar_consulta(
        sem.consultar_calidad
    )


@tool(
    "consultar_decision",
    "Consulta la decision global de evaluacion, "
    "las estrategias aprobadas o rechazadas "
    "y los motivos de cada decision. "
    "No representa una orden de inversion real.",
    {}
)
async def consultar_decision(args):

    return ejecutar_consulta(
        sem.consultar_decision
    )


TOOLS_ESTRATEGIAS = [
    resumen_estrategias,
    consultar_estrategia,
    comparar_estrategias,
    consultar_benchmark,
    consultar_validacion,
    consultar_calidad,
    consultar_decision
]
