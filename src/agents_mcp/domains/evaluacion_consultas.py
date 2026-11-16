
"""Consultas de resultados para el agente de Evaluación/Riesgo."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "data" / "eval"

ARCHIVOS = {
    "portafolio": "portfolio_metrics.json",
    "backtest": "backtest_topk.json",
    "experimentos": "experiments.json",
    "estrategias": "strategies_comparison.json",
    "validacion": "strategy_validation.json",
}

def obtener_seccion(datos: dict, clave: str) -> dict:
    """Obtiene una sección del JSON de forma segura."""
    seccion = datos.get(clave)
    return seccion if isinstance(seccion, dict) else {}


def mostrar_valor(datos: dict, clave: str) -> str:
    """Muestra un valor o indica que no está disponible."""
    valor = datos.get(clave)

    if valor is None:
        return "No disponible"

    return str(valor)

def cargar_resultado(tipo: str) -> dict:
    """Lee un JSON de evaluación sin modificarlo."""

    if tipo not in ARCHIVOS:
        return {"error": f"Consulta desconocida: {tipo}"}

    ruta = EVAL_DIR / ARCHIVOS[tipo]

    if not ruta.is_file():
        return {"error": f"Archivo no encontrado: {ruta.name}"}

    try:
        with ruta.open("r", encoding="utf-8") as archivo:
            datos = json.load(archivo)
    except (OSError, json.JSONDecodeError) as error:
        return {"error": f"Error al leer {ruta.name}: {error}"}

    if not isinstance(datos, dict):
        return {"error": f"Formato inválido: {ruta.name}"}

    return datos


def consultar_resultado(tipo: str) -> str:
    """Devuelve las métricas almacenadas en los JSON."""

    datos = cargar_resultado(tipo)

    if "error" in datos:
        return datos["error"]

    if tipo == "portafolio":
        metricas = obtener_seccion(datos, "portfolio_dual")
        benchmark = obtener_seccion(datos, "benchmark_naftrac")

        return (
            "Evaluación del portafolio:\n"
            f"Sharpe neto: {mostrar_valor(metricas, 'sharpe_net')}\n"
            f"Drawdown máximo: {mostrar_valor(metricas, 'max_drawdown')}\n"
            f"Rendimiento neto: {mostrar_valor(metricas, 'total_return_net')}\n"
            f"Sharpe NAFTRAC: {mostrar_valor(benchmark, 'sharpe_ratio')}\n"
            f"Generado: {mostrar_valor(datos, 'generated_at')}"
        )

    if tipo == "backtest":
        estrategia = obtener_seccion(datos, "strategy")
        benchmark = obtener_seccion(datos, "benchmark_naftrac")

        return (
            "Backtest Top-K:\n"
            f"Sharpe neto: {mostrar_valor(estrategia, 'sharpe_net')}\n"
            f"Drawdown máximo: {mostrar_valor(estrategia, 'max_drawdown')}\n"
            f"Rendimiento neto: {mostrar_valor(estrategia, 'total_return_net')}\n"
            f"Rendimiento NAFTRAC: {mostrar_valor(benchmark, 'total_return')}\n"
            f"Periodos: {mostrar_valor(estrategia, 'n_periods')}"
        )

    if tipo == "experimentos":
        resumen = obtener_seccion(datos, "summary")

        return (
            "Experimentos:\n"
            f"Noticias ayudan: {mostrar_valor(resumen, 'news_helps')}\n"
            f"Stacking supera AUC ponderado: "
            f"{mostrar_valor(resumen, 'stacking_beats_auc_weight')}\n"
            f"Sharpe positivo encontrado: "
            f"{mostrar_valor(resumen, 'positive_sharpe_found')}\n"
            f"Advertencia de pocas observaciones: "
            f"{mostrar_valor(resumen, 'low_n_warning')}"
        )

    if tipo == "estrategias":
        return (
            "Comparación de estrategias:\n"
            f"Total: {mostrar_valor(datos, 'total_estrategias')}\n"
            f"Aprobadas: {mostrar_valor(datos, 'aprobadas')}\n"
            f"Generado: {mostrar_valor(datos, 'generated_at')}"
        )

    if tipo == "validacion":
        ia = obtener_seccion(datos, "validacion_ia")

        return (
            "Validación de estrategia:\n"
            f"Estrategia: {mostrar_valor(datos, 'estrategia')}\n"
            f"Aprobada: {mostrar_valor(datos, 'aprobada')}\n"
            f"Resumen: {mostrar_valor(datos, 'resumen')}\n"
            f"Estado IA: {mostrar_valor(ia, 'estado')}"
        )

    return "Consulta no disponible."

    return "Consulta no disponible."
