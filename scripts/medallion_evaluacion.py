
"""Pipeline Medallion exclusivo del módulo de Evaluación/Riesgo."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
MEDALLION_DIR = EVAL_DIR / "medallion"
BRONZE_DIR = MEDALLION_DIR / "bronze"

ARCHIVOS_ENTRADA = (
    "backtest_topk.json",
    "portfolio_metrics.json",
    "experiments.json",
)


def calcular_hash(ruta: Path) -> str:
    """Calcula SHA-256 para verificar la integridad de un archivo."""
    sha256 = hashlib.sha256()

    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(65536), b""):
            sha256.update(bloque)

    return sha256.hexdigest()


def ejecutar_bronze() -> dict:
    """Copia los resultados originales sin modificarlos."""
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    manifiesto = {
        "capa": "bronze",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archivos": [],
    }

    for nombre in ARCHIVOS_ENTRADA:
        origen = EVAL_DIR / nombre
        destino = BRONZE_DIR / nombre

        if not origen.is_file():
            raise FileNotFoundError(
                f"No existe el archivo de entrada: {nombre}"
            )

        # Validamos que sea JSON antes de incorporarlo.
        with origen.open("r", encoding="utf-8") as archivo:
            datos = json.load(archivo)

        if not isinstance(datos, dict):
            raise ValueError(
                f"El archivo {nombre} debe contener un objeto JSON"
            )

        # Copia exacta de los bytes originales.
        shutil.copyfile(origen, destino)

        hash_origen = calcular_hash(origen)
        hash_destino = calcular_hash(destino)

        if hash_origen != hash_destino:
            raise ValueError(
                f"Error de integridad al copiar {nombre}"
            )

        manifiesto["archivos"].append({
            "nombre": nombre,
            "origen": f"data/eval/{nombre}",
            "destino": f"data/eval/medallion/bronze/{nombre}",
            "tamano_bytes": destino.stat().st_size,
            "sha256": hash_destino,
            "integridad": "OK",
        })

    ruta_manifiesto = BRONZE_DIR / "manifest.json"
    ruta_manifiesto.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return manifiesto


def ejecutar_silver() -> dict:
    """Valida calidad e integridad de los resultados Bronze."""
    import math

    silver_dir = MEDALLION_DIR / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    errores = []
    advertencias = []
    datos_validados = {}

    manifiesto_path = BRONZE_DIR / "manifest.json"

    if not manifiesto_path.is_file():
        raise FileNotFoundError("No existe el manifiesto Bronze")

    manifiesto = json.loads(
        manifiesto_path.read_text(encoding="utf-8")
    )

    entradas = {
        item["nombre"]: item
        for item in manifiesto["archivos"]
    }

    campos_requeridos = {
        "backtest_topk.json": {
            "strategy": [
                "sharpe_net",
                "max_drawdown",
                "win_rate_net",
                "total_return_net",
                "auc",
            ],
        },
        "portfolio_metrics.json": {
            "portfolio_dual": [
                "sharpe_net",
                "max_drawdown",
                "win_rate_net",
                "total_return_net",
                "auc",
            ],
        },
        "experiments.json": {
            "summary": [
                "news_helps",
                "stacking_beats_auc_weight",
                "positive_sharpe_found",
            ],
        },
    }

    for nombre, secciones in campos_requeridos.items():
        ruta = BRONZE_DIR / nombre

        if nombre not in entradas or not ruta.is_file():
            errores.append(f"Archivo faltante: {nombre}")
            continue

        if calcular_hash(ruta) != entradas[nombre]["sha256"]:
            errores.append(f"Integridad incorrecta: {nombre}")
            continue

        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errores.append(f"JSON inválido en {nombre}: {exc}")
            continue

        if not isinstance(datos, dict):
            errores.append(f"{nombre}: estructura JSON incorrecta")
            continue

        errores_previos = len(errores)

        for seccion, campos in secciones.items():
            contenido = datos.get(seccion)

            if not isinstance(contenido, dict):
                errores.append(
                    f"{nombre}: falta la sección {seccion}"
                )
                continue

            for campo in campos:
                valor = contenido.get(campo)

                if valor is None:
                    errores.append(
                        f"{nombre}.{seccion}.{campo}: faltante"
                    )
                    continue

                if isinstance(valor, float) and not math.isfinite(valor):
                    errores.append(
                        f"{nombre}.{seccion}.{campo}: no finito"
                    )
                    continue

                if nombre == "experiments.json":
                    if not isinstance(valor, bool):
                        errores.append(
                            f"{nombre}.{seccion}.{campo}: "
                            "debe ser booleano"
                        )
                    continue

                if isinstance(valor, bool) or not isinstance(
                    valor, (int, float)
                ):
                    errores.append(
                        f"{nombre}.{seccion}.{campo}: "
                        "debe ser numérico"
                    )
                    continue

                if campo in ("auc", "win_rate_net"):
                    if not 0 <= valor <= 1:
                        errores.append(
                            f"{nombre}.{seccion}.{campo}: "
                            "fuera de rango [0, 1]"
                        )

                if campo == "max_drawdown":
                    if not -1 <= valor <= 0:
                        errores.append(
                            f"{nombre}.{seccion}.{campo}: "
                            "fuera de rango [-1, 0]"
                        )

                if campo == "total_return_net" and valor < -1:
                    errores.append(
                        f"{nombre}.{seccion}.{campo}: "
                        "rendimiento inferior a -100%"
                    )

        if len(errores) == errores_previos:
            datos_validados[nombre] = datos

    # Comprobar consistencia entre las dos fuentes financieras.
    if (
        "backtest_topk.json" in datos_validados
        and "portfolio_metrics.json" in datos_validados
    ):
        estrategia = datos_validados[
            "backtest_topk.json"
        ]["strategy"]

        portafolio = datos_validados[
            "portfolio_metrics.json"
        ]["portfolio_dual"]

        for campo in (
            "sharpe_net",
            "max_drawdown",
            "win_rate_net",
            "total_return_net",
            "auc",
        ):
            if not math.isclose(
                estrategia[campo],
                portafolio[campo],
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                advertencias.append(
                    f"Inconsistencia entre fuentes: {campo}"
                )

    reporte = {
        "capa": "silver",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "estado": "VALIDO" if not errores else "INVALIDO",
        "archivos_validados": list(datos_validados),
        "errores": errores,
        "advertencias": advertencias,
    }

    salida = silver_dir / "quality_report.json"
    salida.write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return reporte


def ejecutar_gold(reporte_silver: dict) -> dict:
    """Consolida métricas financieras verificadas por Silver."""

    gold_dir = MEDALLION_DIR / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    salida = gold_dir / "evaluacion_riesgo.json"

    # Eliminar el reporte anterior para evitar datos desactualizados.
    salida.unlink(missing_ok=True)

    if reporte_silver["estado"] != "VALIDO":
        raise ValueError(
            "Gold no puede generarse: Silver contiene errores"
        )

    requeridos = {
        "backtest_topk.json",
        "portfolio_metrics.json",
        "experiments.json",
    }

    if not requeridos.issubset(
        set(reporte_silver["archivos_validados"])
    ):
        raise ValueError(
            "Gold requiere los tres archivos validados"
        )

    # Verificar nuevamente la integridad de Bronze.
    manifiesto = json.loads(
        (BRONZE_DIR / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    hashes = {
        item["nombre"]: item["sha256"]
        for item in manifiesto["archivos"]
    }

    for nombre in requeridos:
        ruta = BRONZE_DIR / nombre
        if calcular_hash(ruta) != hashes.get(nombre):
            raise ValueError(
                f"Integridad Bronze incorrecta: {nombre}"
            )

    backtest = json.loads(
        (BRONZE_DIR / "backtest_topk.json").read_text(
            encoding="utf-8"
        )
    )

    portafolio = json.loads(
        (BRONZE_DIR / "portfolio_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    experimentos = json.loads(
        (BRONZE_DIR / "experiments.json").read_text(
            encoding="utf-8"
        )
    )

    estrategia = backtest["strategy"]
    benchmark = backtest["benchmark_naftrac"]

    sharpe_neto = estrategia["sharpe_net"]

    if sharpe_neto < 0:
        decision = "NO_BET_LIVE"
    else:
        decision = "REQUIERE_REVISION"

    reporte = {
        "capa": "gold",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "producto": "evaluacion_riesgo",
        "calidad": {
            "estado": reporte_silver["estado"],
            "archivos_validados": len(
                reporte_silver["archivos_validados"]
            ),
            "advertencias": reporte_silver["advertencias"],
        },
        "estrategia": {
            "sharpe_bruto": estrategia["sharpe_gross"],
            "sharpe_neto": sharpe_neto,
            "rendimiento_neto": estrategia["total_return_net"],
            "max_drawdown": estrategia["max_drawdown"],
            "win_rate": estrategia["win_rate_net"],
            "auc": estrategia["auc"],
            "periodos": estrategia["n_periods"],
        },
        "benchmark": {
            "nombre": "NAFTRAC",
            "sharpe": benchmark["sharpe_ratio"],
            "rendimiento": benchmark["total_return"],
            "max_drawdown": benchmark["max_drawdown"],
        },
        "comparacion": {
            "diferencia_sharpe": (
                sharpe_neto - benchmark["sharpe_ratio"]
            ),
            "diferencia_rendimiento": (
                estrategia["total_return_net"]
                - benchmark["total_return"]
            ),
        },
        "experimentos": experimentos["summary"],
        "decision": decision,
        "fuentes": sorted(requeridos),
    }


    salida.write_text(
        json.dumps(
            reporte,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return reporte



if __name__ == "__main__":
        # Invalidar el Gold anterior antes de iniciar otra ejecución.
    gold_anterior = MEDALLION_DIR / "gold" / "evaluacion_riesgo.json"
    gold_anterior.unlink(missing_ok=True)
    bronze = ejecutar_bronze()

    print("\n=== MEDALLION: BRONZE ===")
    for archivo in bronze["archivos"]:
        print(f"{archivo['nombre']}: {archivo['integridad']}")

    silver = ejecutar_silver()

    print("\n=== MEDALLION: SILVER ===")
    print("Estado:", silver["estado"])
    print("Archivos validados:", len(silver["archivos_validados"]))
    print("Errores:", len(silver["errores"]))
    print("Advertencias:", len(silver["advertencias"]))

    for error in silver["errores"]:
        print("ERROR:", error)

    for advertencia in silver["advertencias"]:
        print("ADVERTENCIA:", advertencia)

    if silver["estado"] != "VALIDO":
        print("\nGold omitido por errores de calidad.")
        raise SystemExit(1)

    gold = ejecutar_gold(silver)

    print("\n=== MEDALLION: GOLD ===")
    print("Producto:", gold["producto"])
    print("Decisión:", gold["decision"])
    print("Sharpe neto:", gold["estrategia"]["sharpe_neto"])
    print(
        "Rendimiento neto:",
        gold["estrategia"]["rendimiento_neto"],
    )
    print(
        "Max Drawdown:",
        gold["estrategia"]["max_drawdown"],
    )
    print("\nPipeline Medallion completado.")
