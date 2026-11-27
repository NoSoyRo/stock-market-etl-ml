"""Pipeline Medallion para multiples estrategias financieras."""

from __future__ import annotations

import json
import math
import shutil

from datetime import datetime, timezone
from pathlib import Path

from scripts.medallion_evaluacion import calcular_hash


ROOT = Path(__file__).resolve().parents[1]

MEDALLION_DIR = ROOT / "data/eval/medallion"

BRONZE_DIR = MEDALLION_DIR / "bronze"
SILVER_DIR = MEDALLION_DIR / "silver"
GOLD_DIR = MEDALLION_DIR / "gold"


ESTRATEGIAS = {
    "topk_2": "topk_2.json",
    "topk_5": "topk_5.json",
    "topk_10": "topk_10.json",
    "topk_5_ranking": "topk_5_ranking.json",
    "sma_50_200": "sma_portfolio.json",
    "rsi_14": "rsi_portfolio.json",
}


ENTRADAS = {
    f"original_{nombre}.json":
        ROOT / "data/estrategias_corregidas" / archivo
    for nombre, archivo in ESTRATEGIAS.items()
}

ENTRADAS.update({
    "buy_hold_naftrac.json":
        ROOT / "data/estrategias_homologadas/buy_hold_naftrac.json",

    "strategies_catalog.json":
        ROOT / "data/estrategias_homologadas/strategies_catalog.json",

    "strategies_comparison_real.json":
        ROOT / "data/eval/strategies_comparison_real.json",

    "universo_tradicional.json":
        ROOT / "data/estrategias_corregidas/universo_tradicional.json",
})


def ahora():
    return datetime.now(timezone.utc).isoformat()


def leer_json(path):

    datos = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(datos, dict):
        raise ValueError(
            f"Se esperaba un objeto JSON: {path}"
        )

    return datos


def guardar_json(path, datos):

    path.parent.mkdir(parents=True, exist_ok=True)

    temporal = path.with_suffix(".tmp")

    temporal.write_text(
        json.dumps(
            datos,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    temporal.replace(path)


def ejecutar_bronze():

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    manifiesto_path = (
        BRONZE_DIR / "manifest_estrategias.json"
    )

    manifiesto_path.unlink(missing_ok=True)

    # Verificar todas las entradas antes de copiarlas.
    for nombre, origen in ENTRADAS.items():

        if not origen.is_file():
            raise FileNotFoundError(
                f"Entrada faltante: {origen}"
            )

        leer_json(origen)

    manifiesto = {
        "capa": "bronze",
        "producto": "evaluacion_multiples_estrategias",
        "generated_at": ahora(),
        "archivos": []
    }

    for nombre, origen in ENTRADAS.items():

        destino = BRONZE_DIR / nombre

        hash_original = calcular_hash(origen)

        shutil.copyfile(origen, destino)

        hash_copia = calcular_hash(destino)

        if hash_original != hash_copia:
            raise ValueError(
                f"Integridad incorrecta: {nombre}"
            )

        manifiesto["archivos"].append({
            "nombre": nombre,
            "origen": str(origen.relative_to(ROOT)),
            "destino": str(destino.relative_to(ROOT)),
            "tamano_bytes": destino.stat().st_size,
            "sha256": hash_copia,
            "integridad": "OK"
        })

    manifiesto["total_archivos"] = len(
        manifiesto["archivos"]
    )

    guardar_json(manifiesto_path, manifiesto)

    return manifiesto


def cargar_bronze():

    manifiesto = leer_json(
        BRONZE_DIR / "manifest_estrategias.json"
    )

    entradas = {
        item["nombre"]: item
        for item in manifiesto["archivos"]
    }

    if (
        set(entradas) != set(ENTRADAS)
        or len(manifiesto["archivos"]) != len(ENTRADAS)
    ):
        raise ValueError(
            "Manifiesto Bronze incompleto o incorrecto"
        )

    datos = {}

    for nombre, item in entradas.items():

        path = BRONZE_DIR / nombre

        if not path.is_file():
            raise FileNotFoundError(
                f"Archivo Bronze faltante: {nombre}"
            )

        if calcular_hash(path) != item["sha256"]:
            raise ValueError(
                f"Integridad Bronze incorrecta: {nombre}"
            )

        datos[nombre] = leer_json(path)

    return datos



def exigir(condicion, mensaje):

    if not condicion:
        raise ValueError(mensaje)


def numero(valor):

    return (
        isinstance(valor, (int, float))
        and not isinstance(valor, bool)
        and math.isfinite(valor)
    )


def iguales(a, b):

    return (
        numero(a)
        and numero(b)
        and math.isclose(
            a, b,
            rel_tol=1e-8,
            abs_tol=1e-8
        )
    )


def validar_finanzas(nombre, datos):

    from datetime import date

    metricas = datos["strategy"]
    config = datos["config"]
    periodo = datos["evaluation_period"]
    fechas = datos["dates"]

    capital = config["capital"]
    sesiones = periodo["trading_sessions"]

    exigir(
        numero(capital) and capital > 0,
        f"{nombre}: capital invalido"
    )

    exigir(
        type(sesiones) is int and sesiones > 0,
        f"{nombre}: sesiones invalidas"
    )

    exigir(
        len(fechas) == sesiones
        and fechas == sorted(set(fechas)),
        f"{nombre}: calendario incorrecto"
    )

    for fecha in fechas:
        date.fromisoformat(fecha)

    exigir(
        fechas[0] == periodo["start"]
        and fechas[-1] == periodo["end"],
        f"{nombre}: fechas inconsistentes"
    )

    exigir(
        periodo["periods_per_year"] == 252
        and periodo["metric_frequency"] == "daily",
        f"{nombre}: frecuencia incorrecta"
    )

    exigir(
        metricas["n_periods"] == sesiones,
        f"{nombre}: numero de periodos incorrecto"
    )

    for campo in (
        "sharpe_gross",
        "sharpe_net",
        "max_drawdown",
        "calmar_ratio",
        "win_rate_net",
        "total_return_gross",
        "total_return_net",
        "final_equity_gross",
        "final_equity_net"
    ):

        exigir(
            numero(metricas[campo]),
            f"{nombre}: metrica invalida: {campo}"
        )

    exigir(
        -1 <= metricas["max_drawdown"] <= 0,
        f"{nombre}: drawdown fuera de rango"
    )

    exigir(
        0 <= metricas["win_rate_net"] <= 1,
        f"{nombre}: win rate fuera de rango"
    )

    auc = metricas["auc"]

    exigir(
        auc is None
        or (numero(auc) and 0 <= auc <= 1),
        f"{nombre}: AUC fuera de rango"
    )

    for campo in ("commission_pct", "slippage_pct"):

        exigir(
            numero(config[campo])
            and config[campo] >= 0,
            f"{nombre}: costo invalido: {campo}"
        )

    exigir(
        iguales(
            config["round_trip_cost"],
            2 * (
                config["commission_pct"]
                + config["slippage_pct"]
            )
        ),
        f"{nombre}: costo total inconsistente"
    )

    for tipo in ("gross", "net"):

        curva = datos["curves"][tipo]
        retornos = datos["daily_returns"][tipo]

        exigir(
            len(curva) == sesiones + 1,
            f"{nombre}: longitud de curva {tipo}"
        )

        exigir(
            len(retornos) == sesiones,
            f"{nombre}: longitud de retornos {tipo}"
        )

        exigir(
            all(numero(v) and v > 0 for v in curva),
            f"{nombre}: curva {tipo} invalida"
        )

        exigir(
            iguales(curva[0], capital),
            f"{nombre}: capital inicial inconsistente"
        )

        exigir(
            iguales(
                curva[-1],
                metricas[f"final_equity_{tipo}"]
            ),
            f"{nombre}: capital final inconsistente"
        )

        exigir(
            iguales(
                curva[-1] / capital - 1,
                metricas[f"total_return_{tipo}"]
            ),
            f"{nombre}: retorno acumulado inconsistente"
        )

        for i, retorno in enumerate(retornos):

            esperado = curva[i + 1] / curva[i] - 1

            exigir(
                iguales(retorno, esperado),
                f"{nombre}: retorno diario incorrecto: {i}"
            )

    exigir(
        metricas["final_equity_net"]
        <= metricas["final_equity_gross"] + 1e-6,
        f"{nombre}: capital neto superior al bruto"
    )


def validar_estrategia(nombre, datos, original, benchmark):

    validar_finanzas(nombre, datos)

    metricas = datos["strategy"]
    referencia = benchmark["strategy"]
    periodo = datos["evaluation_period"]
    contexto = datos["evaluation_context"]

    exigir(
        datos["strategy_name"] == nombre,
        f"{nombre}: identificador incorrecto"
    )

    tipo = datos["strategy_type"]

    exigir(
        tipo in ("machine_learning", "rule_based"),
        f"{nombre}: tipo desconocido"
    )

    es_ml = tipo == "machine_learning"

    exigir(
        contexto["auc_applicable"] is es_ml
        and contexto["temporal_split_applicable"] is es_ml,
        f"{nombre}: aplicabilidad incorrecta"
    )

    if not es_ml:

        exigir(
            metricas["auc"] is None,
            f"{nombre}: AUC no aplicable"
        )

    exigir(
        datos["dates"] == benchmark["dates"],
        f"{nombre}: calendario diferente al benchmark"
    )

    for campo in (
        "capital",
        "commission_pct",
        "slippage_pct",
        "round_trip_cost"
    ):

        exigir(
            iguales(
                datos["config"][campo],
                benchmark["config"][campo]
            ),
            f"{nombre}: configuracion diferente: {campo}"
        )

    for campo in (
        "total_return_gross",
        "total_return_net",
        "final_equity_gross",
        "final_equity_net"
    ):

        actual = metricas[campo]
        anterior = original["strategy"][campo]

        if campo in (
            "final_equity_gross",
            "final_equity_net"
        ):

            consistente = (
                numero(actual)
                and numero(anterior)
                and math.isclose(
                    actual,
                    anterior,
                    rel_tol=0.0,
                    abs_tol=0.005000001
                )
            )

            tipo_curva = (
                "gross"
                if campo == "final_equity_gross"
                else "net"
            )

            exigir(
                iguales(
                    datos["curves"][tipo_curva][-1],
                    original["curves"][tipo_curva][-1]
                ),
                f"{nombre}: capital final de curva alterado"
            )

        else:

            consistente = iguales(actual, anterior)

        exigir(
            consistente,
            f"{nombre}: resultado original alterado: {campo}"
        )

    exigir(
        metricas["auc"] == original["strategy"]["auc"],
        f"{nombre}: AUC original alterado"
    )

    exigir(
        datos["operations"] == original["operations"],
        f"{nombre}: operaciones originales alteradas"
    )

    exigir(
        datos["config"] == original["config"],
        f"{nombre}: configuracion original alterada"
    )

    exigir(
        datos["benchmark_naftrac"]["symbol"]
        == benchmark["config"]["symbol"],
        f"{nombre}: simbolo benchmark incorrecto"
    )

    exigir(
        datos["benchmark_naftrac"]["cost_basis"] == "net",
        f"{nombre}: benchmark no utiliza costos netos"
    )

    for campo in (
        "total_return_net",
        "total_return_gross",
        "sharpe_net",
        "sharpe_gross",
        "max_drawdown",
        "final_equity_net",
        "final_equity_gross",
        "n_periods"
    ):

        exigir(
            iguales(
                datos["benchmark_naftrac"][campo],
                referencia[campo]
            ),
            f"{nombre}: benchmark inconsistente: {campo}"
        )

    exigir(
        datos["curves"]["naftrac"]
        == benchmark["curves"]["net"],
        f"{nombre}: curva benchmark incorrecta"
    )

    exigir(
        datos["curves"]["naftrac_gross"]
        == benchmark["curves"]["gross"],
        f"{nombre}: curva bruta benchmark incorrecta"
    )

    exigir(
        datos["daily_returns"]["naftrac"]
        == benchmark["daily_returns"]["net"],
        f"{nombre}: retornos benchmark incorrectos"
    )

    for campo, metrica in (
        ("excess_return_net", "total_return_net"),
        ("sharpe_delta_net", "sharpe_net"),
        ("capital_difference_net", "final_equity_net")
    ):

        esperado = metricas[metrica] - referencia[metrica]

        exigir(
            iguales(datos["vs_benchmark"][campo], esperado),
            f"{nombre}: comparacion incorrecta: {campo}"
        )

    sin_operaciones = (
        periodo["closed_trades"] == 0
        and periodo["open_positions"] == 0
    )

    exigir(
        contexto["no_trades"] is sin_operaciones,
        f"{nombre}: contexto operativo incorrecto"
    )

    exigir(
        type(contexto["exploratory"]) is bool,
        f"{nombre}: contexto exploratorio incorrecto"
    )

    exigir(
        datos["homologation"]["benchmark_pending"] is False
        and datos["homologation"]["common_calendar_verified"] is True
        and datos["homologation"]["common_costs_verified"] is True,
        f"{nombre}: homologacion incompleta"
    )


def ejecutar_silver():

    from src.agent.strategy_comparator import StrategyComparator

    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    catalogo_path = SILVER_DIR / "strategies_catalog.json"
    comparacion_path = SILVER_DIR / "strategies_comparison.json"
    benchmark_path = SILVER_DIR / "buy_hold_naftrac.json"

    for path in (
        catalogo_path,
        comparacion_path,
        benchmark_path,
        SILVER_DIR / "quality_report_estrategias.json",
        GOLD_DIR / "evaluacion_multiples_estrategias.json"
    ):
        path.unlink(missing_ok=True)

    reporte = {
        "capa": "silver",
        "producto": "evaluacion_multiples_estrategias",
        "generated_at": ahora(),
        "estado": "INVALIDO",
        "estrategias_validadas": [],
        "errores": [],
        "advertencias": []
    }

    try:

        entradas = cargar_bronze()

        catalogo = entradas["strategies_catalog.json"]
        benchmark = entradas["buy_hold_naftrac.json"]
        comparacion = entradas["strategies_comparison_real.json"]

        exigir(
            set(catalogo) == set(ESTRATEGIAS),
            "Catalogo de estrategias incompleto"
        )

        validar_finanzas("NAFTRAC", benchmark)

        exigir(
            benchmark["strategy"]["auc"] is None,
            "NAFTRAC no debe tener AUC"
        )

        exigir(
            benchmark["config"]["symbol"] == "NAFTRACISHRS.MX",
            "Benchmark independiente incorrecto"
        )

        for nombre, datos in catalogo.items():

            try:

                validar_estrategia(
                    nombre,
                    datos,
                    entradas[f"original_{nombre}.json"],
                    benchmark
                )

                reporte["estrategias_validadas"].append(nombre)

                contexto = datos["evaluation_context"]
                periodo = datos["evaluation_period"]

                if contexto["no_trades"]:
                    reporte["advertencias"].append(
                        f"{nombre}: sin operaciones"
                    )

                if contexto["exploratory"]:
                    reporte["advertencias"].append(
                        f"{nombre}: resultado exploratorio"
                    )

                if periodo["open_positions"] > 0:
                    reporte["advertencias"].append(
                        f"{nombre}: posiciones abiertas al cierre"
                    )

                if periodo.get("stale_valuation_asset_days", 0) > 0:
                    reporte["advertencias"].append(
                        f"{nombre}: valoraciones con ultimo precio disponible"
                    )

            except (ValueError, TypeError, KeyError) as exc:

                reporte["errores"].append(str(exc))

        recalculada = StrategyComparator(
            max_drawdown_limit=0.30
        ).evaluate(catalogo)

        for campo in (
            "total_estrategias",
            "aprobadas",
            "evaluaciones",
            "comparacion"
        ):

            exigir(
                comparacion[campo] == recalculada[campo],
                f"Reporte del comparador inconsistente: {campo}"
            )

        if not reporte["errores"]:

            guardar_json(catalogo_path, catalogo)
            guardar_json(comparacion_path, comparacion)
            guardar_json(benchmark_path, benchmark)

            reporte["estado"] = "VALIDO"

            reporte["total_estrategias"] = len(catalogo)
            reporte["benchmark_validado"] = True

            reporte["manifest_sha256"] = calcular_hash(
                BRONZE_DIR / "manifest_estrategias.json"
            )

            reporte["archivos_silver"] = {
                path.name: calcular_hash(path)
                for path in (
                    catalogo_path,
                    comparacion_path,
                    benchmark_path
                )
            }

    except (OSError, ValueError, TypeError, KeyError) as exc:

        reporte["errores"].append(str(exc))

    guardar_json(
        SILVER_DIR / "quality_report_estrategias.json",
        reporte
    )

    return reporte



def cargar_silver():

    reporte = leer_json(
        SILVER_DIR / "quality_report_estrategias.json"
    )

    exigir(
        reporte["estado"] == "VALIDO",
        "Silver no contiene resultados validos"
    )

    exigir(
        set(reporte["estrategias_validadas"])
        == set(ESTRATEGIAS),
        "Silver no contiene todas las estrategias"
    )

    exigir(
        reporte["benchmark_validado"] is True,
        "Benchmark no validado"
    )

    exigir(
        not reporte["errores"],
        "Silver contiene errores"
    )

    exigir(
        calcular_hash(
            BRONZE_DIR / "manifest_estrategias.json"
        ) == reporte["manifest_sha256"],
        "El manifiesto Bronze ha cambiado"
    )

    bronze = cargar_bronze()

    requeridos = {
        "strategies_catalog.json",
        "strategies_comparison.json",
        "buy_hold_naftrac.json"
    }

    exigir(
        set(reporte["archivos_silver"]) == requeridos,
        "Archivos Silver incompletos"
    )

    datos = {}

    for nombre, hash_esperado in (
        reporte["archivos_silver"].items()
    ):

        path = SILVER_DIR / nombre

        exigir(
            path.is_file(),
            f"Archivo Silver faltante: {nombre}"
        )

        exigir(
            calcular_hash(path) == hash_esperado,
            f"Integridad Silver incorrecta: {nombre}"
        )

        datos[nombre] = leer_json(path)

        nombre_bronze = (
            "strategies_comparison_real.json"
            if nombre == "strategies_comparison.json"
            else nombre
        )

        exigir(
            datos[nombre] == bronze[nombre_bronze],
            f"Silver difiere de Bronze: {nombre}"
        )

    return reporte, datos


def construir_gold():
    """Construye Gold en memoria sin modificar archivos."""

    from src.agent.strategy_comparator import StrategyComparator

    calidad, datos = cargar_silver()

    catalogo = datos["strategies_catalog.json"]
    comparacion = datos["strategies_comparison.json"]
    benchmark = datos["buy_hold_naftrac.json"]

    recalculada = StrategyComparator(
        max_drawdown_limit=0.30
    ).evaluate(catalogo)

    for campo in (
        "total_estrategias",
        "aprobadas",
        "evaluaciones",
        "comparacion"
    ):

        exigir(
            comparacion[campo] == recalculada[campo],
            f"Comparacion Silver inconsistente: {campo}"
        )

    estrategias = {}

    for nombre, datos_estrategia in catalogo.items():

        metricas = datos_estrategia["strategy"]
        periodo = datos_estrategia["evaluation_period"]
        contexto = datos_estrategia["evaluation_context"]

        evaluacion = comparacion["evaluaciones"][nombre]
        resumen = evaluacion["resumen"]

        aprobada = evaluacion["aprobada"]

        motivos = []

        if contexto["no_trades"]:

            motivos.append(
                "La estrategia no realizo operaciones"
            )

        if contexto["exploratory"]:

            motivos.append(
                "Resultado exploratorio sin validacion independiente"
            )

        if metricas["sharpe_net"] <= 0:

            motivos.append(
                "Sharpe neto no positivo"
            )

        if metricas["total_return_net"] <= 0:

            motivos.append(
                "Rendimiento neto no positivo"
            )

        if abs(metricas["max_drawdown"]) > 0.30:

            motivos.append(
                "Drawdown superior al limite del 30%"
            )

        if datos_estrategia["vs_benchmark"][
            "excess_return_net"
        ] <= 0:

            motivos.append(
                "No supera el rendimiento neto del benchmark"
            )

        if contexto["auc_applicable"]:

            auc = metricas["auc"]

            if auc is None or auc <= 0.5:

                motivos.append(
                    "AUC sin capacidad discriminatoria suficiente"
                )

        if not aprobada:

            decision = "NO_BET_LIVE"

        else:

            decision = "REQUIERE_REVISION"

        estrategias[nombre] = {
            "nombre": nombre,
            "tipo": datos_estrategia["strategy_type"],
            "universo": datos_estrategia["universe"],
            "metricas": metricas,
            "comparacion_benchmark": (
                datos_estrategia["vs_benchmark"]
            ),
            "configuracion": datos_estrategia["config"],
            "periodo": periodo,
            "contexto": contexto,
            "operaciones_cerradas": periodo["closed_trades"],
            "posiciones_abiertas": periodo["open_positions"],
            "validacion": evaluacion,
            "aprobada": aprobada,
            "decision": decision,
            "motivos": motivos,
            "fuentes": datos_estrategia["data_sources"]
        }

    ranking_rendimiento = sorted(
        estrategias,
        key=lambda nombre: (
            -estrategias[nombre]["metricas"]["total_return_net"],
            nombre
        )
    )

    ranking_sharpe = sorted(
        estrategias,
        key=lambda nombre: (
            -estrategias[nombre]["metricas"]["sharpe_net"],
            nombre
        )
    )

    ranking_cumplimiento = [
        item["estrategia"]
        for item in comparacion["comparacion"]
    ]

    total_aprobadas = sum(
        item["aprobada"]
        for item in estrategias.values()
    )

    exigir(
        total_aprobadas == comparacion["aprobadas"],
        "Cantidad de estrategias aprobadas inconsistente"
    )

    referencia = benchmark["strategy"]

    reporte = {
        "capa": "gold",
        "schema_version": "1.0.0",
        "producto": "evaluacion_multiples_estrategias",
        "generated_at": ahora(),

        "calidad": {
            "estado": calidad["estado"],
            "estrategias_validadas": len(estrategias),
            "benchmark_validado": True,
            "advertencias": calidad["advertencias"],
            "errores": calidad["errores"]
        },

        "periodo_evaluacion": benchmark["evaluation_period"],

        "configuracion_comun": {
            "moneda": "MXN",
            "capital_inicial": benchmark["config"]["capital"],
            "commission_pct": benchmark["config"]["commission_pct"],
            "slippage_pct": benchmark["config"]["slippage_pct"],
            "round_trip_cost": benchmark["config"]["round_trip_cost"],
            "periods_per_year": 252,
            "metric_frequency": "daily",
            "max_drawdown_limit": 0.30
        },

        "benchmark": {
            "nombre": "NAFTRAC",
            "simbolo": benchmark["config"]["symbol"],
            "metodo": "buy_and_hold",
            "base_costos": "net",
            "metricas": referencia,
            "calidad_datos": benchmark["data_quality"],
            "fuentes": benchmark["data_sources"]
        },

        "total_estrategias": len(estrategias),
        "total_aprobadas": total_aprobadas,
        "total_no_aprobadas": len(estrategias) - total_aprobadas,

        "estrategias": estrategias,

        "rankings": {
            "rendimiento_neto": ranking_rendimiento,
            "sharpe_neto": ranking_sharpe,
            "cumplimiento": ranking_cumplimiento
        },

        "resumen": {
            "mayor_rendimiento_neto": ranking_rendimiento[0],
            "mayor_sharpe_neto": ranking_sharpe[0],
            "mayor_cumplimiento": [
                nombre
                for nombre in ranking_cumplimiento
                if estrategias[nombre]["validacion"]["resumen"][
                    "proporcion_cumplimiento"
                ] == estrategias[ranking_cumplimiento[0]][
                    "validacion"
                ]["resumen"]["proporcion_cumplimiento"]
            ],
            "estrategias_superan_benchmark": [
                nombre
                for nombre, item in estrategias.items()
                if item["comparacion_benchmark"][
                    "excess_return_net"
                ] > 0
            ],
            "estrategias_sin_operaciones": [
                nombre
                for nombre, item in estrategias.items()
                if item["contexto"]["no_trades"]
            ],
            "estrategias_exploratorias": [
                nombre
                for nombre, item in estrategias.items()
                if item["contexto"]["exploratory"]
            ]
        },

        "decision": (
            "NO_BET_LIVE"
            if total_aprobadas == 0
            else "REQUIERE_REVISION"
        ),

        "limitaciones": [
            "Resultados de simulacion historica, no inversiones reales",
            "No garantiza rendimientos futuros",
            "Universo de emisoras sujeto a sesgo de supervivencia",
            "No incluye dividendos ni intereses sobre efectivo",
            "Las posiciones abiertas se valoran al cierre",
            "El ranking exploratorio requiere validacion independiente",
            "La calidad de datos no implica aprobacion financiera"
        ],

        "trazabilidad": {
            "manifest_bronze_sha256": calidad["manifest_sha256"],
            "archivos_silver": calidad["archivos_silver"],
            "quality_report_sha256": calcular_hash(
                SILVER_DIR / "quality_report_estrategias.json"
            )
        }
    }

    return reporte


def ejecutar_gold():
    """Genera y guarda el producto Gold validado."""

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    salida = GOLD_DIR / "evaluacion_multiples_estrategias.json"

    salida.unlink(missing_ok=True)

    reporte = construir_gold()

    guardar_json(salida, reporte)

    return reporte


def ejecutar_pipeline():

    # Evitar publicar resultados anteriores si falla
    # una nueva ejecucion del pipeline.

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    (
        GOLD_DIR / "evaluacion_multiples_estrategias.json"
    ).unlink(missing_ok=True)

    (
        SILVER_DIR / "quality_report_estrategias.json"
    ).unlink(missing_ok=True)

    print("\n=== MEDALLION ESTRATEGIAS: BRONZE ===")

    bronze = ejecutar_bronze()

    for item in bronze["archivos"]:

        print(
            item["nombre"],
            item["integridad"],
            item["tamano_bytes"],
            "bytes"
        )

    print("Archivos incorporados:", bronze["total_archivos"])

    print("\n=== MEDALLION ESTRATEGIAS: SILVER ===")

    silver = ejecutar_silver()

    print("Estado:", silver["estado"])

    print(
        "Estrategias validadas:",
        len(silver["estrategias_validadas"])
    )

    print("Errores:", len(silver["errores"]))

    for error in silver["errores"]:
        print("ERROR:", error)

    print("Advertencias:", len(silver["advertencias"]))

    for advertencia in silver["advertencias"]:
        print("ADVERTENCIA:", advertencia)

    if silver["estado"] != "VALIDO":

        raise ValueError(
            "Pipeline detenido: Silver invalido"
        )

    print("\n=== MEDALLION ESTRATEGIAS: GOLD ===")

    gold = ejecutar_gold()

    print("Producto:", gold["producto"])
    print("Calidad:", gold["calidad"]["estado"])
    print("Estrategias:", gold["total_estrategias"])
    print("Aprobadas:", gold["total_aprobadas"])
    print("No aprobadas:", gold["total_no_aprobadas"])
    print("Decision:", gold["decision"])

    print("\n=== RANKING POR RENDIMIENTO NETO ===")

    for nombre in gold["rankings"]["rendimiento_neto"]:

        item = gold["estrategias"][nombre]
        metricas = item["metricas"]

        print(
            nombre,
            f'Retorno={metricas["total_return_net"]:.2%}',
            f'Sharpe={metricas["sharpe_net"]:.4f}',
            f'MDD={metricas["max_drawdown"]:.2%}',
            f'Capital=${metricas["final_equity_net"]:,.2f}'
        )

    print("\n=== ARCHIVO FINAL ===")

    print(
        GOLD_DIR / "evaluacion_multiples_estrategias.json"
    )

    print("\nPIPELINE MEDALLION COMPLETADO")

    return gold


if __name__ == "__main__":

    ejecutar_pipeline()
