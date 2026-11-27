"""Capa semantica del producto Gold de multiples estrategias."""

import json
from pathlib import Path

from scripts import medallion_estrategias as med


ROOT = Path(__file__).resolve().parents[3]

GOLD_PATH = (
    ROOT
    / "data"
    / "eval"
    / "medallion"
    / "gold"
    / "evaluacion_multiples_estrategias.json"
)

AVISO = (
    "Resultados de simulaciones historicas con fines educativos. "
    "No constituyen asesoria financiera ni garantizan "
    "rendimientos futuros."
)


def exigir(condicion, mensaje):

    if not condicion:
        raise ValueError(mensaje)


def cargar_producto():

    if not GOLD_PATH.is_file():

        raise FileNotFoundError(
            "Producto Gold no disponible. Ejecuta: "
            "python -m scripts.medallion_estrategias"
        )

    datos = med.leer_json(GOLD_PATH)

    exigir(
        datos["capa"] == "gold",
        "Capa incorrecta"
    )

    exigir(
        datos["producto"] == "evaluacion_multiples_estrategias",
        "Producto incorrecto"
    )

    exigir(
        datos["schema_version"] == "1.0.0",
        "Version de esquema no compatible"
    )

    exigir(
        datos["calidad"]["estado"] == "VALIDO",
        "Producto Gold invalido"
    )

    exigir(
        not datos["calidad"]["errores"],
        "El producto contiene errores de calidad"
    )

    estrategias = datos["estrategias"]

    exigir(
        isinstance(estrategias, dict) and estrategias,
        "Catalogo de estrategias invalido"
    )

    exigir(
        len(estrategias) == datos["total_estrategias"],
        "Cantidad de estrategias inconsistente"
    )

    exigir(
        all(
            isinstance(e["aprobada"], bool)
            for e in estrategias.values()
        ),
        "Estado de aprobacion invalido"
    )

    aprobadas = sum(
        e["aprobada"]
        for e in estrategias.values()
    )

    exigir(
        aprobadas == datos["total_aprobadas"],
        "Cantidad de aprobaciones inconsistente"
    )

    exigir(
        len(estrategias) - aprobadas
        == datos["total_no_aprobadas"],
        "Cantidad de rechazos inconsistente"
    )

    decision_esperada = (
        "REQUIERE_REVISION"
        if aprobadas
        else "NO_BET_LIVE"
    )

    exigir(
        datos["decision"] == decision_esperada,
        "Decision global inconsistente"
    )

    # Verificar las fuentes y su integridad mediante
    # las comprobaciones existentes de Medallion.
    med.cargar_silver()

    trazabilidad = datos["trazabilidad"]

    manifest = (
        med.BRONZE_DIR / "manifest_estrategias.json"
    )

    quality = (
        med.SILVER_DIR / "quality_report_estrategias.json"
    )

    exigir(
        med.calcular_hash(manifest)
        == trazabilidad["manifest_bronze_sha256"],
        "Gold no corresponde al Bronze actual"
    )

    exigir(
        med.calcular_hash(quality)
        == trazabilidad["quality_report_sha256"],
        "Gold no corresponde al Silver actual"
    )

    for nombre, sha256 in trazabilidad["archivos_silver"].items():

        exigir(
            nombre in {
                "strategies_catalog.json",
                "strategies_comparison.json",
                "buy_hold_naftrac.json"
            },
            "Fuente Silver no reconocida"
        )

        exigir(
            med.calcular_hash(med.SILVER_DIR / nombre)
            == sha256,
            f"Fuente Silver alterada: {nombre}"
        )

    esperado = med.construir_gold()

    almacenado = {
        clave: valor
        for clave, valor in datos.items()
        if clave != "generated_at"
    }

    reconstruido = {
        clave: valor
        for clave, valor in esperado.items()
        if clave != "generated_at"
    }

    exigir(
        serializar(almacenado) == serializar(reconstruido)
        or json.dumps(
            almacenado, sort_keys=True, allow_nan=False
        ) == json.dumps(
            reconstruido, sort_keys=True, allow_nan=False
        ),
        "Contenido Gold inconsistente con Silver validado"
    )

    return datos


def respuesta(datos, consulta, resultado):

    return {
        "consulta": consulta,
        "producto": datos["producto"],
        "schema_version": datos["schema_version"],
        "generado_en": datos["generated_at"],
        "periodo_evaluacion": datos["periodo_evaluacion"],
        "configuracion_comun": datos["configuracion_comun"],
        "fuente": str(GOLD_PATH.relative_to(ROOT)),
        "resultado": resultado,
        "aviso": AVISO
    }


def obtener_estrategia(datos, nombre):

    nombre = nombre.strip().lower()

    if nombre not in datos["estrategias"]:

        disponibles = ", ".join(
            sorted(datos["estrategias"])
        )

        raise ValueError(
            f"Estrategia desconocida: {nombre}. "
            f"Disponibles: {disponibles}"
        )

    return datos["estrategias"][nombre]


def resumen_estrategias():

    datos = cargar_producto()

    resultado = {
        "total_estrategias": datos["total_estrategias"],
        "total_aprobadas": datos["total_aprobadas"],
        "total_no_aprobadas": datos["total_no_aprobadas"],
        "decision": datos["decision"],
        "estrategias_disponibles": sorted(
            datos["estrategias"]
        ),
        "resumen": datos["resumen"]
    }

    return respuesta(datos, "resumen_estrategias", resultado)


def consultar_estrategia(nombre):

    datos = cargar_producto()

    estrategia = obtener_estrategia(datos, nombre)

    return respuesta(
        datos,
        "consultar_estrategia",
        estrategia
    )


def comparar_estrategias(criterio="rendimiento_neto"):

    datos = cargar_producto()

    criterios = {
        "rendimiento_neto",
        "sharpe_neto",
        "cumplimiento"
    }

    if criterio not in criterios:

        raise ValueError(
            "Criterio invalido. Opciones: "
            + ", ".join(sorted(criterios))
        )

    resultado = {
        "criterio": criterio,
        "ranking": datos["rankings"][criterio],
        "benchmark": datos["benchmark"],
        "resumen": datos["resumen"]
    }

    return respuesta(
        datos,
        "comparar_estrategias",
        resultado
    )


def consultar_benchmark():

    datos = cargar_producto()

    return respuesta(
        datos,
        "consultar_benchmark",
        datos["benchmark"]
    )


def consultar_validacion(nombre):

    datos = cargar_producto()

    estrategia = obtener_estrategia(datos, nombre)

    resultado = {
        "nombre": estrategia["nombre"],
        "aprobada": estrategia["aprobada"],
        "validacion": estrategia["validacion"],
        "decision": estrategia["decision"],
        "motivos": estrategia["motivos"]
    }

    return respuesta(
        datos,
        "consultar_validacion",
        resultado
    )


def consultar_calidad():

    datos = cargar_producto()

    resultado = {
        "calidad": datos["calidad"],
        "trazabilidad": datos["trazabilidad"],
        "limitaciones": datos["limitaciones"],
        "fuentes_benchmark": datos["benchmark"]["fuentes"],
        "fuentes_estrategias": {
            nombre: estrategia["fuentes"]
            for nombre, estrategia
            in datos["estrategias"].items()
        }
    }

    return respuesta(
        datos,
        "consultar_calidad",
        resultado
    )


def consultar_decision():

    datos = cargar_producto()

    resultado = {
        "decision": datos["decision"],
        "total_aprobadas": datos["total_aprobadas"],
        "total_no_aprobadas": datos["total_no_aprobadas"],
        "estrategias": {
            nombre: {
                "aprobada": estrategia["aprobada"],
                "decision": estrategia["decision"],
                "motivos": estrategia["motivos"]
            }
            for nombre, estrategia
            in datos["estrategias"].items()
        },
        "limitaciones": datos["limitaciones"]
    }

    return respuesta(
        datos,
        "consultar_decision",
        resultado
    )


def serializar(resultado):

    return json.dumps(
        resultado,
        indent=2,
        ensure_ascii=False,
        allow_nan=False
    )
