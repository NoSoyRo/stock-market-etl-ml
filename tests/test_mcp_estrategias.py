"""Pruebas de integracion MCP del producto Gold de estrategias."""

import asyncio
import json

import pytest

from src.agents_mcp.domains import evaluacion_semantica as sem
from src.agents_mcp.domains.evaluacion import evaluacion_agent
from src.agents_mcp.domains.evaluacion_tools import TOOLS_ESTRATEGIAS
from src.agents_mcp.orchestrator import (
    build_options,
    _ALL_TOOLS,
    _SERVER_NAME,
)
from mcp_server.server import mcp


HERRAMIENTAS_GOLD = {
    "resumen_estrategias",
    "consultar_estrategia",
    "comparar_estrategias",
    "consultar_benchmark",
    "consultar_validacion",
    "consultar_calidad",
    "consultar_decision",
}


def test_herramientas_gold_registradas():
    nombres = [t.name for t in TOOLS_ESTRATEGIAS]

    assert len(nombres) == 7
    assert set(nombres) == HERRAMIENTAS_GOLD


def test_agente_evaluacion_integrado():
    nombres = [t.name for t in evaluacion_agent.tools]

    assert len(nombres) == len(set(nombres))
    assert len(nombres) == 10
    assert HERRAMIENTAS_GOLD.issubset(nombres)


def test_herramientas_originales_conservadas():
    nombres = {t.name for t in evaluacion_agent.tools}

    assert {
        "schema_evaluacion",
        "query_evaluacion",
        "consultar_metricas_evaluacion",
    }.issubset(nombres)


def test_orquestador_integrado():
    nombres = [t.name for t in _ALL_TOOLS]

    assert len(nombres) == len(set(nombres))
    assert "consultar_evaluacion" in nombres


def test_orquestador_permite_evaluacion():
    opciones = build_options()

    assert (
        f"mcp__{_SERVER_NAME}__consultar_evaluacion"
        in opciones.allowed_tools
    )


def test_servidor_mcp_externo():
    herramientas = asyncio.run(mcp.list_tools())
    nombres = [t.name for t in herramientas]

    assert len(nombres) == len(set(nombres))
    assert HERRAMIENTAS_GOLD.issubset(nombres)
    assert "preguntar_evaluacion" in nombres
    assert "vision_360" in nombres


@pytest.mark.parametrize(
    "nombre, argumentos",
    [
        ("resumen_estrategias", {}),
        ("consultar_estrategia", {"nombre": "sma_50_200"}),
        ("comparar_estrategias", {"criterio": "sharpe_neto"}),
        ("consultar_benchmark", {}),
        ("consultar_validacion", {"nombre": "rsi_14"}),
        ("consultar_calidad", {}),
        ("consultar_decision", {}),
    ],
)
def test_consulta_herramienta_interna(
    monkeypatch, nombre, argumentos
):
    esperado = {
        "producto": "evaluacion_multiples_estrategias",
        "consulta": nombre,
        "resultado": {"prueba": True},
    }

    monkeypatch.setattr(
        sem, nombre, lambda *args: esperado
    )

    herramienta = next(
        t for t in TOOLS_ESTRATEGIAS
        if t.name == nombre
    )

    respuesta = asyncio.run(
        herramienta.handler(argumentos)
    )

    assert not respuesta.get("isError", False)

    contenido = json.loads(
        respuesta["content"][0]["text"]
    )

    assert contenido == esperado


def test_error_controlado_herramienta(monkeypatch):
    def consulta_invalida(*args):
        raise ValueError("Estrategia inexistente")

    monkeypatch.setattr(
        sem, "consultar_estrategia", consulta_invalida
    )

    herramienta = next(
        t for t in TOOLS_ESTRATEGIAS
        if t.name == "consultar_estrategia"
    )

    respuesta = asyncio.run(
        herramienta.handler({"nombre": "inexistente"})
    )

    assert respuesta["isError"] is True
    assert "Estrategia inexistente" in (
        respuesta["content"][0]["text"]
    )


# Pruebas de integracion con producto Gold real

from test_evaluacion_semantica import entorno


CONSULTAS_GOLD = [
    ("resumen_estrategias", {}),
    ("consultar_estrategia", {"nombre": "sma_50_200"}),
    ("comparar_estrategias", {"criterio": "sharpe_neto"}),
    ("consultar_benchmark", {}),
    ("consultar_validacion", {"nombre": "rsi_14"}),
    ("consultar_calidad", {}),
    ("consultar_decision", {}),
]


def ejecutar_mcp(canal, nombre, argumentos):

    if canal == "interno":

        herramienta = next(
            t for t in TOOLS_ESTRATEGIAS
            if t.name == nombre
        )

        respuesta = asyncio.run(
            herramienta.handler(argumentos)
        )

        return (
            respuesta.get("isError", False),
            respuesta["content"][0]["text"]
        )

    from mcp.server.mcpserver.exceptions import UnexpectedToolError

    try:

        respuesta = asyncio.run(
            mcp.call_tool(nombre, argumentos)
        )

    except UnexpectedToolError as error:

        return True, str(error)

    return respuesta.is_error, respuesta.content[0].text


@pytest.mark.parametrize("canal", ["interno", "externo"])
@pytest.mark.parametrize("nombre, argumentos", CONSULTAS_GOLD)
def test_consulta_gold_real(entorno, canal, nombre, argumentos):

    error, texto = ejecutar_mcp(
        canal, nombre, argumentos
    )

    assert not error

    contenido = json.loads(texto)

    esperado = getattr(sem, nombre)(**argumentos)

    assert contenido == esperado
    assert contenido["producto"] == "evaluacion_multiples_estrategias"
    assert contenido["consulta"] == nombre
    assert contenido["schema_version"] == "1.0.0"
    assert contenido["resultado"]
    assert contenido["aviso"]

    assert contenido["fuente"].endswith(
        "evaluacion_multiples_estrategias.json"
    )

    if nombre in ("resumen_estrategias", "consultar_decision"):

        assert contenido["resultado"]["decision"] == "NO_BET_LIVE"
        assert contenido["resultado"]["total_aprobadas"] == 0


@pytest.mark.parametrize("canal", ["interno", "externo"])
def test_rechazar_estrategia_inexistente(entorno, canal):

    error, texto = ejecutar_mcp(
        canal,
        "consultar_estrategia",
        {"nombre": "estrategia_inexistente"}
    )

    assert error is True


@pytest.mark.parametrize("canal", ["interno", "externo"])
def test_rechazar_gold_faltante(entorno, canal):

    entorno.unlink()

    assert not entorno.exists()

    error, texto = ejecutar_mcp(
        canal,
        "resumen_estrategias",
        {}
    )

    assert error is True


@pytest.mark.parametrize("canal", ["interno", "externo"])
def test_rechazar_gold_manipulado(entorno, canal):

    datos = json.loads(
        entorno.read_text(encoding="utf-8")
    )

    datos["estrategias"]["sma_50_200"][
        "metricas"
    ]["total_return_net"] = 0.99

    entorno.write_text(
        json.dumps(datos),
        encoding="utf-8"
    )

    error, texto = ejecutar_mcp(
        canal,
        "consultar_estrategia",
        {"nombre": "sma_50_200"}
    )

    assert error is True
