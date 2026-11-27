"""Pruebas de la capa semantica de evaluacion."""

import json
import shutil

import pytest

from scripts import medallion_estrategias as med
from src.agents_mcp.domains import evaluacion_semantica as sem


@pytest.fixture
def entorno(tmp_path, monkeypatch):

    entradas = {}

    for nombre, origen in med.ENTRADAS.items():

        destino = tmp_path / "entradas" / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(origen, destino)

        entradas[nombre] = destino

    base = tmp_path / "medallion"

    monkeypatch.setattr(med, "ROOT", tmp_path)
    monkeypatch.setattr(med, "ENTRADAS", entradas)
    monkeypatch.setattr(med, "MEDALLION_DIR", base)
    monkeypatch.setattr(med, "BRONZE_DIR", base / "bronze")
    monkeypatch.setattr(med, "SILVER_DIR", base / "silver")
    monkeypatch.setattr(med, "GOLD_DIR", base / "gold")

    gold = (
        base / "gold"
        / "evaluacion_multiples_estrategias.json"
    )

    monkeypatch.setattr(sem, "ROOT", tmp_path)
    monkeypatch.setattr(sem, "GOLD_PATH", gold)

    med.ejecutar_pipeline()

    return gold


def modificar_gold(ruta, cambio):

    datos = json.loads(
        ruta.read_text(encoding="utf-8")
    )

    cambio(datos)

    ruta.write_text(
        json.dumps(datos),
        encoding="utf-8"
    )


def test_resumen_general(entorno):

    r = sem.resumen_estrategias()["resultado"]

    assert r["total_estrategias"] == 6
    assert r["total_aprobadas"] == 0
    assert r["total_no_aprobadas"] == 6
    assert r["decision"] == "NO_BET_LIVE"


def test_consultar_sma(entorno):

    r = sem.consultar_estrategia(
        "sma_50_200"
    )["resultado"]

    assert r["metricas"]["total_return_net"] > 0
    assert r["metricas"]["sharpe_net"] > 0
    assert r["aprobada"] is False


def test_consultar_rsi(entorno):

    r = sem.consultar_estrategia(
        "rsi_14"
    )["resultado"]

    assert r["metricas"]["auc"] is None
    assert r["metricas"]["sharpe_net"] > 0


def test_normalizar_nombre(entorno):

    assert (
        sem.consultar_estrategia(" RSI_14 ")
        == sem.consultar_estrategia("rsi_14")
    )


def test_estrategia_desconocida(entorno):

    with pytest.raises(
        ValueError,
        match="Estrategia desconocida"
    ):

        sem.consultar_estrategia("inventada")


@pytest.mark.parametrize(
    "criterio",
    [
        "rendimiento_neto",
        "sharpe_neto",
        "cumplimiento"
    ]
)
def test_rankings(entorno, criterio):

    r = sem.comparar_estrategias(
        criterio
    )["resultado"]

    assert r["criterio"] == criterio
    assert len(r["ranking"]) == 6


def test_criterio_desconocido(entorno):

    with pytest.raises(
        ValueError,
        match="Criterio invalido"
    ):

        sem.comparar_estrategias("inventado")


def test_benchmark(entorno):

    r = sem.consultar_benchmark()["resultado"]

    assert r["nombre"] == "NAFTRAC"

    assert (
        r["metricas"]["final_equity_net"]
        == pytest.approx(124535.33, abs=0.01)
    )


def test_validacion(entorno):

    r = sem.consultar_validacion(
        "sma_50_200"
    )["resultado"]

    assert r["aprobada"] is False
    assert r["decision"] == "NO_BET_LIVE"

    assert any(
        "benchmark" in motivo
        for motivo in r["motivos"]
    )


def test_calidad(entorno):

    r = sem.consultar_calidad()["resultado"]

    assert r["calidad"]["estado"] == "VALIDO"
    assert not r["calidad"]["errores"]
    assert len(r["fuentes_estrategias"]) == 6
    assert r["trazabilidad"]


def test_decision(entorno):

    r = sem.consultar_decision()["resultado"]

    assert r["decision"] == "NO_BET_LIVE"

    assert all(
        not e["aprobada"]
        for e in r["estrategias"].values()
    )


def test_aviso_financiero(entorno):

    r = sem.resumen_estrategias()

    assert "simulaciones historicas" in r["aviso"]
    assert "asesoria financiera" in r["aviso"]


def test_serializacion(entorno):

    r = sem.resumen_estrategias()

    assert json.loads(sem.serializar(r)) == r


def test_gold_faltante(entorno):

    entorno.unlink()

    with pytest.raises(FileNotFoundError):

        sem.resumen_estrategias()


def test_gold_invalido(entorno):

    modificar_gold(
        entorno,
        lambda d: d["calidad"].update(
            estado="INVALIDO"
        )
    )

    with pytest.raises(
        ValueError,
        match="Gold invalido"
    ):

        sem.resumen_estrategias()


def test_gold_aprobaciones_alteradas(entorno):

    modificar_gold(
        entorno,
        lambda d: d.update(total_aprobadas=6)
    )

    with pytest.raises(
        ValueError,
        match="aprobaciones inconsistente"
    ):

        sem.resumen_estrategias()


def test_gold_decision_alterada(entorno):

    modificar_gold(
        entorno,
        lambda d: d.update(
            decision="REQUIERE_REVISION"
        )
    )

    with pytest.raises(
        ValueError,
        match="Decision global inconsistente"
    ):

        sem.resumen_estrategias()


def test_gold_trazabilidad_alterada(entorno):

    modificar_gold(
        entorno,
        lambda d: d["trazabilidad"].update(
            manifest_bronze_sha256="incorrecto"
        )
    )

    with pytest.raises(
        ValueError,
        match="Bronze actual"
    ):

        sem.resumen_estrategias()


def test_bronze_manipulado(entorno):

    ruta = med.BRONZE_DIR / "original_topk_2.json"

    ruta.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):

        sem.resumen_estrategias()


def test_silver_manipulado(entorno):

    ruta = med.SILVER_DIR / "strategies_catalog.json"

    ruta.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):

        sem.resumen_estrategias()


def test_consultas_solo_lectura(entorno):

    archivos = sorted(
        med.MEDALLION_DIR.rglob("*.json")
    )

    antes = {
        ruta: med.calcular_hash(ruta)
        for ruta in archivos
    }

    sem.resumen_estrategias()
    sem.consultar_estrategia("rsi_14")
    sem.comparar_estrategias()
    sem.consultar_benchmark()
    sem.consultar_validacion("sma_50_200")
    sem.consultar_calidad()
    sem.consultar_decision()

    despues = {
        ruta: med.calcular_hash(ruta)
        for ruta in archivos
    }

    assert antes == despues


@pytest.mark.parametrize(
    "ruta_claves,valor",
    [
        (
            ("estrategias", "sma_50_200", "metricas", "total_return_net"),
            0.99,
        ),
        (
            ("estrategias", "rsi_14", "metricas", "sharpe_net"),
            5.0,
        ),
        (
            ("estrategias", "sma_50_200", "metricas", "final_equity_net"),
            999999.0,
        ),
        (
            ("benchmark", "metricas", "total_return_net"),
            -0.50,
        ),
        (
            ("rankings", "rendimiento_neto"),
            [],
        ),
        (
            ("configuracion_comun", "commission_pct"),
            0.0,
        ),
        (
            ("trazabilidad", "archivos_silver"),
            {},
        ),
    ],
)
def test_gold_detecta_alteracion_contenido(
    entorno, ruta_claves, valor
):
    def alterar(datos):
        destino = datos

        for clave in ruta_claves[:-1]:
            destino = destino[clave]

        assert ruta_claves[-1] in destino
        destino[ruta_claves[-1]] = valor

    modificar_gold(entorno, alterar)

    with pytest.raises(
        ValueError,
        match="Contenido Gold inconsistente con Silver validado",
    ):
        sem.cargar_producto()
