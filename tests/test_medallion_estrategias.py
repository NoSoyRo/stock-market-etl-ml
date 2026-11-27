"""Pruebas de integracion del pipeline Medallion de estrategias."""

import json
import shutil

import pytest

from scripts import medallion_estrategias as med


@pytest.fixture
def entorno(tmp_path, monkeypatch):

    entradas_originales = dict(med.ENTRADAS)

    entradas = {}

    for nombre, origen in entradas_originales.items():

        destino = tmp_path / "entradas" / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(origen, destino)

        entradas[nombre] = destino

    directorio = tmp_path / "medallion"

    monkeypatch.setattr(med, "ROOT", tmp_path)
    monkeypatch.setattr(med, "ENTRADAS", entradas)
    monkeypatch.setattr(med, "MEDALLION_DIR", directorio)

    monkeypatch.setattr(
        med, "BRONZE_DIR", directorio / "bronze"
    )

    monkeypatch.setattr(
        med, "SILVER_DIR", directorio / "silver"
    )

    monkeypatch.setattr(
        med, "GOLD_DIR", directorio / "gold"
    )

    return entradas


def modificar_json(path, modificacion):

    datos = med.leer_json(path)

    modificacion(datos)

    med.guardar_json(path, datos)


def ruta_gold():

    return (
        med.GOLD_DIR
        / "evaluacion_multiples_estrategias.json"
    )


def preparar_silver():

    med.ejecutar_bronze()

    reporte = med.ejecutar_silver()

    assert reporte["estado"] == "VALIDO"

    return reporte


def test_bronze_incorpora_diez_archivos(entorno):

    reporte = med.ejecutar_bronze()

    assert reporte["total_archivos"] == 10
    assert len(med.cargar_bronze()) == 10

    for nombre, origen in entorno.items():

        assert med.calcular_hash(origen) == med.calcular_hash(
            med.BRONZE_DIR / nombre
        )


def test_bronze_detecta_archivo_faltante(entorno):

    entorno["original_topk_2.json"].unlink()

    with pytest.raises(FileNotFoundError):

        med.ejecutar_bronze()


def test_bronze_detecta_manipulacion(entorno):

    med.ejecutar_bronze()

    path = med.BRONZE_DIR / "original_topk_2.json"

    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Integridad Bronze"):

        med.cargar_bronze()


def test_silver_valida_seis_estrategias(entorno):

    reporte = preparar_silver()

    assert len(reporte["estrategias_validadas"]) == 6
    assert reporte["benchmark_validado"] is True
    assert reporte["errores"] == []
    assert len(reporte["advertencias"]) == 9


def test_silver_detecta_calendario_incorrecto(entorno):

    modificar_json(
        entorno["strategies_catalog.json"],
        lambda d: d["topk_2"]["dates"].pop()
    )

    med.ejecutar_bronze()

    reporte = med.ejecutar_silver()

    assert reporte["estado"] == "INVALIDO"
    assert reporte["errores"]


def test_silver_detecta_auc_invalido(entorno):

    modificar_json(
        entorno["strategies_catalog.json"],
        lambda d: d["topk_2"]["strategy"].update(
            {"auc": 1.5}
        )
    )

    med.ejecutar_bronze()

    reporte = med.ejecutar_silver()

    assert reporte["estado"] == "INVALIDO"

    assert any(
        "AUC" in error
        for error in reporte["errores"]
    )


def test_silver_detecta_capital_alterado(entorno):

    modificar_json(
        entorno["strategies_catalog.json"],
        lambda d: d["sma_50_200"]["strategy"].update(
            {"final_equity_net": 999999}
        )
    )

    med.ejecutar_bronze()

    reporte = med.ejecutar_silver()

    assert reporte["estado"] == "INVALIDO"

    assert any(
        "capital final" in error
        for error in reporte["errores"]
    )


def test_silver_detecta_comparacion_alterada(entorno):

    modificar_json(
        entorno["strategies_comparison_real.json"],
        lambda d: d.update({"aprobadas": 6})
    )

    med.ejecutar_bronze()

    reporte = med.ejecutar_silver()

    assert reporte["estado"] == "INVALIDO"

    assert any(
        "comparador" in error
        for error in reporte["errores"]
    )


def test_gold_consolida_resultados(entorno):

    preparar_silver()

    gold = med.ejecutar_gold()

    assert gold["capa"] == "gold"
    assert gold["calidad"]["estado"] == "VALIDO"
    assert gold["total_estrategias"] == 6
    assert gold["total_aprobadas"] == 0
    assert gold["total_no_aprobadas"] == 6
    assert gold["decision"] == "NO_BET_LIVE"

    assert ruta_gold().is_file()

    assert med.leer_json(ruta_gold()) == gold


def test_gold_rankings(entorno):

    preparar_silver()

    gold = med.ejecutar_gold()

    resumen = gold["resumen"]

    assert resumen["mayor_rendimiento_neto"] == "sma_50_200"
    assert resumen["mayor_sharpe_neto"] == "rsi_14"

    assert set(resumen["mayor_cumplimiento"]) == {
        "rsi_14", "sma_50_200"
    }

    assert resumen["estrategias_superan_benchmark"] == []


def test_gold_rechaza_silver_invalido(entorno):

    preparar_silver()

    path = med.SILVER_DIR / "quality_report_estrategias.json"

    modificar_json(
        path,
        lambda d: d.update({"estado": "INVALIDO"})
    )

    with pytest.raises(ValueError, match="Silver"):

        med.ejecutar_gold()

    assert not ruta_gold().exists()


def test_gold_detecta_manipulacion_silver(entorno):

    preparar_silver()

    path = med.SILVER_DIR / "strategies_catalog.json"

    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Integridad Silver"):

        med.ejecutar_gold()

    assert not ruta_gold().exists()


def test_gold_detecta_manipulacion_bronze(entorno):

    preparar_silver()

    path = med.BRONZE_DIR / "original_topk_5.json"

    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Integridad Bronze"):

        med.ejecutar_gold()

    assert not ruta_gold().exists()


def test_pipeline_completo(entorno):

    gold = med.ejecutar_pipeline()

    assert gold["total_estrategias"] == 6
    assert gold["calidad"]["estado"] == "VALIDO"

    assert len(list(med.BRONZE_DIR.glob("*.json"))) == 11
    assert len(list(med.SILVER_DIR.glob("*.json"))) == 4
    assert len(list(med.GOLD_DIR.glob("*.json"))) == 1


def test_pipeline_elimina_gold_obsoleto(entorno):

    med.ejecutar_pipeline()

    assert ruta_gold().exists()

    entorno["original_topk_2.json"].unlink()

    with pytest.raises(FileNotFoundError):

        med.ejecutar_pipeline()

    assert not ruta_gold().exists()


def test_pipeline_preserva_entradas(entorno):

    hashes_originales = {
        nombre: med.calcular_hash(path)
        for nombre, path in entorno.items()
    }

    med.ejecutar_pipeline()

    hashes_finales = {
        nombre: med.calcular_hash(path)
        for nombre, path in entorno.items()
    }

    assert hashes_originales == hashes_finales
