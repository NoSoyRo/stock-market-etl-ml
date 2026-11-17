"""cargar_golden guarda el golden en memoria y lo recarga solo si el archivo cambia."""
from __future__ import annotations

import os

import pandas as pd

from productos.fundamentales_pit import pit


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA"],
        "quarter_end": ["2026-03-31"],
        "available_date": ["2026-06-09"],
    })


def _falso_parquet(monkeypatch) -> list[int]:
    lecturas: list[int] = []

    def leer(path, *a, **k):
        lecturas.append(1)
        return _df()

    monkeypatch.setattr(pit.pd, "read_parquet", leer)
    return lecturas


def test_la_segunda_consulta_no_vuelve_a_leer_el_archivo(tmp_path, monkeypatch):
    archivo = tmp_path / "golden.parquet"
    archivo.write_bytes(b"x")
    lecturas = _falso_parquet(monkeypatch)
    pit._CACHE.clear()
    pit.cargar_golden(archivo)
    pit.cargar_golden(archivo)
    assert len(lecturas) == 1


def test_si_el_archivo_cambia_se_vuelve_a_leer(tmp_path, monkeypatch):
    archivo = tmp_path / "golden.parquet"
    archivo.write_bytes(b"x")
    lecturas = _falso_parquet(monkeypatch)
    pit._CACHE.clear()
    pit.cargar_golden(archivo)
    archivo.write_bytes(b"xx")                      # otro tamaño
    os.utime(archivo, ns=(10**18, 10**18))          # y otra fecha de modificación
    pit.cargar_golden(archivo)
    assert len(lecturas) == 2


def test_modificar_el_resultado_no_contamina_la_cache(tmp_path, monkeypatch):
    archivo = tmp_path / "golden.parquet"
    archivo.write_bytes(b"x")
    _falso_parquet(monkeypatch)
    pit._CACHE.clear()
    a = pit.cargar_golden(archivo)
    a["ticker"] = "ZZZ"
    assert pit.cargar_golden(archivo)["ticker"].iloc[0] == "AAA"
