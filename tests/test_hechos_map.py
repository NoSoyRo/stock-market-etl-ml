"""Tests del mapeo código `hechos` → ticker MX (datos sintéticos, sin red)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.hechos_map import (
    ALTA_CORR,
    build_map,
    correlate_codes,
    missing_days,
    normalize_code,
    resolve_ticker,
)


def _wide(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Matriz fecha×código con ruido independiente y un volumen objetivo."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-01", periods=n)
    target = pd.Series(np.exp(rng.normal(15, 0.5, n)), index=idx)
    wide = pd.DataFrame(
        {c: np.exp(rng.normal(12, 0.5, n)) for c in ["RUIDO1", "RUIDO2", "RUIDO3"]},
        index=idx,
    )
    return wide, target


def test_normalize_code_quita_asterisco_y_sufijo():
    assert normalize_code("GENTERA*") == "GENTERA"
    assert normalize_code("walmex.mx") == "WALMEX"
    assert normalize_code(" AMXB ") == "AMXB"


def test_alias_explicito_resuelve_sin_correlacion():
    wide, target = _wide()
    wide["GENTERA*"] = target * 1.0  # mismo volumen, pero el método debe ser alias
    res = resolve_ticker("GENTERA", target, wide, aliases={"GENTERA": ["GENTERA*"]})
    assert res["codigo"] == "GENTERA*"
    assert res["metodo"] == "alias"


def test_nombre_normalizado_resuelve():
    wide, target = _wide()
    wide["TLEVISACPO"] = target * 2.0
    res = resolve_ticker("TLEVISACPO", target, wide, aliases={})
    assert res["codigo"] == "TLEVISACPO"
    assert res["metodo"] == "nombre"


def test_correlacion_encuentra_codigo_con_otro_nombre():
    wide, target = _wide()
    rng = np.random.default_rng(1)
    wide["XYZ123"] = target * np.exp(rng.normal(0, 0.05, len(target)))
    res = resolve_ticker("AMXB", target, wide, aliases={})
    assert res["codigo"] == "XYZ123"
    assert res["metodo"] == "correlacion"
    assert res["confianza"] == "alta"
    assert res["corr"] >= ALTA_CORR


def test_desempate_segundo_cercano_baja_confianza():
    wide, target = _wide()
    rng = np.random.default_rng(2)
    wide["A1"] = target * np.exp(rng.normal(0, 0.05, len(target)))
    wide["A2"] = target * np.exp(rng.normal(0, 0.05, len(target)))  # casi igual de bueno
    res = resolve_ticker("AMXB", target, wide, aliases={})
    assert res["codigo"] in {"A1", "A2"}
    assert res["confianza"] != "alta"  # segundo > 0.6: ambiguo
    assert res["segundo_corr"] > 0.6


def test_desempate_determinista_por_volumen_total():
    # Dos códigos con correlación idéntica: gana el de mayor volumen total.
    wide, target = _wide()
    wide["CHICO"] = target * 0.1
    wide["GRANDE"] = target * 10.0
    res = resolve_ticker("AMXB", target, wide, aliases={})
    assert res["codigo"] == "GRANDE"


def test_sin_candidato_devuelve_none():
    wide, target = _wide()
    res = resolve_ticker("AMXB", target, wide, aliases={})
    assert res["codigo"] is None
    assert res["confianza"] == "ninguna"


def test_correlate_codes_ignora_pocos_dias():
    wide, target = _wide()
    wide["POCOS"] = np.nan
    wide.loc[wide.index[:5], "POCOS"] = target.iloc[:5]
    corr = correlate_codes(target, wide, min_days=30)
    assert "POCOS" not in corr.index


def test_build_map_conflicto_gana_mayor_correlacion():
    wide, t1 = _wide()
    rng = np.random.default_rng(3)
    wide["COMUN"] = t1 * np.exp(rng.normal(0, 0.02, len(t1)))
    t2 = t1 * np.exp(rng.normal(0, 0.4, len(t1)))  # peor parecido
    m = build_map({"T1": t1, "T2": t2}, wide, aliases={})
    assert m["COMUN"]["ticker"] == "T1"
    assert "T2" not in {v["ticker"] for v in m.values()}


def test_missing_days_separa_captura_de_feriado():
    hechos = pd.to_datetime(["2026-03-02", "2026-03-04"])
    # 03-03 mercado con volumen (hueco de captura); 03-05 sin volumen (feriado)
    vol = pd.Series(
        [10.0, 10.0, 10.0, 0.0],
        index=pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"]),
    )
    out = missing_days(hechos, vol)
    assert out["captura"] == ["2026-03-03"]
    assert out["sin_mercado"] == ["2026-03-05"]


def test_nombre_con_segundo_parecido_sigue_alta():
    wide, target = _wide()
    rng = np.random.default_rng(4)
    wide["AC"] = target * np.exp(rng.normal(0, 0.05, len(target)))
    wide["OTRA"] = target * np.exp(rng.normal(0, 0.3, len(target)))  # otra emisora correlacionada
    res = resolve_ticker("AC", target, wide, aliases={})
    assert res["metodo"] == "nombre" and res["confianza"] == "alta"
