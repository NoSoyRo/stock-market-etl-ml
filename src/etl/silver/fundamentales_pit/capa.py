"""SILVER de Fundamentales PIT (Equipo Rojo).

Convierte los JSON de bronze en una tabla con UNA fila por emisora y trimestre,
con MONTOS (no solo ratios) y de dónde salió cada cifra. Si un trimestre llega
dos veces (reporte propio y comparativo del año siguiente), gana la cifra
original: el comparativo se publica un año después y puede venir reexpresado.

Sale en data/fundamentals/silver/fundamentales_trimestre.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from productos.fundamentales_pit.pit import BASE, ROOT
from src.etl.bronze.fundamentales_pit.capa import BRONZE

FUENTE = "databursatil:/v2/financieros"
SILVER = BASE / "silver" / "fundamentales_trimestre.parquet"
LEGACY = ROOT / "data" / "fundamentals.parquet"

# Campos de resultado_trimestre (nombres tal como los usa src/api/fundamentals.py).
CAMPOS = {
    "ingresos": ("revenue",),
    "utilidad_bruta": ("grossprofit",),
    "utilidad_operacion": ("profitlossfromoperatingactivities",),
    "utilidad_neta": ("profitloss",),
    # UPA: NO verificado que la API lo traiga. Si viene, se toma; si no, queda NaN.
    "upa": ("basicearningslosspershare", "basicearningspershare",
            "dilutedearningslosspershare", "earningspershare"),
}

SILVER_COLS = ["ticker", "quarter_end", "ingresos", "utilidad_bruta", "utilidad_operacion",
               "utilidad_neta", "upa", "origen", "reexpresado", "formato",
               "primera_descarga", "fuente"]


def _monto(item) -> float:
    """Line item ['etiqueta', monto] -> monto; cualquier otra cosa -> NaN."""
    if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], (int, float)):
        return float(item[1])
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        return float(item)
    return np.nan


def _fin_de_periodo(pkey: str) -> pd.Timestamp | None:
    """'2024-01-01_2024-03-31' -> 2024-03-31."""
    try:
        return pd.Timestamp(pkey.split("_")[1])
    except Exception:
        return None


def _periodo_de(quarter_end: pd.Timestamp) -> str:
    return f"{quarter_end.quarter}T_{quarter_end.year}"


def filas_de_bronze(registro: dict) -> list[dict]:
    """Un archivo bronze -> filas candidatas de silver (una por periodo que traiga)."""
    filas = []
    fin = registro.get("financieros", "resultado_trimestre")
    data = (registro.get("payload") or {}).get(fin)
    if registro.get("status_code") != 200 or not isinstance(data, dict):
        return filas
    for pkey, campos in data.items():
        qend = _fin_de_periodo(pkey)
        if qend is None or not isinstance(campos, dict):
            continue
        fila = {"ticker": registro["ticker"], "quarter_end": qend,
                "origen": "reporte_propio" if _periodo_de(qend) == registro["periodo"] else "comparativo",
                "descargado": pd.Timestamp(registro["fetched_at"]).tz_convert(None)}
        for col, claves in CAMPOS.items():
            fila[col] = next((v for k in claves if not np.isnan(v := _monto(campos.get(k)))), np.nan)
        # Bancos y aseguradoras no reportan revenue/grossprofit: se marca en vez de
        # descartar la emisora entera (hipótesis de por qué fallan GFNORTEO,
        # GENTERA, Q y GFINBURO; confirmar con el JSON crudo).
        sin_estructura = np.isnan(fila["ingresos"]) and np.isnan(fila["utilidad_bruta"])
        fila["formato"] = "sin_ingresos_corporativos" if sin_estructura and campos else "corporativo"
        filas.append(fila)
    return filas


def _elegir(grupo: pd.DataFrame) -> pd.Series:
    """Una fila por (ticker, trimestre): gana la cifra ORIGINAL (reporte propio,
    la primera descarga); el comparativo solo si no hay otra cosa."""
    g = grupo.assign(_rank=(grupo["origen"] != "reporte_propio").astype(int))
    g = g.sort_values(["_rank", "descargado"])
    elegida = g.iloc[0].copy()
    otras = g.iloc[1:]
    reexp = False
    for c in ("ingresos", "utilidad_neta"):
        a = elegida[c]
        for b in otras[c]:
            if np.isfinite(a) and np.isfinite(b) and abs(a) > 0 and abs(b - a) / abs(a) > 0.005:
                reexp = True
    elegida["reexpresado"] = reexp
    elegida["primera_descarga"] = g["descargado"].min()
    return elegida.drop(labels=["_rank", "descargado"])


def silver_desde_bronze(origen: Path = BRONZE) -> pd.DataFrame:
    filas = []
    for archivo in sorted(origen.glob("*/*/*.json")):
        filas.extend(filas_de_bronze(json.loads(archivo.read_text(encoding="utf-8"))))
    if not filas:
        return pd.DataFrame(columns=SILVER_COLS)
    df = pd.DataFrame(filas)
    df = (df.groupby(["ticker", "quarter_end"], group_keys=False)
            .apply(_elegir, include_groups=False).reset_index())
    df["fuente"] = FUENTE
    return df[SILVER_COLS].sort_values(["ticker", "quarter_end"]).reset_index(drop=True)


def silver_desde_legacy(path: Path = LEGACY) -> pd.DataFrame:
    """Respaldo sin token: data/fundamentals.parquet trae RATIOS, no montos (salvo
    fund_revenue_abs / fund_net_abs si se generó con la versión nueva). Se
    conservan los ratios en columnas _legacy para que golden los use."""
    d = pd.read_parquet(path)
    d["quarter_end"] = pd.to_datetime(d["quarter_end"])
    out = pd.DataFrame({
        "ticker": d["ticker"].astype(str),
        "quarter_end": d["quarter_end"],
        "ingresos": d.get("fund_revenue_abs", np.nan),
        "utilidad_bruta": np.nan,
        "utilidad_operacion": np.nan,
        "utilidad_neta": d.get("fund_net_abs", np.nan),
        "upa": np.nan,
        "origen": "legacy_parquet",
        "reexpresado": False,
        "formato": "corporativo",
        "primera_descarga": pd.NaT,
        "fuente": f"{FUENTE} (via data/fundamentals.parquet)",
    })
    for c in ("fund_gross_margin", "fund_oper_margin", "fund_net_margin",
              "fund_revenue_yoy", "fund_earnings_yoy"):
        out[f"{c}_legacy"] = d[c].values
    return out.sort_values(["ticker", "quarter_end"]).reset_index(drop=True)
