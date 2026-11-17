"""GOLDEN de Fundamentales PIT (Equipo Rojo).

Lo que consumen dashboard, agente y modelos. Cada fila trae available_date
(fin de trimestre + rezago fijo) y nadie la debe usar antes de esa fecha.
Calcula márgenes, YoY por FECHA (no por posición) y trae la UPA.

Sale en data/fundamentals/golden/fundamentales_pit.parquet y
data/fundamentals/golden/dividendos_eventos.parquet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from productos.fundamentales_pit.pit import BASE

VERSION_PRODUCTO = "1.0.0"
DIVIDENDOS_MANUAL = BASE / "dividendos_manual.csv"

GOLDEN_COLS = [
    "ticker", "quarter_end", "available_date",
    "fund_revenue_abs", "fund_gross_abs", "fund_oper_abs", "fund_net_abs", "fund_eps",
    "fund_gross_margin", "fund_oper_margin", "fund_net_margin",
    "fund_revenue_yoy", "fund_earnings_yoy",
    "origen", "reexpresado", "formato", "fuente", "version_producto", "generado_en",
]

DIVIDENDOS_COLS = ["ticker", "fecha_anuncio", "fecha_corte", "fecha_pago",
                   "monto_por_accion", "moneda", "fuente", "available_date"]


def _lag_dias() -> int:
    from src.api.fundamentals import PUBLICATION_LAG_DAYS
    return int(PUBLICATION_LAG_DAYS)


def _yoy_por_fecha(df: pd.DataFrame, col: str) -> pd.Series:
    """Crecimiento contra el MISMO trimestre del año anterior, buscado por fecha.
    Si ese trimestre no existe, NaN (nunca otro trimestre)."""
    clave = df["quarter_end"].dt.year * 4 + df["quarter_end"].dt.quarter
    previo = pd.Series(df[col].values, index=pd.MultiIndex.from_arrays([df["ticker"], clave]))
    previo = previo[~previo.index.duplicated()]
    idx = pd.MultiIndex.from_arrays([df["ticker"], clave - 4])
    antes = previo.reindex(idx).to_numpy()
    actual = df[col].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        yoy = np.where(np.isfinite(antes) & (antes != 0), actual / antes - 1, np.nan)
    return pd.Series(yoy, index=df.index)


def _yoy_legacy_valido(df: pd.DataFrame) -> pd.Series:
    """El YoY legacy se calculó con pct_change(4). Solo es correcto si las 4 filas
    previas son los 4 trimestres inmediatos anteriores (sin huecos)."""
    clave = df["quarter_end"].dt.year * 4 + df["quarter_end"].dt.quarter
    ok = pd.Series(True, index=df.index)
    for k in range(1, 5):
        ok &= (clave - df.groupby("ticker")["quarter_end"].transform(
            lambda s, k=k: (s.dt.year * 4 + s.dt.quarter).shift(k))) == k
    return ok


def construir_golden(silver: pd.DataFrame, generado_en: pd.Timestamp | None = None) -> pd.DataFrame:
    s = silver.sort_values(["ticker", "quarter_end"]).reset_index(drop=True).copy()
    s["quarter_end"] = pd.to_datetime(s["quarter_end"])
    g = pd.DataFrame({"ticker": s["ticker"], "quarter_end": s["quarter_end"]})
    g["available_date"] = s["quarter_end"] + pd.Timedelta(days=_lag_dias())
    g["fund_revenue_abs"] = s["ingresos"].astype(float)
    g["fund_gross_abs"] = s["utilidad_bruta"].astype(float)
    g["fund_oper_abs"] = s["utilidad_operacion"].astype(float)
    g["fund_net_abs"] = s["utilidad_neta"].astype(float)
    g["fund_eps"] = s["upa"].astype(float)

    ingresos = g["fund_revenue_abs"].where(g["fund_revenue_abs"] != 0)
    for col, monto in (("fund_gross_margin", "fund_gross_abs"),
                       ("fund_oper_margin", "fund_oper_abs"),
                       ("fund_net_margin", "fund_net_abs")):
        calc = g[monto] / ingresos
        legacy = s.get(f"{col}_legacy")
        g[col] = calc if legacy is None else calc.where(calc.notna(), legacy)

    yoy_rev = _yoy_por_fecha(g, "fund_revenue_abs")
    yoy_net = _yoy_por_fecha(g, "fund_net_abs")
    if "fund_revenue_yoy_legacy" in s:
        valido = _yoy_legacy_valido(g)
        yoy_rev = yoy_rev.where(yoy_rev.notna(), s["fund_revenue_yoy_legacy"].where(valido))
        yoy_net = yoy_net.where(yoy_net.notna(), s["fund_earnings_yoy_legacy"].where(valido))
    g["fund_revenue_yoy"] = yoy_rev
    g["fund_earnings_yoy"] = yoy_net

    # Mismo recorte que la ingesta original para ratios (no para montos).
    for c in ("fund_gross_margin", "fund_oper_margin", "fund_net_margin",
              "fund_revenue_yoy", "fund_earnings_yoy"):
        g[c] = g[c].replace([np.inf, -np.inf], np.nan).clip(-5, 5)

    g["origen"] = s["origen"]
    g["reexpresado"] = s["reexpresado"].fillna(False).astype(bool)
    g["formato"] = s["formato"]
    g["fuente"] = s["fuente"]
    g["version_producto"] = VERSION_PRODUCTO
    g["generado_en"] = generado_en or pd.Timestamp.now().normalize()
    return g[GOLDEN_COLS]


def construir_dividendos(path: Path = DIVIDENDOS_MANUAL) -> pd.DataFrame:
    """Dividendos con sus tres fechas. Fuente pendiente (ver README): mientras
    tanto se lee un CSV capturado a mano con las columnas de la plantilla."""
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in DIVIDENDOS_COLS})
    d = pd.read_csv(path)
    for c in ("fecha_anuncio", "fecha_corte", "fecha_pago"):
        d[c] = pd.to_datetime(d[c])
    mal = d[(d["fecha_anuncio"] > d["fecha_corte"]) | (d["fecha_corte"] > d["fecha_pago"])]
    if not mal.empty:
        raise ValueError(f"Fechas fuera de orden (anuncio <= corte <= pago): {mal.to_dict('records')}")
    d["ticker"] = d["ticker"].str.upper()
    d["available_date"] = d["fecha_anuncio"]   # se sabe el día que se anuncia
    return d[DIVIDENDOS_COLS]
