"""BRONZE de Fundamentales PIT (Equipo Rojo).

Guarda cada respuesta de la API DataBursátil (/v2/financieros) TAL CUAL, en
data/fundamentals/bronze/<ticker>/<financieros>/<periodo>.json, con la fecha de
descarga y el código HTTP. Nunca el token. Sirve para rehacer silver sin volver
a llamar a la API.

Lo corre productos/fundamentales_pit/capas.py (python -m productos.fundamentales_pit.capas bronze).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from productos.fundamentales_pit.pit import BASE

BRONZE = BASE / "bronze"
API = "https://api.databursatil.com/v2/financieros"


def _mapa_emisoras() -> dict[str, str]:
    from src.data.bmv_catalog import BMV_EMISORA_FUNDAMENTALS
    return dict(BMV_EMISORA_FUNDAMENTALS)


def periodos(year_start: int, year_end: int, estricto: bool = True) -> list[str]:
    """estricto=True: los 4 trimestres de TODOS los años (el doble de llamadas que
    la ingesta original, pero cada trimestre trae su cifra original)."""
    anios = range(year_start, year_end + 1) if estricto else range(year_start + 1, year_end + 1, 2)
    return [f"{q}T_{y}" for y in anios for q in (1, 2, 3, 4)]


def _es_error_de_red(archivo: Path) -> bool:
    """True si el JSON guardado no es una respuesta de la API (status None: conexión
    caída, timeout) o no se puede leer. Esos archivos se vuelven a pedir."""
    try:
        return json.loads(archivo.read_text(encoding="utf-8")).get("status_code") is None
    except Exception:
        return True


# Una respuesta que no es 200 de un trimestre reciente puede ser "todavía no se
# publica": la API contesta 400 ("periodo desconocido") hasta que sale el reporte.
VENTANA_REINTENTO_DIAS = 365


def _fin_de_trimestre(periodo: str) -> pd.Timestamp:
    """'3T_2026' -> 2026-09-30."""
    q, anio = periodo.split("T_")
    return pd.Timestamp(year=int(anio), month=int(q) * 3, day=1) + pd.offsets.MonthEnd(0)


def _reintentar(archivo: Path, periodo: str, hoy: pd.Timestamp | None = None) -> bool:
    """True si hay que volver a pedir un archivo de bronze que ya existe:
    - guardó un error de red (status None) o no se puede leer;
    - guardó una respuesta que no es 200 de un trimestre que cerró hace menos de
      VENTANA_REINTENTO_DIAS (o que todavía no cierra): puede que ya se haya publicado.
    Un 400 de un trimestre viejo (por ejemplo, los bancos que la API no ofrece) se
    queda como está, para no repetir esas llamadas en cada corrida."""
    if _es_error_de_red(archivo):
        return True
    status = json.loads(archivo.read_text(encoding="utf-8")).get("status_code")
    if status == 200:
        return False
    hoy = pd.Timestamp(hoy) if hoy is not None else pd.Timestamp.now().normalize()
    return _fin_de_trimestre(periodo) >= hoy - pd.Timedelta(days=VENTANA_REINTENTO_DIAS)


def bajar_bronze(token: str, tickers: list[str] | None = None, year_start: int = 2017,
                 year_end: int | None = None, estricto: bool = True,
                 financieros: tuple[str, ...] = ("resultado_trimestre",),
                 forzar: bool = False, pausa: float = 0.2, http=None, dest: Path = BRONZE) -> dict:
    """Guarda cada respuesta de la API tal cual. Idempotente: si el archivo ya
    existe no vuelve a llamar (salvo forzar=True), excepto si guardó un error de red
    o un "periodo desconocido" de un trimestre reciente (ver _reintentar)."""
    import requests
    http = http or requests
    year_end = year_end or datetime.now().year
    mapa = _mapa_emisoras()
    tickers = tickers or sorted(mapa)
    resumen = {"llamadas": 0, "omitidas_ya_existian": 0, "errores": 0, "por_ticker": {}}
    for ticker in tickers:
        emisora = mapa.get(ticker)
        if not emisora:
            resumen["por_ticker"][ticker] = "sin mapeo en bmv_catalog"
            continue
        ok = 0
        for fin in financieros:
            for per in periodos(year_start, year_end, estricto):
                archivo = dest / ticker / fin / f"{per}.json"
                if archivo.exists() and not forzar and not _reintentar(archivo, per):
                    resumen["omitidas_ya_existian"] += 1
                    continue
                try:
                    r = http.get(API, params={"token": token, "emisora": emisora,
                                              "periodo": per, "financieros": fin}, timeout=30)
                    status = r.status_code
                    try:
                        payload = r.json()
                    except ValueError:
                        payload = {"_texto_no_json": r.text[:2000]}
                except Exception as exc:  # red caída, timeout
                    status, payload = None, {"_error": str(exc)}
                    resumen["errores"] += 1
                resumen["llamadas"] += 1
                archivo.parent.mkdir(parents=True, exist_ok=True)
                archivo.write_text(json.dumps({
                    "ticker": ticker, "emisora": emisora, "periodo": per, "financieros": fin,
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "status_code": status, "payload": payload,
                }, ensure_ascii=False), encoding="utf-8")
                ok += status == 200
                if pausa:
                    time.sleep(pausa)
        resumen["por_ticker"][ticker] = f"{ok} respuestas 200"
    return resumen

