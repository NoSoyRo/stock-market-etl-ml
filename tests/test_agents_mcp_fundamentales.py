"""query_fundamentales del agente de dominio (src/agents_mcp) sobre el golden PIT."""
from __future__ import annotations

import asyncio
import json
import sys
import types

import pandas as pd
import pytest

# src/agent/__init__ arrastra un módulo que importa fcntl (no existe en Windows).
if sys.platform == "win32" and "fcntl" not in sys.modules:
    _f = types.ModuleType("fcntl")
    _f.LOCK_EX = _f.LOCK_UN = _f.LOCK_SH = _f.LOCK_NB = 0
    _f.flock = lambda *a, **k: None
    sys.modules["fcntl"] = _f

pytest.importorskip("claude_agent_sdk")

from src.agents_mcp.domains import fundamentales as dom  # noqa: E402
import src.agent.fundamentales as agente  # noqa: E402


def _golden() -> pd.DataFrame:
    q = pd.to_datetime(["2026-03-31", "2026-06-30", "2026-09-30"])
    g = pd.DataFrame({"ticker": "WALMEX", "quarter_end": q})
    g["available_date"] = g["quarter_end"] + pd.Timedelta(days=70)   # 06-09, 09-08, 12-09
    for c in ("fund_gross_margin", "fund_oper_margin", "fund_net_margin",
              "fund_revenue_yoy", "fund_earnings_yoy", "fund_eps",
              "fund_revenue_abs", "fund_net_abs"):
        g[c] = [0.1, 0.2, 0.3]
    g["origen"] = "api"
    return g


def _llamar(**args) -> dict:
    herramienta = getattr(dom.query_fundamentales, "handler", dom.query_fundamentales)
    r = asyncio.run(herramienta(args))
    return json.loads(r["content"][0]["text"])


def test_la_herramienta_del_profe_respeta_la_fecha_de_corte(monkeypatch):
    monkeypatch.setattr(agente, "cargar_golden", lambda path=None: _golden())
    r = _llamar(ticker="walmex", fecha_consulta="2026-09-01")
    assert r["ok"] and r["ticker"] == "WALMEX"
    visibles = [str(t["quarter_end"])[:10] for t in r["trimestres"]]
    # al 2026-09-01 solo se conocía 2026T1 (disponible 06-09); 2026T2 sale el 09-08.
    assert visibles == ["2026-03-31"]


def test_sin_fecha_usa_hoy_y_un_ticker_desconocido_no_se_inventa(monkeypatch):
    monkeypatch.setattr(agente, "cargar_golden", lambda path=None: _golden())
    r = _llamar(ticker="WALMEX", fecha_consulta="")
    assert r["ok"] and len(r["trimestres"]) >= 1
    r = _llamar(ticker="NOEXISTE", fecha_consulta="")
    assert not r["ok"] and r["error_tipo"] == "ticker_desconocido"


def test_sin_golden_cae_al_archivo_de_siempre(monkeypatch, tmp_path):
    def sin_golden(path=None):
        raise FileNotFoundError("no hay golden")
    monkeypatch.setattr(agente, "cargar_golden", sin_golden)
    monkeypatch.setattr(dom, "FUNDAMENTALS_PARQUET", tmp_path / "no_existe.parquet")
    monkeypatch.delenv("DATABURSATIL_API_TOKEN", raising=False)
    monkeypatch.delenv("DATABURSATIL_TOKEN", raising=False)
    herramienta = getattr(dom.query_fundamentales, "handler", dom.query_fundamentales)
    texto = asyncio.run(herramienta({"ticker": "WALMEX"}))["content"][0]["text"]
    assert "DATABURSATIL_API_TOKEN" in texto and "No tengo" in texto   # respaldo, no un error técnico
