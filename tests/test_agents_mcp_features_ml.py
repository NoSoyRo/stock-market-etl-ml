"""Pruebas del agente Features/ML: sin LLM, red ni escrituras productivas."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pytest
import yaml

pytest.importorskip("claude_agent_sdk")

from src.agents_mcp.domains import features_ml as fm
from src.agents_mcp.domains.features_ml import (
    MODEL_IDS,
    SYSTEM_PROMPT,
    create_ml_agent,
    create_ml_tools,
)
from src.agents_mcp.domains.ml_service import MLService
from src.ml import trial_stats
from src.ml.trials import count_trials

ROOT = Path(__file__).resolve().parents[1]


def _write_catalog(root: Path) -> None:
    target = root / "src" / "agents_mcp" / "domains"
    target.mkdir(parents=True)
    catalog = {
        "version": 1,
        "modelos": [
            {
                "id": model_id,
                "nombre": model_id,
                "estado": "investigacion",
                "evidencia": "sin_skill",
                "resumen": "test",
                "fuentes": [],
                "metricas": {},
                "limites": [],
            }
            for model_id in MODEL_IDS
        ],
    }
    (target / "ml_catalog.yaml").write_text(
        yaml.safe_dump(catalog, allow_unicode=True),
        encoding="utf-8",
    )


def _write_trials(root: Path) -> tuple[Path, dict]:
    path = root / "data" / "eval" / "trial_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"alpha": 1, "h": 5}
    rows = [
        {
            "ts": "2026-11-01T00:00:00+00:00",
            "family": "fam_a",
            "hash": trial_stats.config_hash(cfg),
            "config": cfg,
            "result": {"name": "fam_a:modelo", "metrics": {}},
        },
        {
            "ts": "2026-11-02T00:00:00+00:00",
            "family": "fam_a",
            "hash": trial_stats.config_hash({"alpha": 2}),
            "config": {"alpha": 2},
            "result": {"name": "fam_a:modelo2", "metrics": {}},
        },
        {
            "ts": "2026-11-03T00:00:00+00:00",
            "family": "fam_b",
            "hash": trial_stats.config_hash({"beta": 1}),
            "config": {"beta": 1},
            "result": {"name": "fam_b:modelo", "metrics": {}},
        },
    ]
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path, cfg


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write_catalog(tmp_path)
    _write_trials(tmp_path)
    signals = tmp_path / "data" / "signals" / "history"
    signals.mkdir(parents=True)
    signal = {
        "date": "2026-11-01",
        "timestamp_utc": "2026-11-01T20:00:00+00:00",
        "regime": "?",
        "bets": [{"ticker": "WALMEX", "conviction": 0.6}],
        "top10_ranking": [{"ticker": "WALMEX", "rank": 1}],
        "hash": "abc123",
    }
    (signals / "2026-11-01.json").write_text(json.dumps(signal), encoding="utf-8")
    index = {
        "entries": [
            {"date": "2026-01-02", "n_bets": 1, "bets": ["A"]},
            {"date": "2026-11-20", "n_bets": 1, "bets": ["B"]},
        ],
        "last_updated": "2026-11-20T20:00:00+00:00",
    }
    (signals / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return tmp_path


def _tools(root: Path, *, service=None):
    return {t.name: t for t in create_ml_tools(root=root, service=service)}


def _call(registry, name: str, args: dict | None = None) -> dict:
    reply = asyncio.run(registry[name].handler(args or {}))
    assert reply["content"][0]["type"] == "text"
    payload = json.loads(reply["content"][0]["text"])
    for key in ("ok", "fuentes", "evidencia", "n_trials_global"):
        assert key in payload
    return payload


def test_tool_inventory_and_json_contract(root):
    registry = _tools(root)
    assert set(registry) == {
        "schema_features_ml",
        "registro_experimentos",
        "veredicto_modelo",
        "senales_recientes",
        "estado_track_record",
        "calcular_dsr",
        "plan_experimento",
        "pronostico_precio",
        "recomendar_portafolio",
    }
    payload = _call(registry, "schema_features_ml")
    assert payload["ok"]
    assert len(payload["datos"]["modelos"]) == 10


def test_registro_experimentos_matches_trial_helpers(root):
    registry = _tools(root)
    payload = _call(registry, "registro_experimentos", {"familia": "fam_a", "limite": 2})
    trial_path = root / "data" / "eval" / "trial_log.jsonl"
    assert payload["n_trials_global"] == count_trials(path=str(trial_path))
    assert payload["datos"]["n_por_familia"]["fam_a"] == trial_stats.n_trials("fam_a", trial_path)
    assert payload["datos"]["n_coincidencias"] == 2


def test_veredicto_modelo_unknown_and_broken_metric(root):
    svc = MLService(root)
    unknown = svc.veredicto_modelo("no_existe")
    assert not unknown["ok"]
    assert unknown["error_tipo"] == "modelo_desconocido"

    catalog = yaml.safe_load(svc.catalog_path.read_text(encoding="utf-8"))
    catalog["modelos"][0]["metricas"] = {
        "rota": {"archivo": "data/eval/inexistente.json", "ruta": "x.y"}
    }
    svc.catalog_path.write_text(yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")
    result = svc.veredicto_modelo("supermodel")
    assert result["ok"]
    assert result["datos"]["metricas"]["rota"] is None
    assert result["advertencias"]


def test_real_catalog_has_expected_models_and_metric_routes_resolve():
    svc = MLService(ROOT)
    schema = svc.schema()
    assert schema["ok"]
    assert {m["id"] for m in schema["datos"]["modelos"]} == set(MODEL_IDS)
    for model_id in MODEL_IDS:
        result = svc.veredicto_modelo(model_id)
        assert result["ok"], model_id
        assert not any(
            ("No se pudo resolver" in w or "No existe" in w)
            for w in result["advertencias"]
        ), (model_id, result["advertencias"])


def test_signals_cutoff_staleness_and_missing_date(root):
    svc = MLService(root)
    now = datetime(2026, 11, 6, 18, tzinfo=timezone.utc)
    old = svc.senales_recientes(
        fecha="2026-11-01",
        corte="2026-11-06T18:00:00Z",
        now=now,
    )
    assert old["ok"] and old["datos"]["stale"] is True

    future = svc.senales_recientes(
        corte="2026-11-07T00:00:00Z",
        now=now,
    )
    assert future["error_tipo"] == "corte_futuro"

    missing = svc.senales_recientes(
        fecha="2026-10-31",
        corte="2026-11-06T18:00:00Z",
        now=now,
    )
    assert missing["error_tipo"] == "archivo_ausente"


def test_track_record_counts_closed_horizon(root):
    svc = MLService(root)
    result = svc.estado_track_record(
        now=datetime(2026, 11, 24, 12, tzinfo=timezone.utc)
    )
    assert result["ok"]
    assert result["datos"]["n_signales"] == 2
    assert result["datos"]["n_horizonte_cumplido"] == 1
    assert result["datos"]["n_pendientes"] == 1


def test_dsr_uses_trial_penalty_and_more_trials_reduce_dsr(root):
    svc = MLService(root)
    low_n = svc.calcular_dsr(1.0, 3.0, n_trials=2)
    high_n = svc.calcular_dsr(1.0, 3.0, n_trials=100)
    assert low_n["ok"] and high_n["ok"]
    assert high_n["datos"]["sr0_anual"] > low_n["datos"]["sr0_anual"]
    assert high_n["datos"]["dsr"] < low_n["datos"]["dsr"]


def test_plan_experimento_detects_hash_and_does_not_write(root):
    svc = MLService(root)
    _, cfg = _write_trials(root)
    before = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }
    result = svc.plan_experimento(
        "hipótesis preregistrada",
        "fam_a",
        cfg,
        "DSR > 0.95",
    )
    after = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert result["ok"]
    assert result["datos"]["ya_existe"] is True
    assert before == after


def test_legacy_handlers_return_json(root):
    registry = _tools(root)
    portfolio = _call(registry, "recomendar_portafolio")
    assert portfolio["ok"] is False
    assert portfolio["evidencia"] == "demo_no_vigente"
    assert hasattr(registry["pronostico_precio"], "handler")
    assert hasattr(registry["recomendar_portafolio"], "handler")


def test_pronostico_timeout_is_controlled(root, monkeypatch):
    svc = MLService(root)

    def slow(*args, **kwargs):
        time.sleep(0.10)
        return svc._base(evidencia="sin_skill")

    monkeypatch.setattr(svc, "pronostico_precio", slow)
    monkeypatch.setattr(fm, "TIMEOUT_PRONOSTICO_S", 0.01)
    registry = _tools(root, service=svc)
    result = _call(registry, "pronostico_precio", {"ticker": "WALMEX"})
    assert not result["ok"]
    assert result["error_tipo"] == "timeout"
    assert result["evidencia"] == "sin_skill"


def test_agent_limits_prompt_and_orchestrator_compatibility(root):
    agent = create_ml_agent(root)
    options = agent.build_options()
    assert agent.name == "features-ml"
    assert options.tools == []
    assert options.max_turns == fm.MAX_TURNS
    assert options.max_budget_usd == fm.MAX_BUDGET_USD
    assert set(t.name for t in agent.tools) == set(_tools(root))

    prompt = SYSTEM_PROMPT.lower()
    for phrase in (
        "n global de trials",
        "ic por fecha",
        "open t+1",
        "p-hacking",
        "producción",
        "compra/venta",
        "stale",
        "evaluacion",
        "lab",
        "noticias",
    ):
        assert phrase in prompt

    from src.agents_mcp.orchestrator import build_options

    orch = build_options()
    assert orch is not None
