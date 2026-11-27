"""Pruebas del agente Features/ML: sin LLM, red ni escrituras productivas."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import time
import types

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
    assert options.strict_mcp_config is True
    assert options.setting_sources == []
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


def _set_evidence(root: Path, model_id: str, value: str) -> None:
    path = MLService(root).catalog_path
    catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    for item in catalog["modelos"]:
        if item["id"] == model_id:
            item["evidencia"] = value
    path.write_text(yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")


def _all_calls() -> dict[str, dict]:
    return {
        "schema_features_ml": {},
        "registro_experimentos": {},
        "veredicto_modelo": {"modelo": "supermodel"},
        "senales_recientes": {},
        "estado_track_record": {},
        "calcular_dsr": {"sharpe_anual": 1.0, "anios": 3.0},
        "plan_experimento": {
            "hipotesis": "h",
            "familia": "fam_a",
            "config": {"alpha": 9},
            "criterio_exito": "DSR > 0.95",
        },
        "pronostico_precio": {"ticker": "WALMEX"},
        "recomendar_portafolio": {},
    }


@pytest.fixture
def fake_forecast(monkeypatch):
    """Evita el cómputo real de scripts/forecast_horizons.py."""
    import pandas as pd

    import scripts.forecast_horizons as fh

    def run(ticker, model, label, days):
        return {
            "cutoff_date": "2026-09-10",
            "price0": 100.0,
            "daily": pd.DataFrame({"price_model": [101.0]}),
            "mae_model": 1.0,
            "mae_naive": 2.0,
        }

    monkeypatch.setattr(fh, "run_sin_ancla", run)


def test_every_tool_returns_full_contract(root, fake_forecast):
    registry = _tools(root)
    calls = _all_calls()
    assert set(calls) == set(registry)
    for name, args in calls.items():
        payload = _call(registry, name, args)
        for key in ("as_of", "advertencias", "datos"):
            assert key in payload, (name, key)


def test_evidence_labels_come_from_catalog(root):
    _set_evidence(root, "supermodel", "senal_debil_no_explotable")
    registry = _tools(root)
    verdict = _call(registry, "veredicto_modelo", {"modelo": "supermodel"})
    signals = _call(registry, "senales_recientes")
    assert verdict["evidencia"] == "senal_debil_no_explotable"
    assert signals["evidencia"] == "senal_debil_no_explotable"


def test_signals_without_catalog_are_labelled_not_evaluated(root):
    MLService(root).catalog_path.unlink()
    signals = _call(_tools(root), "senales_recientes")
    assert signals["ok"]
    assert signals["evidencia"] == "no_evaluado"
    assert any("catálogo" in w for w in signals["advertencias"])


@pytest.mark.parametrize("content", ["modelos: [sin cerrar", "- 1\n- 2\n", "modelos: 5\n"])
def test_tools_never_raise_with_broken_catalog(root, fake_forecast, content):
    MLService(root).catalog_path.write_text(content, encoding="utf-8")
    registry = _tools(root)
    for name, args in _all_calls().items():
        _call(registry, name, args)
    for name, args in (
        ("schema_features_ml", {}),
        ("veredicto_modelo", {"modelo": "supermodel"}),
    ):
        assert _call(registry, name, args)["error_tipo"] == "archivo_ausente"


def test_tools_never_raise_with_broken_trial_log(root, fake_forecast):
    path = root / "data" / "eval" / "trial_log.jsonl"
    path.write_text("[1]\n", encoding="utf-8")
    registry = _tools(root)
    for name, args in _all_calls().items():
        payload = _call(registry, name, args)
        assert payload["n_trials_global"] is None, name


def test_dsr_without_trial_count_is_an_error(root):
    path = root / "data" / "eval" / "trial_log.jsonl"
    path.write_text("[1]\n", encoding="utf-8")
    result = _call(_tools(root), "calcular_dsr", {"sharpe_anual": 1.0, "anios": 3.0})
    assert result["error_tipo"] == "archivo_ausente"


def test_plan_experimento_includes_text_draft(root):
    result = MLService(root).plan_experimento(
        "la reversión mejora el IC", "fam_a", {"alpha": 9}, "DSR > 0.95"
    )
    draft = result["datos"]["borrador"]
    assert isinstance(draft, str)
    for piece in (
        "la reversión mejora el IC",
        "fam_a",
        result["datos"]["preregistro"]["config_hash"],
        '"alpha": 9',
        "DSR > 0.95",
    ):
        assert piece in draft


def test_track_record_empty_index_has_null_as_of(root):
    svc = MLService(root)
    (svc.signals_dir / "index.json").write_text('{"entries": []}', encoding="utf-8")
    result = svc.estado_track_record()
    assert result["ok"]
    assert result["as_of"] is None


def test_dsr_matches_trial_stats_deflated_sharpe(root):
    import numpy as np
    from scipy import stats

    returns = np.random.default_rng(7).normal(0.0008, 0.01, 756)
    expected = trial_stats.deflated_sharpe(returns, n_trials=50)
    result = MLService(root).calcular_dsr(
        returns.mean() / returns.std(ddof=1) * np.sqrt(252),
        len(returns) / 252,
        n_trials=50,
        skew=float(stats.skew(returns)),
        kurtosis=float(stats.kurtosis(returns, fisher=False)),
    )
    assert round(result["datos"]["dsr"], 3) == expected["dsr"]
    assert round(result["datos"]["sr0_anual"], 3) == expected["sr0_anual"]


def test_quant_senior_still_consumes_legacy_tools(fake_forecast, monkeypatch):
    # quant_senior importa `anthropic` al cargar, pero estas dos etapas no lo usan.
    if importlib.util.find_spec("anthropic") is None:
        monkeypatch.setitem(sys.modules, "anthropic", types.ModuleType("anthropic"))
        monkeypatch.delitem(sys.modules, "src.agents_mcp.quant_senior", raising=False)
    quant_senior = importlib.import_module("src.agents_mcp.quant_senior")

    forecast = json.loads(asyncio.run(quant_senior._gather_ml("WALMEX")))
    assert forecast["ok"] and forecast["evidencia"] == "sin_skill"
    portfolio = asyncio.run(quant_senior._gather_optimizacion("WALMEX"))
    assert "demo_no_vigente" in portfolio


def _real_catalog() -> dict[str, dict]:
    path = MLService(ROOT).catalog_path
    catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in catalog["modelos"]}


def test_real_catalog_every_model_has_resolved_metrics():
    svc = MLService(ROOT)
    for model_id in MODEL_IDS:
        metrics = svc.veredicto_modelo(model_id)["datos"]["metricas"]
        assert metrics, model_id
        assert all(v is not None for v in metrics.values()), (model_id, metrics)


def test_real_catalog_sources_cover_metric_files_and_exist():
    for model_id, item in _real_catalog().items():
        for spec in item["metricas"].values():
            assert spec["archivo"] in item["fuentes"], (model_id, spec["archivo"])
        for source in item["fuentes"]:
            assert (ROOT / source).exists(), (model_id, source)


def test_real_catalog_reviewed_verdicts():
    catalog = _real_catalog()
    assert catalog["supermodel"]["evidencia"] == "no_evaluado"
    assert catalog["reversion_5d"]["evidencia"] == "senal_debil_no_explotable"
    assert catalog["regimen_hmm"]["evidencia"] == "sin_skill"


def test_reversion_ic_route_points_to_open_entry_row():
    """El reporte guarda los IC en una lista: si cambia el orden, esto debe fallar."""
    specs = _real_catalog()["reversion_5d"]["metricas"]
    route = specs["ic_open_t1"]["ruta"]
    assert route.endswith(".mean")
    report = json.loads((ROOT / specs["ic_open_t1"]["archivo"]).read_text(encoding="utf-8"))
    row = report["ic"][int(route.split(".")[1])]
    assert (row["señal"], row["entrada"], row["subconjunto"]) == (
        "ret_5d", "open_close", "todo",
    )
    assert specs["t_open_t1"]["ruta"] == route.replace(".mean", ".t")


def test_dot_get_reads_list_positions_and_rejects_bad_ones():
    data = {"ic": [{"mean": 1}, {"mean": 2}]}
    assert MLService._dot_get(data, "ic.1.mean") == 2
    for route in ("ic.2.mean", "ic.-1.mean", "ic.x.mean"):
        with pytest.raises(KeyError):
            MLService._dot_get(data, route)
