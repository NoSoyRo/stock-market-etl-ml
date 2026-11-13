"""Registro de ensayos para descontar suerte en el DSR (adaptador sobre `trial_stats`).

Hay UNA sola bitácora: ``data/eval/trial_log.jsonl`` (formato de `trial_stats.log_trial`:
``family``, ``hash``, ``config``, ``result``). Esta API (`log_trial(name, ...)` idempotente y
`count_trials`) la conserva para los estudios M-series: el ``name`` y las métricas viajan en
``result``, y la familia es lo que está antes de ``:`` en el nombre (``m9_bakeoff:ridge`` →
``m9_bakeoff``). Así `trial_stats.n_trials(familia)` (N por familia) y `count_trials()` (N global)
leen el mismo archivo.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.ml import trial_stats as ts

DEFAULT_PATH = "data/eval/trial_log.jsonl"
config_hash = ts.config_hash


def _family(name: str) -> str:
    return name.split(":")[0]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _name(row: dict) -> str:
    """Nombre del ensayo: el de `result.name` (M-series) o la familia (bitácora de core-satellite)."""
    res = row.get("result") or {}
    return res.get("name") or row.get("family", "")


def log_trial(name: str, config: dict, metrics: dict, n_obs: int, *,
              path: str = DEFAULT_PATH) -> bool:
    """Anexa un ensayo. Idempotente: (name, hash del config) ya registrado => no duplica.

    Devuelve True si escribió, False si ya existía.
    """
    p = Path(path)
    h = ts.config_hash(config)
    if any(r.get("hash") == h and _name(r) == name for r in _read(p)):
        return False
    ts.log_trial(_family(name), config,
                 {"name": name, "metrics": metrics, "n_obs": int(n_obs)}, p)
    return True


def count_trials(tag: str | None = None, *, path: str = DEFAULT_PATH) -> int:
    """Nº GLOBAL de ensayos únicos (nombre, hash); con `tag`, solo los cuyo nombre lo contiene.

    Es el N conservador (todo se probó sobre la misma historia). Para N por familia usa
    `trial_stats.n_trials(familia, path)`.
    """
    seen = {(_name(r), r.get("hash")) for r in _read(Path(path))}
    if tag is not None:
        seen = {k for k in seen if tag in k[0]}
    return len(seen)
