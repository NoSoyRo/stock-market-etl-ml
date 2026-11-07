"""Registro de ensayos (append-only) para descontar suerte en el DSR."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = "data/eval/trials.jsonl"


def config_hash(config: dict) -> str:
    """Hash estable (sha256 corto) de un config; independiente del orden de llaves."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def log_trial(name: str, config: dict, metrics: dict, n_obs: int, *,
              path: str = DEFAULT_PATH) -> bool:
    """Anexa un ensayo. Idempotente: (name, hash del config) ya registrado => no duplica.

    Devuelve True si escribió, False si ya existía.
    """
    p = Path(path)
    h = config_hash(config)
    if any(r["name"] == name and r["config_hash"] == h for r in _read(p)):
        return False
    rec = {
        "name": name,
        "config_hash": h,
        "config": config,
        "metrics": metrics,
        "n_obs": int(n_obs),
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return True


def count_trials(tag: str | None = None, *, path: str = DEFAULT_PATH) -> int:
    """Nº de ensayos únicos; con `tag`, solo los cuyo nombre lo contiene."""
    recs = _read(Path(path))
    if tag is not None:
        recs = [r for r in recs if tag in r["name"]]
    return len(recs)
