"""Limpieza de EPS reportados (`data/eps/reported_eps.json`) con disponibilidad segura.

Reglas fijas (recon 2026-11-09):
  1. duplicados (ticker, fecha de anuncio): se deja el último.
  2. anuncio en sábado/domingo: se mueve al siguiente día hábil (`weekend_moved`).
  3. `surprise == 0` exacto: probable "sin consenso" → NaN (`surprise_zero_nan`).
  4. |surprise| > 5: se marca `extreme`, no se borra.
`available_from` = siguiente día hábil tras `announced`: la hora de anuncio varía y los
resultados MX salen sobre todo tras el cierre, así que se usa d+1 conservador.
Los feriados BMV no se modelan (solo lunes a viernes).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS_JSON = Path(__file__).parent.parent.parent / "data" / "eps" / "reported_eps.json"
EPS_CLEAN = Path(__file__).parent.parent.parent / "data" / "eps" / "eps_clean.parquet"
EXTREME_SURPRISE = 5.0
COLUMNS = ["ticker", "announced", "reported", "surprise", "available_from", "flags"]
RULES = ["weekend_moved", "surprise_zero_nan", "extreme"]


def _siguiente_habil(fechas: pd.Series, *, estricto: bool) -> pd.Series:
    """Siguiente lunes-viernes. `estricto`: siempre posterior; si no, la misma fecha si ya es hábil."""
    d = fechas.to_numpy(dtype="datetime64[D]")
    if estricto:
        d = d + np.timedelta64(1, "D")
    return pd.Series(np.busday_offset(d, 0, roll="forward").astype("datetime64[ns]"), index=fechas.index)


def clean_eps(raw: dict | pd.DataFrame) -> pd.DataFrame:
    """dict ticker→lista de eventos, o DataFrame con `ticker`, `announced`, `reported`, `surprise`."""
    if isinstance(raw, dict):
        filas = [dict(ev, ticker=t) for t, evs in raw.items() for ev in (evs or [])]
        df = pd.DataFrame(filas, columns=["ticker", "announced", "reported", "surprise"])
    else:
        df = raw[["ticker", "announced", "reported", "surprise"]].copy()
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["announced"] = pd.to_datetime(df["announced"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    df["reported"] = pd.to_numeric(df["reported"], errors="coerce")
    df["surprise"] = pd.to_numeric(df["surprise"], errors="coerce")

    df = df.drop_duplicates(["ticker", "announced"], keep="last").reset_index(drop=True)

    fin_semana = df["announced"].dt.dayofweek >= 5
    df.loc[fin_semana, "announced"] = _siguiente_habil(df.loc[fin_semana, "announced"], estricto=False)

    cero = df["surprise"] == 0
    df.loc[cero, "surprise"] = np.nan
    extremo = df["surprise"].abs() > EXTREME_SURPRISE

    marcas = pd.DataFrame({"weekend_moved": fin_semana, "surprise_zero_nan": cero, "extreme": extremo})
    df["flags"] = marcas.apply(lambda r: ",".join(k for k in RULES if r[k]), axis=1)
    df["available_from"] = _siguiente_habil(df["announced"], estricto=True)
    return df[COLUMNS].sort_values(["announced", "ticker"]).reset_index(drop=True)


def flag_counts(clean: pd.DataFrame, n_input: int) -> dict[str, int]:
    """Filas que toca cada regla (duplicados = filas de entrada menos filas de salida)."""
    out = {"input_rows": int(n_input), "duplicates_removed": int(n_input - len(clean))}
    for r in RULES:
        out[r] = int(clean["flags"].str.contains(r).sum())
    out["output_rows"] = int(len(clean))
    return out


def build_eps_clean(src: Path = EPS_JSON, dst: Path = EPS_CLEAN) -> tuple[pd.DataFrame, dict[str, int]]:
    """Lee el JSON, limpia, guarda el parquet y devuelve (tabla, conteo por regla)."""
    raw = json.loads(Path(src).read_text())
    n_in = sum(len(v or []) for v in raw.values())
    clean = clean_eps(raw)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(dst, index=False)
    return clean, flag_counts(clean, n_in)
