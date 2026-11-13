"""Congela reglas para prueba prospectiva en data/eval/preregistros/<id>.json.

Cada JSON guarda: definición exacta, criterio de éxito, fecha de corte, hash del código, hash de
los datos usados (hasta el corte) y hash del documento de preregistro, más la potencia (semanas
necesarias). No evalúa nada. Si el archivo ya existe NO lo sobrescribe (congelado = inmutable).

Uso:
    .venv/bin/python scripts/freeze_preregistro.py            # congela las dos reglas
    .venv/bin/python scripts/freeze_preregistro.py --dry-run  # imprime sin escribir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORTE = "2026-11-12"
PRIMERA_SENAL = "2026-11-13"
OUT_DIR = ROOT / "data" / "eval" / "preregistros"
DOC = "avances/2026-11-12_N2_preregistro.md"
MAP_JSON = ROOT / "data" / "microstructure" / "hechos_ticker_map.json"
OOS_JSON = ROOT / "data" / "eval" / "oos_v2.json"

CODE_FLUJO = [
    "src/ml/flow_features.py", "src/data/hechos_map.py",
    "src/ml/shared/evaluators/cross_section.py", "src/ml/shared/evaluators/costs.py",
    "src/ml/shared/evaluators/portfolio_rules.py",
]
CODE_V1 = [
    "scripts/run_oos_v2.py", "scripts/backtest_topk.py", "scripts/run_m1_report.py",
    "scripts/run_rotation_study.py", "src/ml/ensemble.py", "src/ml/features.py",
    "src/ml/rank_models.py", "src/ml/cs_targets.py", "src/ml/kb_enhance.py",
    "src/ml/shared/evaluators/cross_section.py", "src/ml/shared/evaluators/costs.py",
    "src/ml/shared/evaluators/portfolio_rules.py", "src/ml/trials.py",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_files(rel_paths: list[str]) -> dict:
    """Hash por archivo y hash combinado (orden alfabético). Falla si falta alguno."""
    per = {p: sha256_file(ROOT / p) for p in sorted(rel_paths)}
    comb = hashlib.sha256("".join(f"{k}:{v}" for k, v in per.items()).encode()).hexdigest()
    return {"combinado": comb, "archivos": per}


def hash_ohlcv_hasta(tickers: list[str], corte: str) -> dict:
    """Hash de OHLCV por ticker **recortado al corte**: sigue siendo reproducible cuando el
    parquet crezca con días nuevos; si alguien revisa el pasado, el hash cambia."""
    from src.data.store import get

    per = {}
    for t in sorted(tickers):
        df = get(t)
        if df is None or df.empty:
            per[t] = None
            continue
        df = df[df.index <= pd.Timestamp(corte)]
        h = pd.util.hash_pandas_object(df.round(6), index=True).values.tobytes()
        per[t] = hashlib.sha256(h).hexdigest()
    comb = hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()
    return {"corte": corte, "combinado": comb, "por_ticker": per}


def hash_hechos_hasta(corte: str) -> dict:
    d = ROOT / "data" / "microstructure" / "hechos"
    per = {p.stem: sha256_file(p) for p in sorted(d.glob("*.parquet")) if p.stem <= corte}
    comb = hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()
    return {"corte": corte, "n_archivos": len(per), "combinado": comb}


def semanas_necesarias(sigma: float) -> dict:
    """Semanas para detectar un IC de 0.03 y 0.01 dada la std semanal `sigma` del IC.

    `se_igual_ic`: SE = IC (t≈1), n = (σ/IC)².  `t2`: |t| ≥ 2, n = (2σ/IC)².
    """
    out = {"sigma_semanal": round(sigma, 4)}
    for ic in (0.03, 0.01):
        out[f"ic_{ic}"] = {
            "se_igual_ic": int(round((sigma / ic) ** 2)),
            "t2": int(round((2 * sigma / ic) ** 2)),
        }
    return out


def sigma_oos() -> float:
    res = json.loads(OOS_JSON.read_text())["results"]
    return float(sum(v["ic"]["std"] for v in res.values()) / len(res))


def universo_flujo() -> list[str]:
    """Tickers con confianza alta en el mapa ∩ panel BMV (nombres de archivo sin `.MX`)."""
    from scripts.backtest_topk import load_universe_tickers

    panel = {v.removesuffix(".MX") for v in load_universe_tickers("bmv").values()}
    mapa = json.loads(MAP_JSON.read_text())["mapa"]
    return sorted(v["ticker"] for v in mapa.values()
                  if v["confianza"] == "alta" and v["ticker"] in panel)


def regla_flujo(sigma: float) -> dict:
    uni = universo_flujo()
    return {
        "id": "flujo_residual",
        "descripcion": "Imbalance de casas 5d residualizado por retorno 5d (idea 3.1); signo esperado negativo.",
        "corte": CORTE, "primera_senal": PRIMERA_SENAL,
        "documento": {"ruta": DOC, "sha256": sha256_file(ROOT / DOC)},
        "definicion": {
            "universo": uni, "n_universo": len(uni),
            "mapa": "data/microstructure/hechos_ticker_map.json (confianza alta)",
            "imb_5d": "media de flow_imbalance diario en 5 días hábiles, >=4 de 5 presentes; sin rellenar",
            "ret_5d": "Close(t)/Close(t-5)-1",
            "resid": "residuo por fecha de OLS imb_5d ~ ret_5d, min 15 nombres",
            "fecha_senal": "último día hábil de cada semana",
            "objetivo": "retorno 5d, entrada Open t+1 -> Close t+6",
            "ic": "Spearman por fecha; t via ic_summary (bloques de 4 semanas)",
            "cartera": {"largo": "5 menores resid", "freq": "semanal", "pesos": "iguales",
                        "costo_por_lado": 0.003, "benchmark": "EW del universo, misma frecuencia"},
            "semilla": 42,
        },
        "criterio": {
            "signo_esperado": "negativo",
            "exito": {"semanas_min": 26, "ic_medio_max": -0.04, "t_max": -2.0, "neto_vs_ew": ">"},
            "abandono": {"semanas_min": 26, "ic_medio_min": -0.02},
            "inconcluso": "entre -0.04 y -0.02 o t > -2: seguir hasta 52 semanas con t <= -2.4",
            "miradas_intermedias": "a 13 semanas solo fuga: |IC| > 0.10 = sospecha",
            "ensayos": "log_trial familia n2_flujo, solo al evaluar",
        },
        "potencia_semanas": semanas_necesarias(sigma),
        "hash_codigo": hash_files(CODE_FLUJO),
        "hash_datos": {"hechos": hash_hechos_hasta(CORTE),
                       "ohlcv": hash_ohlcv_hasta([f"{t}.MX" for t in uni], CORTE),
                       "mapa_json": sha256_file(MAP_JSON)},
    }


def regla_v1(sigma: float) -> dict:
    from scripts.backtest_topk import load_universe_tickers

    uni = sorted(v for v in load_universe_tickers("bmv").values())
    return {
        "id": "reglas_v1",
        "descripcion": "E-frozen, E-refit, R-frozen, R-refit y REV de oos_v2, sin reajustes.",
        "corte": CORTE, "primera_senal": PRIMERA_SENAL,
        "documento": {"ruta": "avances/2026-11-09_oos_v2_preregistro.md",
                      "sha256": sha256_file(ROOT / "avances/2026-11-09_oos_v2_preregistro.md"),
                      "preregistro_n2": DOC, "preregistro_n2_sha256": sha256_file(ROOT / DOC)},
        "definicion": {
            "reglas": ["E-frozen", "E-refit", "R-frozen", "R-refit", "REV"],
            "definidas_en": "avances/2026-11-09_oos_v2_preregistro.md (carteras, panel, costos, semilla 42)",
            "script_referencia": "scripts/run_oos_v2.py",
            "universo_panel": uni,
            "entrenamiento": {"frozen": "train < 2023-03-09", "refit": "train hasta 2026-06-09 (sin reentrenar)"},
            "ventana_evaluacion": f"fechas de señal >= {PRIMERA_SENAL}",
        },
        "criterio": {
            "E_R": "S1 IC medio > 0; S2 neto@0.30% > EW neto. Falla grosera: IC <= 0 o exceso < -0.0725%/sem",
            "REV": "S1 IC > 0; S2' bruto > EW bruto. Falla grosera: IC <= 0",
            "alarma_fuga": "IC medio > 0.10",
        },
        "reevaluacion": {"cada_semanas": 13, "primera": "señales 2026-11-13 -> 2027-02-12",
                         "ventana": "acumulada desde la primera señal",
                         "ensayos": "5 filas log_trial por evaluación, familia oos_v2_prosp"},
        "potencia_semanas": semanas_necesarias(sigma),
        "hash_codigo": hash_files(CODE_V1),
        "hash_datos": {"ohlcv": hash_ohlcv_hasta(uni, CORTE)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sigma = sigma_oos()
    for builder in (regla_flujo, regla_v1):
        reg = builder(sigma)
        path = OUT_DIR / f"{reg['id']}.json"
        if path.exists():
            print(f"[{reg['id']}] YA EXISTE ({path.relative_to(ROOT)}): congelado, no se sobrescribe")
            continue
        if args.dry_run:
            print(f"[{reg['id']}] dry-run: escribiría {path.relative_to(ROOT)}; "
                  f"código={reg['hash_codigo']['combinado'][:12]} "
                  f"potencia={reg['potencia_semanas']}")
            continue
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{reg['id']}] congelado → {path.relative_to(ROOT)} "
              f"(código {reg['hash_codigo']['combinado'][:12]}, potencia {reg['potencia_semanas']})")


if __name__ == "__main__":
    main()
