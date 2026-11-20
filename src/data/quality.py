"""
Máscara de calidad de datos y filtro de liquidez (M3a).

Marca filas basura de un OHLCV diario (volumen 0, precio congelado, OHLC
inválido, saltos sospechosos, huecos) de forma **causal**: la bandera de la
fecha t solo usa datos hasta t, así que agregar datos futuros nunca cambia
una bandera pasada.

Tipos de serie (`kind`):
    equity    — acciones/ETFs/FIBRAs: se aplican todas las reglas.
    benchmark — como equity, pero `valid` NO excluye por volumen/congelado
                (p. ej. NAFTRAC.MX: se marca, no se descarta).
    macro     — FX, índices, futuros: sin reglas de volumen ni liquidez.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MARKET_DIR = Path(__file__).parent.parent.parent / "data" / "market"
QUALITY_DIR = Path(__file__).parent.parent.parent / "data" / "quality"
MASK_PATH = QUALITY_DIR / "quality_mask.parquet"

FLAGS = ["zero_volume", "flat_run", "bad_ohlc", "suspect_jump", "long_gap", "illiquid"]
COLUMNS = FLAGS + ["valid"]

JUMP_THRESHOLD = 0.25        # |ret| diario sospechoso
FLAT_RUN_LEN = 3             # cierres iguales consecutivos
MAX_GAP_DAYS = 7             # días calendario entre filas
LIQ_WINDOW = 60              # mediana móvil de Close×Volume
LIQ_MIN_PERIODS = 20

# Umbral de liquidez (mediana 60d de Close×Volume, en moneda local) por universo.
LIQ_THRESHOLD = {"mx": 1_000_000.0, "us": 1_000_000.0}

# Series macro/FX/índices (sin volumen confiable) y benchmarks.
MACRO_TICKERS = {"MXN=X", "CL=F", "GSPC", "^GSPC"}
BENCHMARK_TICKERS = {"NAFTRAC.MX", "NAFTRACISHRS.MX"}


def kind_for(ticker: str) -> str:
    """Tipo de serie según el ticker."""
    if ticker in MACRO_TICKERS:
        return "macro"
    if ticker in BENCHMARK_TICKERS:
        return "benchmark"
    return "equity"


def universe_for(ticker: str) -> str:
    """'mx' para tickers .MX, 'us' para el resto."""
    return "mx" if ticker.endswith(".MX") else "us"


def compute_quality_mask(
    df: pd.DataFrame, *, kind: str = "equity", liq_threshold: float | None = None
) -> pd.DataFrame:
    """Banderas booleanas por fecha (mismo índice que `df`).

    `df`: índice de fechas ascendente y columnas Open/High/Low/Close/Volume.
    `liq_threshold`: umbral de `illiquid`; por defecto el del universo "us".
    """
    if kind not in ("equity", "benchmark", "macro"):
        raise ValueError(f"kind desconocido: {kind}")
    out = pd.DataFrame(False, index=df.index, columns=COLUMNS)
    if df.empty:
        return out

    o, h, l, c, v = (df[x].astype(float) for x in ("Open", "High", "Low", "Close", "Volume"))
    use_volume = kind != "macro"

    # OHLC inválido (aplica a todos): NaN, precio ≤ 0, High<Low, O/C fuera de [Low, High]
    prices = pd.concat([o, h, l, c], axis=1)
    out["bad_ohlc"] = (
        prices.isna().any(axis=1) | (prices <= 0).any(axis=1) | (h < l)
        | (o > h) | (o < l) | (c > h) | (c < l)
    )

    zero_vol = v.fillna(0) == 0
    if use_volume:
        out["zero_volume"] = zero_vol
        # ≥3 cierres iguales (día t cierra la racha de 3) con volumen 0 o High == Low
        same3 = (c == c.shift(1)) & (c == c.shift(2))
        out["flat_run"] = same3 & (zero_vol | (h == l))

    # Saltos sospechosos: |ret|>25% con volumen 0, o rebote opuesto >25% al día
    # siguiente. El rebote se marca en el día del rebote (t+1) para ser causal.
    ret = c.pct_change()
    big = ret.abs() > JUMP_THRESHOLD
    rebound = big & big.shift(1, fill_value=False) & (np.sign(ret) == -np.sign(ret.shift(1)))
    out["suspect_jump"] = (big & zero_vol & use_volume) | rebound

    # Huecos: fila posterior a > 7 días calendario sin datos
    idx = pd.DatetimeIndex(df.index)
    gap_days = pd.Series(idx, index=df.index).diff().dt.days
    out["long_gap"] = gap_days > MAX_GAP_DAYS

    # Liquidez: mediana móvil 60d de Close×Volume, desplazada 1 día (solo pasado)
    if use_volume:
        thr = LIQ_THRESHOLD["us"] if liq_threshold is None else liq_threshold
        dv = (c * v).rolling(LIQ_WINDOW, min_periods=LIQ_MIN_PERIODS).median().shift(1)
        out["illiquid"] = dv.isna() | (dv < thr)

    excluded = ["bad_ohlc", "suspect_jump", "long_gap"]
    if kind == "equity":
        excluded += ["zero_volume", "flat_run"]
    out["valid"] = ~out[excluded].any(axis=1)
    return out.astype(bool)


def load_quality_mask(path: Path | str = MASK_PATH) -> pd.DataFrame:
    """Carga la máscara larga (columnas: ticker, date, banderas, valid)."""
    return pd.read_parquet(path)


def apply_mask(
    df: pd.DataFrame, mask: pd.DataFrame, ticker: str, *, require_liquid: bool = False
) -> pd.DataFrame:
    """Filtra las filas de `df` (índice de fechas) que la máscara marca como no válidas.

    Fechas sin información en la máscara se conservan. Con `require_liquid`
    también se descartan las filas `illiquid`.
    """
    m = mask[mask["ticker"] == ticker].set_index("date")
    m = m[~m.index.duplicated(keep="last")]
    ok = m["valid"] & ~m["illiquid"] if require_liquid else m["valid"]
    ok = ok.reindex(pd.DatetimeIndex(df.index)).fillna(True).to_numpy(dtype=bool)
    return df[ok]


def mask_coverage(mask: pd.DataFrame, tickers: list[str], dates) -> dict:
    """Cobertura de la máscara sobre un conjunto de tickers y fechas (no modifica nada).

    Devuelve `missing_tickers` (ausentes de la máscara) y `frac_after_max`: por ticker
    presente, fracción de `dates` posteriores a su fecha máxima en la máscara
    (esas filas pasarían sin filtrar en `apply_mask`).
    """
    d = pd.DatetimeIndex(dates)
    max_by = mask.groupby("ticker")["date"].max()
    missing = [t for t in tickers if t not in max_by.index]
    frac = {t: float((d > max_by[t]).mean()) if len(d) else 0.0 for t in tickers if t in max_by.index}
    return {"missing_tickers": missing, "frac_after_max": frac}
