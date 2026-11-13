"""
Triple-Barrier Labeling + Sample Uniqueness Weights.

Metodología de López de Prado ("Advances in Financial Machine Learning", Cap. 3-4).

Idea central:
  En vez de etiquetar "¿subió en H días?" (label fijo en el tiempo),
  etiquetamos como realmente operamos:

    Desde el precio de entrada en t, se ponen 3 barreras:
      - Superior (target):  entry * (1 + mult_target * vol)   → label +1
      - Inferior (stop):    entry * (1 - mult_stop   * vol)   → label  0 (perdió)
      - Vertical (tiempo):  t + horizon días                  → label según signo

    La primera barrera tocada define el resultado.

Esto produce labels que coinciden con la estrategia real (stops ATR + targets),
en vez de un target arbitrario que ignora cómo cerramos posiciones.

Uso:
    from src.ml.labeling import triple_barrier_labels

    labels = triple_barrier_labels(
        close=raw["Close"], high=raw["High"], low=raw["Low"],
        vol=atr_pct_series, horizon=5,
        mult_target=1.5, mult_stop=1.5,
    )
    # labels["label"]:   1 = ganó (tocó target o cerró arriba)
    #                    0 = perdió (tocó stop o cerró abajo)
    # labels["touch"]:   "target" | "stop" | "time"
    # labels["ret"]:     retorno realizado del trade
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close:       pd.Series,
    high:        pd.Series,
    low:         pd.Series,
    vol:         pd.Series,
    horizon:     int   = 5,
    mult_target: float = 1.5,
    mult_stop:   float = 1.5,
    open_:       pd.Series | None = None,
    entry:       str   = "close",
    tie_break:   str   = "target",
) -> pd.DataFrame:
    """
    Calcula labels triple-barrier para cada fecha de la serie.

    Args:
        close:       Serie de precios de cierre (índice temporal).
        high:        Serie de máximos diarios.
        low:         Serie de mínimos diarios.
        vol:         Volatilidad relativa por fila (ej. atr_pct = ATR/precio).
                     Define el ancho de las barreras.
        horizon:     Barras hasta la barrera vertical.
        mult_target: Múltiplo de vol para la barrera superior (target).
        mult_stop:   Múltiplo de vol para la barrera inferior (stop).
        open_:       Serie de aperturas. Obligatoria con entry="next_open" o
                     tie_break="gap_aware".
        entry:       Regla de entrada.
                       "close"     → Close[i] (default, comportamiento histórico).
                       "next_open" → Open[i+1]; el horizonte cuenta desde esa
                                     barra (vertical en i+1+horizon, igual que
                                     `compute_target(use_open_entry=True)`).
        tie_break:   Qué pasa si una misma vela toca ambas barreras.
                       "target"    → gana el target (optimista, default histórico).
                       "stop"      → gana el stop (conservador).
                       "gap_aware" → si el Open ya abre más allá de una barrera,
                                     el llenado es al Open (no al precio de la
                                     barrera); si no hay gap y se tocan ambas,
                                     gana el stop.

    Returns:
        DataFrame alineado con `close` con columnas:
          - label  (int):   1 = trade ganador, 0 = perdedor
          - touch  (str):   "target" | "stop" | "time"
          - ret    (float): retorno realizado (%) del trade
          - bars   (int):   barras desde la señal hasta el cierre del trade
    """
    if entry not in ("close", "next_open"):
        raise ValueError("entry debe ser 'close' o 'next_open'")
    if tie_break not in ("target", "stop", "gap_aware"):
        raise ValueError("tie_break debe ser 'target', 'stop' o 'gap_aware'")
    if (entry == "next_open" or tie_break == "gap_aware") and open_ is None:
        raise ValueError("open_ es obligatorio con entry='next_open' o tie_break='gap_aware'")

    n = len(close)
    idx = close.index

    c = close.values
    h = high.values
    l = low.values
    v = vol.values
    o = open_.reindex(idx).values if open_ is not None else None
    shift = 1 if entry == "next_open" else 0

    labels = np.full(n, np.nan)
    touches = np.empty(n, dtype=object)
    rets    = np.full(n, np.nan)
    bars    = np.full(n, np.nan)

    for i in range(n):
        if i + 1 >= n or np.isnan(v[i]) or v[i] <= 0:
            continue

        # Entrada: cierre de la señal, o apertura de la barra siguiente.
        entry_px = o[i + 1] if entry == "next_open" else c[i]
        if np.isnan(entry_px) or entry_px <= 0:
            continue
        up = entry_px * (1 + mult_target * v[i])
        dn = entry_px * (1 - mult_stop   * v[i])

        end = min(i + shift + horizon, n - 1)

        touched = None
        exit_price = c[end]
        nbars = end - i

        # Recorrer la ventana buscando la primera barrera tocada
        for j in range(i + 1, end + 1):
            hit_up = h[j] >= up
            hit_dn = l[j] <= dn
            if tie_break == "gap_aware":
                # Gap: el Open ya abrió fuera de una barrera → llenado al Open.
                if o[j] >= up:
                    touched, exit_price, nbars = "target", o[j], j - i
                    break
                if o[j] <= dn:
                    touched, exit_price, nbars = "stop", o[j], j - i
                    break
                if hit_up and hit_dn:        # ambiguo sin gap: el peor caso
                    touched, exit_price, nbars = "stop", dn, j - i
                    break
            if tie_break == "stop":
                if hit_dn:
                    touched, exit_price, nbars = "stop", dn, j - i
                    break
                if hit_up:
                    touched, exit_price, nbars = "target", up, j - i
                    break
            else:
                if hit_up:
                    touched, exit_price, nbars = "target", up, j - i
                    break
                if hit_dn:
                    touched, exit_price, nbars = "stop", dn, j - i
                    break

        if touched is None:
            # Barrera vertical: clasificar por signo del retorno
            touched = "time"
            exit_price = c[end]

        ret = (exit_price - entry_px) / entry_px * 100
        labels[i]  = 1 if ret > 0 else 0
        touches[i] = touched
        rets[i]    = ret
        bars[i]    = nbars

    return pd.DataFrame({
        "label": labels,
        "touch": touches,
        "ret":   rets,
        "bars":  bars,
    }, index=idx)


def sample_uniqueness_weights(
    bars: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """
    Pesos de unicidad (López de Prado, Cap. 4).

    Observaciones cuyos labels se solapan en el tiempo aportan información
    redundante. Este peso reduce la influencia de observaciones con mucho
    solapamiento, evitando que el modelo sobre-aprenda períodos densos.

    Aproximación práctica: el peso es inversamente proporcional a cuántas
    otras observaciones comparten el mismo período de holding.

    Args:
        bars:    Serie con el número de barras que duró cada trade.
        horizon: Horizonte máximo (para normalizar).

    Returns:
        Serie de pesos normalizados (media ≈ 1.0).
    """
    # Concurrencia: cuántos trades están "abiertos" en cada barra
    n = len(bars)
    concurrency = np.ones(n)
    b = bars.fillna(horizon).values.astype(int)

    counts = np.zeros(n + horizon + 1)
    for i in range(n):
        span = max(1, b[i])
        for j in range(i, min(i + span, n)):
            counts[j] += 1

    weights = np.ones(n)
    for i in range(n):
        span = max(1, b[i])
        # Peso = promedio de 1/concurrencia sobre el período de holding
        w = np.mean([1.0 / counts[j] for j in range(i, min(i + span, n)) if counts[j] > 0])
        weights[i] = w if w > 0 else 1.0

    s = pd.Series(weights, index=bars.index)
    # Normalizar para que la media sea 1
    return s / s.mean()


def barrier_stats(labels: pd.DataFrame) -> dict:
    """Resumen de la distribución de labels y barreras tocadas."""
    valid = labels.dropna(subset=["label"])
    if len(valid) == 0:
        return {}
    touch_dist = valid["touch"].value_counts(normalize=True).to_dict()
    return {
        "n":             len(valid),
        "win_rate":      round(valid["label"].mean(), 4),
        "avg_ret":       round(valid["ret"].mean(), 4),
        "avg_bars":      round(valid["bars"].mean(), 2),
        "pct_target":    round(touch_dist.get("target", 0), 4),
        "pct_stop":      round(touch_dist.get("stop", 0), 4),
        "pct_time":      round(touch_dist.get("time", 0), 4),
    }
