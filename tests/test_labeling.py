"""Pruebas de la triple barrera: entrada, reglas de empate y gaps (M2a)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.labeling import triple_barrier_labels


def _legacy(close, high, low, vol, horizon=5, mult_target=1.5, mult_stop=1.5):
    """Copia literal de la implementación previa a M2a (referencia 'golden')."""
    n = len(close)
    c, h, l, v = close.values, high.values, low.values, vol.values
    labels = np.full(n, np.nan)
    touches = np.empty(n, dtype=object)
    rets = np.full(n, np.nan)
    bars = np.full(n, np.nan)
    for i in range(n):
        if i + 1 >= n or np.isnan(v[i]) or v[i] <= 0:
            continue
        entry = c[i]
        up = entry * (1 + mult_target * v[i])
        dn = entry * (1 - mult_stop * v[i])
        end = min(i + horizon, n - 1)
        touched, exit_price, nbars = None, c[end], end - i
        for j in range(i + 1, end + 1):
            if h[j] >= up:
                touched, exit_price, nbars = "target", up, j - i
                break
            if l[j] <= dn:
                touched, exit_price, nbars = "stop", dn, j - i
                break
        if touched is None:
            touched, exit_price = "time", c[end]
        ret = (exit_price - entry) / entry * 100
        labels[i], touches[i], rets[i], bars[i] = (1 if ret > 0 else 0), touched, ret, nbars
    return pd.DataFrame({"label": labels, "touch": touches, "ret": rets, "bars": bars},
                        index=close.index)


def _serie(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    spread = np.abs(rng.normal(0, 0.01, n)) + 0.001
    s = lambda x: pd.Series(x, index=idx)  # noqa: E731
    return s(close), s(close * (1 + spread)), s(close * (1 - spread)), s(open_), \
        s(np.full(n, 0.02))


def _mini(opens, highs, lows, closes, vol=0.02):
    """Serie corta a mano: la fila 0 es la fecha de la señal."""
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    f = lambda x: pd.Series(x, index=idx, dtype=float)  # noqa: E731
    return f(closes), f(highs), f(lows), f(opens), f([vol] * len(closes))


# ─── Golden: los defaults no cambian ─────────────────────────────────────────

def test_defaults_match_legacy_implementation():
    c, h, l, o, v = _serie()
    nuevo = triple_barrier_labels(c, h, l, v, horizon=5)
    pd.testing.assert_frame_equal(nuevo, _legacy(c, h, l, v, horizon=5))


def test_open_is_ignored_with_close_entry_and_target_tiebreak():
    c, h, l, o, v = _serie()
    sin_open = triple_barrier_labels(c, h, l, v)
    con_open = triple_barrier_labels(c, h, l, v, open_=o)
    pd.testing.assert_frame_equal(sin_open, con_open)


# ─── Entrada ─────────────────────────────────────────────────────────────────

def test_next_open_uses_open_of_next_bar():
    # señal en 0; Open[1]=110 es la entrada; ninguna barrera se toca.
    c, h, l, o, v = _mini(
        opens=[100, 110, 110, 110, 110, 110, 110],
        highs=[100, 111, 111, 111, 111, 111, 111],
        lows=[100, 109, 109, 109, 109, 109, 109],
        closes=[100, 110, 110, 110, 110, 110, 121],
    )
    out = triple_barrier_labels(c, h, l, v, horizon=5, entry="next_open", open_=o)
    r = out.iloc[0]
    # salida vertical en i+1+horizon = 6 → Close[6]=121 vs entrada 110
    assert r["touch"] == "time"
    assert r["ret"] == pytest.approx((121 - 110) / 110 * 100)
    assert r["bars"] == 6


def test_next_open_barriers_are_built_from_open_not_close():
    # entrada 100 → target 103 (vol 0.02 × 1.5). Con entrada en el Close (50) no se tocaría.
    c, h, l, o, v = _mini(
        opens=[50, 100, 100, 100],
        highs=[50, 100.5, 103.5, 100],
        lows=[50, 99.5, 99.5, 100],
        closes=[50, 100, 100, 100],
    )
    out = triple_barrier_labels(c, h, l, v, horizon=3, entry="next_open", open_=o)
    r = out.iloc[0]
    assert r["touch"] == "target"
    assert r["ret"] == pytest.approx(3.0)
    assert r["bars"] == 2


def test_next_open_requires_open():
    c, h, l, o, v = _serie(30)
    with pytest.raises(ValueError):
        triple_barrier_labels(c, h, l, v, entry="next_open")


def test_next_open_last_signal_is_nan():
    c, h, l, o, v = _serie(30)
    out = triple_barrier_labels(c, h, l, v, entry="next_open", open_=o)
    assert np.isnan(out["label"].iloc[-1])


# ─── Empate en una vela ──────────────────────────────────────────────────────

def _vela_ambigua():
    # entrada Close[0]=100; vela 1 toca target (103) y stop (97); abre dentro.
    return _mini(
        opens=[100, 100, 100, 100],
        highs=[100, 104, 100, 100],
        lows=[100, 96, 100, 100],
        closes=[100, 100, 100, 100],
    )


def test_tiebreak_target_wins_by_default():
    c, h, l, o, v = _vela_ambigua()
    r = triple_barrier_labels(c, h, l, v, horizon=3, open_=o).iloc[0]
    assert r["touch"] == "target" and r["ret"] == pytest.approx(3.0)


def test_tiebreak_stop_wins():
    c, h, l, o, v = _vela_ambigua()
    r = triple_barrier_labels(c, h, l, v, horizon=3, open_=o, tie_break="stop").iloc[0]
    assert r["touch"] == "stop" and r["ret"] == pytest.approx(-3.0)
    assert r["label"] == 0


def test_tiebreak_gap_aware_stop_wins_when_open_inside():
    c, h, l, o, v = _vela_ambigua()
    r = triple_barrier_labels(c, h, l, v, horizon=3, open_=o, tie_break="gap_aware").iloc[0]
    assert r["touch"] == "stop" and r["ret"] == pytest.approx(-3.0)


def test_tiebreak_invalid_raises():
    c, h, l, o, v = _vela_ambigua()
    with pytest.raises(ValueError):
        triple_barrier_labels(c, h, l, v, tie_break="otro")


def test_gap_aware_requires_open():
    c, h, l, o, v = _vela_ambigua()
    with pytest.raises(ValueError):
        triple_barrier_labels(c, h, l, v, tie_break="gap_aware")


# ─── Gaps: el llenado es al Open, no a la barrera ────────────────────────────

def test_gap_below_stop_fills_at_open():
    # stop = 97; el día 1 abre en 90 (gap) → pérdida de 10%, no de 3%.
    c, h, l, o, v = _mini(
        opens=[100, 90, 90, 90],
        highs=[100, 92, 90, 90],
        lows=[100, 88, 90, 90],
        closes=[100, 90, 90, 90],
    )
    legacy = triple_barrier_labels(c, h, l, v, horizon=3, open_=o).iloc[0]
    gap = triple_barrier_labels(c, h, l, v, horizon=3, open_=o, tie_break="gap_aware").iloc[0]
    assert legacy["ret"] == pytest.approx(-3.0)
    assert gap["touch"] == "stop" and gap["ret"] == pytest.approx(-10.0)


def test_gap_above_target_fills_at_open():
    # target = 103; el día 1 abre en 108 → ganancia de 8%, no de 3%.
    c, h, l, o, v = _mini(
        opens=[100, 108, 108, 108],
        highs=[100, 110, 108, 108],
        lows=[100, 107, 108, 108],
        closes=[100, 108, 108, 108],
    )
    gap = triple_barrier_labels(c, h, l, v, horizon=3, open_=o, tie_break="gap_aware").iloc[0]
    assert gap["touch"] == "target" and gap["ret"] == pytest.approx(8.0)


def test_gap_aware_open_beyond_target_wins_even_if_candle_also_hits_stop():
    # abre en 105 (> target 103) y luego cae a 90: el llenado fue al Open, gana el target.
    c, h, l, o, v = _mini(
        opens=[100, 105, 90, 90],
        highs=[100, 106, 90, 90],
        lows=[100, 90, 90, 90],
        closes=[100, 95, 90, 90],
    )
    gap = triple_barrier_labels(c, h, l, v, horizon=3, open_=o, tie_break="gap_aware").iloc[0]
    assert gap["touch"] == "target" and gap["ret"] == pytest.approx(5.0)
