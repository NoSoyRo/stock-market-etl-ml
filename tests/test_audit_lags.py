"""Pruebas de los helpers de la auditoría de rezagos (M2b), con datos sintéticos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_lags import lead_lag_corr, merge_asof_same_day, ts_ic, verdict


def _serie(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(size=n), index=idx)


def test_feature_con_fuga_cae_con_lag():
    fwd = _serie()
    feat = fwd + 0.3 * _serie(seed=1)  # conoce el futuro
    a, b = ts_ic(feat, fwd, 0), ts_ic(feat, fwd, 1)
    assert a["ic"] > 0.8 and abs(b["ic"]) < 0.2
    assert verdict(a["ic"], b["ic"], a["se"]) == "sospechosa"


def test_feature_rezagada_es_ok():
    fwd = _serie()
    feat = fwd.shift(1)  # causal: dato viejo
    a, b = ts_ic(feat, fwd, 0), ts_ic(feat, fwd, 1)
    assert verdict(a["ic"], b["ic"], a["se"]) == "ok"


def test_lead_lag_detecta_adelanto():
    m = _serie()
    otro = m.shift(-1)  # otro_t = m_{t+1}: aparece en k=-1 (otro de ayer == m de hoy)
    c = lead_lag_corr(m, otro)
    assert c["t-1"] > 0.99 and abs(c["t"]) < 0.2


def test_merge_asof_ya_no_usa_el_dato_el_mismo_dia():
    # Tras el fix (allow_exact_matches=False) el dato publicado el día D se usa desde D+1.
    r = merge_asof_same_day(pd.Timestamp("2024-03-12"))
    assert r["mismo_dia"] is False
    assert r["primer_dia_con_dato"] > r["available_date"]
