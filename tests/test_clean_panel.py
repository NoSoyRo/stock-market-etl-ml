"""Tests de M3c · `build_panel(use_mask=...)` con un panel sintético (sin red ni datos reales)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.backtest_topk as bt  # noqa: E402
from src.data.quality import COLUMNS  # noqa: E402

N_TICKERS = 12
N_DAYS = 420
BAD_OPEN0 = ("T0", 100)      # Open = 0 en el día 100
BAD_JUMP = ("T1", 200)       # Open ≈ 0 (salto de ~1e9 %) en el día 200, marcado por la máscara


def _raw(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=N_DAYS)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.015, N_DAYS)))
    open_ = close * (1 + rng.normal(0, 0.003, N_DAYS))
    return pd.DataFrame(
        {"Open": open_, "High": np.maximum(open_, close) * 1.004,
         "Low": np.minimum(open_, close) * 0.996, "Close": close,
         "Volume": rng.integers(1_000_000, 2_000_000, N_DAYS).astype(float)},
        index=idx,
    )


def _universe() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    raws = {f"T{i}": _raw(i) for i in range(N_TICKERS)}
    t, d = BAD_OPEN0
    raws[t].iloc[d, raws[t].columns.get_loc("Open")] = 0.0
    t, d = BAD_JUMP
    raws[t].iloc[d, raws[t].columns.get_loc("Open")] = raws[t]["Close"].iloc[d] * 1e-9
    return raws, {k: f"{k}.MX" for k in raws}


def _mask(raws: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Máscara sintética: todo válido salvo el salto (BAD_JUMP). El Open=0 NO está marcado."""
    parts = []
    for name, raw in raws.items():
        m = pd.DataFrame(False, index=raw.index, columns=COLUMNS)
        m["valid"] = True
        parts.append(m.rename_axis("date").reset_index().assign(ticker=f"{name}.MX"))
    mask = pd.concat(parts, ignore_index=True)
    t, d = BAD_JUMP
    sel = (mask["ticker"] == f"{t}.MX") & (mask["date"] == raws[t].index[d])
    mask.loc[sel, ["suspect_jump", "valid"]] = [True, False]
    return mask


@pytest.fixture()
def setup(monkeypatch, tmp_path):
    raws, tickers = _universe()
    macro = {n: _raw(100 + i) for i, n in enumerate(("USDMXN", "NAFTRAC", "WTI", "SP500"))}
    monkeypatch.setattr(bt, "store_get", lambda n: raws.get(n, macro.get(n)))
    monkeypatch.setattr(bt, "MACRO_TICKERS", list(macro))
    # FEATURE_COLS crece con el tiempo (p. ej. columnas de tasas que este panel sintético no
    # tiene): el test solo necesita unas pocas features reales para que `dropna` funcione.
    monkeypatch.setattr(bt, "FEATURE_COLS", ["ret_5d", "ret_10d", "rsi_14"])
    # Fuentes de tasas y dividendos aisladas de los datos reales del repo (vacías por defecto).
    monkeypatch.setattr(bt, "build_rates_panel", lambda: pd.DataFrame())
    monkeypatch.setattr(bt, "DIVIDENDS_FILE", tmp_path / "no_hay_dividendos.json")
    return raws, tickers, _mask(raws)


def test_sin_mascara_se_cuelan(setup):
    _, tickers, _ = setup
    df = bt.build_panel(tickers)
    fr = df["fwd_ret"]
    assert (fr.abs() > 1e6).any()                       # salto ~1e9 %
    assert (~np.isfinite(fr)).any() or (fr.abs() > 100).sum() >= 2


def test_default_es_false_y_no_agrega_nada(setup):
    _, tickers, _ = setup
    a = bt.build_panel(tickers)
    b = bt.build_panel(tickers, use_mask=False)
    pd.testing.assert_frame_equal(a, b)
    assert not a.attrs


def test_golden_comportamiento_previo(setup):
    """Cifras del `build_panel` previo (sin use_mask) sobre este panel sintético."""
    _, tickers, _ = setup
    df = bt.build_panel(tickers)
    assert df.shape == GOLDEN["shape"]
    assert float(df["y"].sum()) == GOLDEN["y_sum"]
    assert float(df.loc[df["fwd_ret"].abs() < 100, "fwd_ret"].sum()) == pytest.approx(GOLDEN["fwd_sum"], rel=1e-9)
    assert float(df["uni_n"].sum()) == GOLDEN["uni_n_sum"]


def test_con_mascara_desaparecen_y_el_resto_queda_igual(setup):
    _, tickers, mask = setup
    sin = bt.build_panel(tickers)
    con = bt.build_panel(tickers, use_mask=True, mask=mask)
    assert np.isfinite(con["fwd_ret"]).all() and con["fwd_ret"].abs().max() <= 100
    t0, d0 = BAD_OPEN0
    assert not ((con["ticker"] == t0) & (con.index == setup[0][t0].index[d0])).any()
    assert len(con) < len(sin)

    # Las filas conservadas coinciden en lo propio de cada ticker (fwd_ret y features sin ranking).
    key = lambda x: x.reset_index(names="date").set_index(["date", "ticker"])
    a, b = key(sin), key(con)
    comun = b.index
    for col in ("fwd_ret", "y", "ret_5d"):
        np.testing.assert_allclose(a.loc[comun, col].to_numpy(), b[col].to_numpy())
    # Y fila completa idéntica en fechas no tocadas por filas eliminadas.
    tocadas = set(a.index.difference(b.index).get_level_values(0))
    libres = [i for i in comun if i[0] not in tocadas]
    assert len(libres) > 1000
    pd.testing.assert_frame_equal(a.loc[libres], b.loc[libres])


def test_reporte_de_descartes_por_regla(setup):
    raws, tickers, mask = setup
    out: list = []
    bt.build_panel(tickers, use_mask=True, mask=mask, drops_out=out)
    drops = out[0]
    assert {"ticker", "date", "mask_invalid", "open_nonpos", "fwd_ret_outlier"} <= set(drops.columns)
    assert drops["mask_invalid"].sum() == 1
    assert drops["open_nonpos"].sum() == 1
    assert drops["fwd_ret_outlier"].sum() == 2          # día previo a cada Open basura
    assert set(drops["ticker"]) == {BAD_OPEN0[0], BAD_JUMP[0]}

    cut = raws["T0"].index[150]
    s = bt.summarize_mask_drops(drops, cut)
    assert s["train"]["rows_total"] == 2 and s["test"]["rows_total"] == 2
    assert s["train"]["open_nonpos"]["rows"] == 1 and s["test"]["open_nonpos"]["rows"] == 0
    assert s["test"]["mask_invalid"]["rows"] == 1 and s["test"]["fwd_ret_outlier"]["rows"] == 1
    assert s["train"]["tickers_total"] == 1 and s["test"]["tickers_total"] == 1


def test_mascara_se_carga_sola_si_no_se_pasa(setup, monkeypatch):
    _, tickers, mask = setup
    monkeypatch.setattr(bt, "load_quality_mask", lambda: mask)
    out: list = []
    bt.build_panel(tickers, use_mask=True, drops_out=out)
    assert out[0]["mask_invalid"].sum() == 1


# Recalculado con el set reducido de FEATURE_COLS del fixture (el real creció con columnas de tasas);
# 62 columnas = 54 + 8 auxiliares (tasas y dividendos, en NaN aquí).
GOLDEN = {"shape": (4800, 62), "y_sum": 2311.0, "fwd_sum": -534.8894144052009, "uni_n_sum": 57600.0}


# ─── Columnas de tasas y dividendos en el panel ───────────────────────────────

def test_panel_trae_las_columnas_auxiliares_sin_perder_filas(setup, monkeypatch, tmp_path):
    raws, tickers, _ = setup
    idx = next(iter(raws.values())).index
    rates = pd.DataFrame(
        {"tiie28": 7.0, "tiie91": 7.1, "cete28": 6.5, "cete91": 6.6, "cete182": 6.7,
         "cete364": 6.9, "tasa_obj": 7.0}, index=idx[-30:])
    monkeypatch.setattr(bt, "build_rates_panel", lambda: rates)
    name = next(iter(tickers))
    ex = str(idx[100].date())
    f = tmp_path / "dividends.json"
    f.write_text(json.dumps({name: [{"pago_date": ex, "ex_date": ex, "pago": 1.0,
                                     "tipo": "DIVIDENDO EN EFECTIVO", "divisa": "MXN",
                                     "source": "historico"}]}))
    monkeypatch.setattr(bt, "DIVIDENDS_FILE", f)

    con = bt.build_panel(tickers)                          # aux_features=True por defecto
    sin = bt.build_panel(tickers, aux_features=False)

    assert all(c in con.columns for c in bt.AUX_FEATURE_COLS)
    assert len(con) == len(sin)                            # las auxiliares no recortan el panel
    assert con["tiie28_level"].notna().any()               # tasas con datos locales
    assert con.loc[con["ticker"] == name, "dias_a_ex_date"].notna().any()   # dividendos del ticker
    assert con.loc[con["ticker"] != name, "dias_a_ex_date"].isna().all()    # sin historial: NaN


def test_sin_fuentes_locales_las_auxiliares_quedan_en_nan(setup):
    _, tickers, _ = setup                                  # fixture: tasas y dividendos vacíos
    df = bt.build_panel(tickers)
    assert all(c in df.columns for c in bt.AUX_FEATURE_COLS)
    assert df[bt.AUX_FEATURE_COLS].isna().all().all()
    assert len(df) > 0                                     # y el panel sigue existiendo


def test_model_feature_cols_excluye_tasas_y_dividendos(monkeypatch):
    todas = [f"f{i}" for i in range(5)] + bt.AUX_FEATURE_COLS
    monkeypatch.setattr(bt, "FEATURE_COLS", todas)
    assert bt.model_feature_cols(todas) == [f"f{i}" for i in range(5)]
    assert bt.model_feature_cols(["f0", "tiie28_level"]) == ["f0"]
