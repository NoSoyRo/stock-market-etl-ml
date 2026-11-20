"""Tests de M9: etiquetas, grupos por fecha, early stopping causal, determinismo y pesos."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.rank_models import (
    AntiNoiseArm, RankerArm, RidgeArm, XsLgbmArm, auc_weights, blend_scores, date_groups,
    ic_by_tercile, inner_purged_split, quintile_labels, rank_features, shrunk_ic_weights,
    tercile_buckets, weights_path,
)

COLS = ["rn_a", "rn_b", "rn_c"]


def _panel(n_dates: int = 120, n_names: int = 12, seed: int = 0) -> pd.DataFrame:
    """Panel sintético: rn_a predice (débilmente) el retorno; rn_b, rn_c son ruido."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    d = np.repeat(dates, n_names)
    df = pd.DataFrame({"ticker": np.tile([f"T{i}" for i in range(n_names)], n_dates),
                       "a": rng.normal(size=len(d)), "b": rng.normal(size=len(d)),
                       "c": rng.normal(size=len(d))}, index=d)
    df["fwd_ret"] = 0.8 * df["a"] + rng.normal(size=len(d))
    df = rank_features(df, ["a", "b", "c"])
    df["pct_fwd"] = df.groupby(df.index)["fwd_ret"].rank(pct=True)
    return df


def test_quintile_labels_por_fecha():
    df = _panel(10, 10)
    y = quintile_labels(df.index, df["fwd_ret"].to_numpy())
    assert set(y) == {0, 1, 2, 3, 4}
    for _, g in pd.Series(y, index=df.index).groupby(level=0):
        assert sorted(g.value_counts().tolist()) == [2] * 5  # 10 nombres => 2 por quintil
    # el mejor retorno de la fecha lleva etiqueta 4
    s = pd.Series(df["fwd_ret"].to_numpy(), index=df.index)
    top = s.groupby(level=0).idxmax()
    first = df.index[0]
    assert y[(df.index == first) & (s.to_numpy() == s[first].max())][0] == 4
    assert len(top) == 10


def test_date_groups_suma_y_contiguidad():
    df = _panel(8, 5)
    g = date_groups(df.index)
    assert g.sum() == len(df) and (g == 5).all() and len(g) == 8
    with pytest.raises(ValueError):
        date_groups(df.index[::-1])


def test_ranker_agrupa_por_fecha_y_aprende():
    df = _panel(150, 12)
    tr, te = df.iloc[:1200], df.iloc[1200:]
    arm = RankerArm(COLS, "rank_xendcg", {"n_estimators": 60, "num_leaves": 7}).fit(tr)
    p = arm.predict(te)
    assert len(p) == len(te)
    from src.ml.shared.evaluators.cross_section import ic_by_date
    d = pd.DataFrame({"s": p, "r": te["fwd_ret"].to_numpy()}, index=te.index)
    assert ic_by_date(d, "s", "r", min_names=10).mean() > 0.2


def test_ranker_objetivo_invalido():
    with pytest.raises(ValueError):
        RankerArm(COLS, "otro")


def test_inner_split_purga_y_orden():
    df = _panel(200, 4)
    tr, va = inner_purged_split(df.index, val_frac=0.15, horizon=5, embargo=5)
    dtr, dva = df.index[tr], df.index[va]
    uniq = np.sort(df.index.unique())
    assert dtr.max() < dva.min()
    pos = {d: i for i, d in enumerate(uniq)}
    assert pos[dva.min()] - pos[dtr.max()] > 10  # horizonte + embargo
    assert len(set(tr) & set(va)) == 0
    assert len(va) == pytest.approx(0.15 * len(df), rel=0.05)


def test_early_stopping_solo_usa_el_train():
    """La elección del nº de árboles depende solo del train recibido: lo ocurrido después no la mueve."""
    df = _panel(260, 10)
    train = df.iloc[:2000]
    arm = AntiNoiseArm(COLS, num_leaves=4, min_child_samples=50, seeds=(0, 1), n_max=100, check_every=20)
    n1 = arm.best_iteration(train)
    # Alterar por completo lo posterior al train no cambia nada (no se le pasa al arm)
    df2 = df.copy()
    df2.iloc[2000:, df2.columns.get_loc("fwd_ret")] = 99.0
    n2 = arm.best_iteration(df2.iloc[:2000])
    assert n1 == n2
    # La validación interna es posterior al train interno (causal dentro del train)
    tr, va = inner_purged_split(train.index)
    assert train.index[tr].max() < train.index[va].min()
    assert arm.check_every <= n1 <= arm.n_max


@pytest.mark.parametrize("make", [
    lambda: XsLgbmArm(COLS, {"n_estimators": 40, "num_leaves": 7}),
    lambda: RankerArm(COLS, "lambdarank", {"n_estimators": 40, "num_leaves": 7}),
    lambda: RidgeArm(COLS),
    lambda: AntiNoiseArm(COLS, 4, 50, seeds=(0, 1), n_max=60, check_every=20),
])
def test_determinismo_con_semilla(make):
    df = _panel(120, 12)
    tr, te = df.iloc[:1000], df.iloc[1000:]
    a = make().fit(tr).predict(te)
    b = make().fit(tr).predict(te)
    np.testing.assert_allclose(a, b)
    assert np.isfinite(a).all()


def test_ridge_encuentra_la_feature_buena():
    df = _panel(150, 12)
    arm = RidgeArm(COLS).fit(df)
    coef = dict(zip(COLS, arm.model_.coef_))
    assert coef["rn_a"] > 5 * abs(coef["rn_b"])


def test_pesos_shrinkage_y_auc():
    w = shrunk_ic_weights(pd.Series({"x": 0.04, "y": 0.0, "z": -0.02}), lam=0.25)
    assert w.sum() == pytest.approx(1.0)
    assert w["x"] > w["y"] > 0 and w["z"] == pytest.approx(0.75 / 3)  # IC negativo: solo la parte 1/N
    assert shrunk_ic_weights(pd.Series({"x": -1.0, "y": -2.0})).tolist() == [0.5, 0.5]
    wa = auc_weights(pd.Series({"x": 0.53, "y": 0.50, "z": 0.48}))
    assert wa["x"] == pytest.approx(1.0) and wa["y"] == 0.0
    assert auc_weights(pd.Series({"x": 0.4, "y": 0.5})).tolist() == [0.5, 0.5]


def test_weights_path_causal():
    """Cambiar el IC realizado después de i − lag no cambia el peso en i."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=400)
    hist = pd.DataFrame(rng.normal(0.02, 0.1, size=(400, 3)), index=idx, columns=list("xyz"))
    w1 = weights_path(hist, "ic", window=100, lag=10, every=5, min_hist=50)
    h2 = hist.copy()
    h2.iloc[300:] += 5.0
    w2 = weights_path(h2, "ic", window=100, lag=10, every=5, min_hist=50)
    # posiciones < 300 + lag no deben cambiar
    pd.testing.assert_frame_equal(w1.iloc[:310], w2.iloc[:310])
    assert not w1.iloc[320:].equals(w2.iloc[320:])
    np.testing.assert_allclose(w1.sum(axis=1), 1.0)
    eq = weights_path(hist, "equal")
    assert (eq == 1 / 3).all().all()
    # sin historia suficiente => 1/N
    assert (w1.iloc[:50] == 1 / 3).all().all()


def test_blend_y_terciles():
    df = _panel(30, 9)
    sc = pd.DataFrame({"u": df["rn_a"], "v": df["rn_b"]})
    w = pd.DataFrame({"u": 0.75, "v": 0.25}, index=df.index.unique())
    out = blend_scores(sc, w)
    np.testing.assert_allclose(out.to_numpy(), 0.75 * sc["u"] + 0.25 * sc["v"])
    b = tercile_buckets(df.index, df["a"].to_numpy())
    assert set(b) == {0, 1, 2}
    assert (pd.Series(b, index=df.index).groupby(level=0).value_counts() == 3).all()
    df["score"] = df["a"]
    df["sig"] = df["b"]
    res = ic_by_tercile(df, "score", "fwd_ret", "sig", min_names=3)
    assert set(res) == {0, 1, 2} and all(res[t].mean() > 0.3 for t in res)
