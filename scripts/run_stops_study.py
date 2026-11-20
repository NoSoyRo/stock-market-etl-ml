"""
M8 · Estudio de stops por σ̂, Kelly estimado, veto de cola y chequeo de régimen.

Uso:
    .venv/bin/python scripts/run_stops_study.py          # completo (~3-5 min)
    .venv/bin/python scripts/run_stops_study.py --fast   # 16 tickers, 60 sorteos

Diseño FIJADO antes de mirar resultados
  (a) Triple barrera h=5, entrada Open t+1, desempate gap_aware, costo 0.30%/lado.
      Barreras k·σ̂5 (σ̂5 = EWMA21 diaria·√5) vs k·ATR%, k ∈ {1.0, 1.5}. Comparación primaria:
      k=1.5 (el múltiplo en producción). "Sirve" si σ̂ sube la expectativa neta por operación.
  (b) Kelly fijo (3.32/2.44) vs estimado (pagos del train, retorno 5d neto de 0.6%).
  (c) Tail = 1[r5 < −k·σ̂5], k ∈ {1.0, 1.5}. Modelo LightGBM en train (purgado) vs baseline
      "solo vol pasada" (σ̂5). Veto = quitar el 20% de nombres con mayor riesgo de cola por
      fecha. Top-5 con calendarios: semanal plano (5d, salida K) y selección M5 (10d, salida 2K).
      "Sirve" si MDD sube (menos negativo) y el neto no baja más de 1 error estándar.
  (d) Régimen: vol20 y ret5 medios por estado del HMM (solo tabla).
Convenciones: fwd_ret en % → fracciones; ppy = 50.4; semilla 42; costo base 0.30%/lado.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from scripts.backtest_topk import (  # noqa: E402
    HORIZON, SEED, TRAIN_RATIO, build_panel, load_universe_tickers, model_feature_cols,
)
from scripts.run_m1_report import rebalance_panel, sharpe  # noqa: E402
from scripts.run_rotation_study import wide  # noqa: E402
from src.data.store import get as store_get  # noqa: E402
from src.ml.ensemble import EnsembleModel  # noqa: E402
from src.ml.labeling import triple_barrier_labels  # noqa: E402
from src.ml.risk_engine import HIST_AVG_LOSS, HIST_AVG_WIN, kelly_fraction  # noqa: E402
from src.ml.shared.evaluators.cross_section import auc_by_date, ic_summary  # noqa: E402
from src.ml.shared.evaluators.costs import breakeven_cost_per_side, net_returns, turnover  # noqa: E402
from src.ml.shared.evaluators.portfolio_rules import (  # noqa: E402
    equal_weight_weights, held_returns, rebalance_step, select_with_hysteresis,
)
from src.ml.stops import (  # noqa: E402
    daily_sigma, estimate_payoffs, horizon_sigma, kelly_fraction_estimated,
    max_drawdown, paired_block_bootstrap, tail_label,
)

PPY = 252 / HORIZON
COST = 0.003                 # por lado
KS = (1.0, 1.5)              # múltiplos de barrera y de cola
VETO_FRAC = 0.20
MAX_ABS_RET_PCT = 100.0      # |fwd_ret| mayor = fila basura (Open≈0): se excluye de los pagos
SCHEDULES = {"semanal_K5": (5, 5, 5), "M5_10d_K5_2K": (10, 5, 10)}  # (freq_días, k, exit_rank)
TAIL_PARAMS = dict(objective="binary", verbosity=-1, n_estimators=200, num_leaves=15,
                   learning_rate=0.03, min_child_samples=100, feature_fraction=0.6,
                   bagging_fraction=0.8, bagging_freq=1, reg_lambda=5.0, n_jobs=2,
                   random_state=SEED)


# ─── (a) triple barrera ──────────────────────────────────────────────────────

def barrier_study(tickers: dict[str, str], test_start: pd.Timestamp, picks: pd.DataFrame) -> dict:
    """σ̂ vs ATR en la triple barrera: % tiempo/stop/target y expectativa neta por operación."""
    series: dict[str, list[pd.DataFrame]] = {f"{m}_{k}": [] for m in ("atr", "sig") for k in KS}
    for name in tickers:
        raw = store_get(name)
        if raw is None or len(raw) < 200:
            continue
        prev_c = raw["Close"].shift(1)
        tr = pd.concat([raw["High"] - raw["Low"], (raw["High"] - prev_c).abs(),
                        (raw["Low"] - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean() / (raw["Close"] + 1e-8)
        sig5 = horizon_sigma(daily_sigma(raw["Close"], span=21), HORIZON)
        for m, vol in (("atr", atr), ("sig", sig5)):
            for k in KS:
                tb = triple_barrier_labels(
                    raw["Close"], raw["High"], raw["Low"], vol, horizon=HORIZON,
                    mult_target=k, mult_stop=k, open_=raw["Open"],
                    entry="next_open", tie_break="gap_aware")
                tb = tb.dropna(subset=["ret"]).copy()
                tb["net"] = tb["ret"] - 2 * COST * 100       # % neto de ida y vuelta
                tb["ticker"] = name
                series[f"{m}_{k}"].append(tb)
    out: dict = {}
    for key, parts in series.items():
        tb = pd.concat(parts)
        # solo fechas donde existen ambas σ̂ y ATR (comparación sobre las mismas señales)
        out[key] = tb
    common = None
    for tb in out.values():
        ix = tb.set_index("ticker", append=True).index
        common = ix if common is None else common.intersection(ix)
    res: dict = {"n_signals": int(len(common)), "configs": {}}
    per_date: dict[str, dict[str, pd.Series]] = {}
    for key, tb in out.items():
        tb = tb.set_index("ticker", append=True).loc[common]
        win = {"todas": tb, "test": tb[tb.index.get_level_values(0) >= test_start]}
        pk = picks.set_index([picks.index, "ticker"]).index
        sel = tb[tb.index.isin(pk)] if len(pk) else tb.iloc[:0]
        win["top5_semanal"] = sel
        res["configs"][key] = {}
        per_date[key] = {}
        for wname, sub in win.items():
            d = sub.groupby(level=0)["net"].mean()
            per_date[key][wname] = d
            s = ic_summary(d, step=HORIZON, block=4, seed=0)
            tc = sub["touch"].value_counts(normalize=True)
            res["configs"][key][wname] = {
                "n": int(len(sub)), "pct_time": float(tc.get("time", 0)),
                "pct_stop": float(tc.get("stop", 0)), "pct_target": float(tc.get("target", 0)),
                "net_per_trade_pct": float(sub["net"].mean()),
                "gross_per_trade_pct": float(sub["ret"].mean()),
                "win_rate_gross": float((sub["ret"] > 0).mean()),
                "per_date_mean": s,
            }
    res["delta_sig_minus_atr"] = {}
    for k in KS:
        res["delta_sig_minus_atr"][str(k)] = {}
        for wname in ("todas", "test", "top5_semanal"):
            a, b = per_date[f"sig_{k}"][wname], per_date[f"atr_{k}"][wname]
            ix = a.index.intersection(b.index)
            diff = (a.loc[ix] - b.loc[ix]).iloc[::HORIZON]
            r = paired_block_bootstrap(a.loc[ix].iloc[::HORIZON], b.loc[ix].iloc[::HORIZON],
                                       lambda x: float(x.mean()), seed=1)
            res["delta_sig_minus_atr"][str(k)][wname] = {
                **r, "n_dates": int(len(diff)), "sirve": bool(r["diff"] > 0)}
    return res


# ─── (b) Kelly ───────────────────────────────────────────────────────────────

def kelly_study(train_ret_pct: pd.Series, p_up: pd.Series, top_p: pd.Series) -> dict:
    pay = estimate_payoffs(train_ret_pct, cost_pct=2 * COST * 100)

    def fr(p, kind):
        if kind == "fijo":
            return np.array([kelly_fraction(x) for x in p])
        return np.array([kelly_fraction_estimated(x, pay) for x in p])

    out = {"payoffs_estimados": {"avg_win_pct": pay.avg_win, "avg_loss_pct": pay.avg_loss,
                                 "n_train": int(train_ret_pct.notna().sum())},
           "payoffs_fijos": {"avg_win_pct": HIST_AVG_WIN, "avg_loss_pct": HIST_AVG_LOSS},
           "p_equilibrio": {"fijo": HIST_AVG_LOSS / (HIST_AVG_WIN + HIST_AVG_LOSS),
                            "estimado": pay.avg_loss / (pay.avg_win + pay.avg_loss)
                            if pay.avg_win + pay.avg_loss > 0 else None},
           "p_up_dist": {q: float(p_up.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
    for kind in ("fijo", "estimado"):
        f50 = float(fr([0.50], kind)[0])
        f_all, f_top = fr(p_up.to_numpy(), kind), fr(top_p.to_numpy(), kind)
        out[kind] = {"f_con_p050": f50, "f_media_p_up": float(f_all.mean()),
                     "pct_con_apuesta": float((f_all > 0).mean()),
                     "f_media_top5": float(f_top.mean()),
                     "pct_top5_con_apuesta": float((f_top > 0).mean())}
    return out


# ─── (c) cola y veto ─────────────────────────────────────────────────────────

def fit_tail_scores(train: pd.DataFrame, test: pd.DataFrame, cut: pd.Timestamp) -> dict[float, pd.Series]:
    """Modelos de cola por k entrenados en train purgado (sin los últimos ~7 hábiles)."""
    import lightgbm as lgb
    tr = train[train.index < cut - pd.Timedelta(days=10)]
    cols = model_feature_cols(train.columns) + ["log_sig5"]
    out = {}
    for k in KS:
        y = tr[f"tail_{k}"]
        ok = y.notna()
        m = lgb.LGBMClassifier(**TAIL_PARAMS).fit(tr.loc[ok, cols], y[ok].astype(int))
        out[k] = pd.Series(m.predict_proba(test[cols])[:, 1], index=test.index)
    return out


def train_vol_sign(train: pd.DataFrame, cut: pd.Timestamp) -> dict[float, int]:
    """Signo (+1/−1) con que σ̂ predice la cola EN TRAIN (AUC pooled > 0.5 → +1).

    La cola se mide en múltiplos de σ̂, así que σ̂ alta la vuelve MÁS difícil (AUC < 0.5): el
    baseline justo usa la orientación aprendida en train, no la del test.
    """
    tr = train[train.index < cut - pd.Timedelta(days=10)]
    out = {}
    for k in KS:
        ok = tr[f"tail_{k}"].notna()
        out[k] = 1 if roc_auc_score(tr.loc[ok, f"tail_{k}"].astype(int), tr.loc[ok, "sig5"]) >= 0.5 else -1
    return out


def tail_auc_table(rp: pd.DataFrame, sign: dict[float, int]) -> dict:
    res = {}
    for k in KS:
        y = f"tail_{k}"
        a_mod = auc_by_date(rp, y, f"ptail_{k}")
        rp = rp.assign(_base=sign[k] * rp["sig5"])
        a_raw = auc_by_date(rp, y, "sig5")
        a_base = auc_by_date(rp, y, "_base")
        ix = a_mod.index.intersection(a_base.index)
        d = paired_block_bootstrap(a_mod.loc[ix], a_base.loc[ix], lambda x: float(x.mean()), seed=2)
        res[str(k)] = {"tasa_cola": float(rp[y].mean()), "signo_vol_en_train": sign[k],
                       "auc_modelo": ic_summary(a_mod, step=1, seed=0),
                       "auc_vol_pasada_cruda": ic_summary(a_raw, step=1, seed=0),
                       "auc_baseline_vol_pasada": ic_summary(a_base, step=1, seed=0),
                       "modelo_menos_baseline": d, "gana_al_baseline": bool(d["ci_lo"] > 0)}
    return res


def veto_scores(rp: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Matriz fecha×ticker True donde el nombre queda vetado (top VETO_FRAC por fecha)."""
    w = wide(rp, score_col)
    pct = w.rank(axis=1, pct=True, method="first")
    return pct > (1.0 - VETO_FRAC)


def run_strategy(scores: pd.DataFrame, ret: pd.DataFrame, k: int, exit_rank: int,
                 freq: int) -> tuple[pd.Series, pd.Series]:
    """Devuelve (bruto, rotación) del top-k con histéresis."""
    w = select_with_hysteresis(scores, k, exit_rank, freq)
    return held_returns(w, ret), turnover(w)


def summarize(g: pd.Series, t: pd.Series) -> dict:
    n = net_returns(g, t, COST)
    return {"turnover": float(t.mean()), "gross_mean": float(g.mean()),
            "net_mean": float(n.mean()), "net_se": float(n.std(ddof=1) / np.sqrt(len(n))),
            "sharpe_net": sharpe(n), "mdd_net": max_drawdown(n),
            "breakeven_per_side": breakeven_cost_per_side(g, t), "n": int(len(n))}


def veto_study(rp: pd.DataFrame, n_draws: int) -> dict:
    ret = wide(rp, "ret")
    rng = np.random.default_rng(SEED)
    jitter = pd.DataFrame(rng.uniform(0, 1e-9, size=ret.shape), index=ret.index, columns=ret.columns)
    base = wide(rp, "p_up").reindex(index=ret.index, columns=ret.columns) + jitter
    vetoes = {"ninguno": None}
    for k in KS:
        vetoes[f"cola_k{k}"] = veto_scores(rp, f"ptail_{k}").reindex_like(ret).fillna(False)
    vetoes["vol_pasada"] = veto_scores(rp, "sig5").reindex_like(ret).fillna(False)
    res: dict = {}
    for sname, (freq, k, ex) in SCHEDULES.items():
        res[sname] = {"strategies": {}, "checks": {}}
        nets, mats = {}, {}
        # equiponderado a la misma frecuencia
        ew = equal_weight_weights(base, freq)
        ge, te = held_returns(ew, ret), turnover(ew)
        res[sname]["equiponderado"] = summarize(ge, te)
        for vname, mask in vetoes.items():
            sc = base if mask is None else base.mask(mask)
            g, t = run_strategy(sc, ret, k, ex, freq)
            res[sname]["strategies"][vname] = summarize(g, t)
            nets[vname] = net_returns(g, t, COST)
            # aleatorio sobre el mismo universo elegible (con el mismo veto)
            sims = []
            for d in range(n_draws):
                r_sc = pd.DataFrame(np.random.default_rng(1000 + d).uniform(size=ret.shape),
                                    index=ret.index, columns=ret.columns)
                r_sc = r_sc.where(sc.notna()) if mask is not None else r_sc
                gg, tt = run_strategy(r_sc, ret, k, ex, freq)
                nn = net_returns(gg, tt, COST)
                sims.append((float(nn.mean()), max_drawdown(nn), sharpe(nn)))
            sims = np.array(sims)
            m = res[sname]["strategies"][vname]
            m["aleatorio_neto_medio_p5_p50_p95"] = [float(x) for x in np.percentile(sims[:, 0], [5, 50, 95])]
            m["percentil_en_aleatorio_neto"] = float((sims[:, 0] < m["net_mean"]).mean() * 100)
            m["aleatorio_mdd_mediana"] = float(np.median(sims[:, 1]))
        b = res[sname]["strategies"]["ninguno"]
        for vname in vetoes:
            if vname == "ninguno":
                continue
            m = res[sname]["strategies"][vname]
            d_mean = paired_block_bootstrap(nets[vname], nets["ninguno"], lambda x: float(x.mean()), seed=3)
            d_mdd = paired_block_bootstrap(nets[vname], nets["ninguno"], max_drawdown, seed=4)
            res[sname]["checks"][vname] = {
                "delta_neto_medio": d_mean, "delta_mdd": d_mdd,
                "se_neto_sin_veto": b["net_se"],
                "recorta_mdd": bool(m["mdd_net"] > b["mdd_net"]),
                "neto_no_baja_mas_de_1se": bool(m["net_mean"] - b["net_mean"] >= -b["net_se"]),
                "sirve": bool(m["mdd_net"] > b["mdd_net"]
                              and m["net_mean"] - b["net_mean"] >= -b["net_se"]),
            }
    return res


# ─── (d) régimen ─────────────────────────────────────────────────────────────

def regime_table(test_start: pd.Timestamp) -> dict:
    from src.ml.regime import RegimeDetector, _build_features
    naf, sp = store_get("NAFTRAC"), store_get("SP500")
    det = RegimeDetector().fit(naf, sp)
    labels = det.predict_series(naf, sp)
    feats = _build_features(naf, sp).reindex(labels.index)
    df = feats.assign(regime=labels)

    def tab(d: pd.DataFrame) -> dict:
        rows = {}
        for r, g in d.groupby("regime"):
            rows[r] = {"n_dias": int(len(g)), "pct_tiempo": float(len(g) / len(d) * 100),
                       "vol20_diaria_media_pct": float(g["vol20"].mean() * 100),
                       "vol20_anualizada_media_pct": float(g["vol20"].mean() * np.sqrt(252) * 100),
                       "ret5_medio_pct": float(g["ret5"].mean() * 100),
                       "ret5_std_pct": float(g["ret5"].std() * 100)}
        return rows

    def eta2(d: pd.DataFrame, col: str) -> float:
        tot = ((d[col] - d[col].mean()) ** 2).sum()
        bet = d.groupby("regime")[col].apply(lambda g: len(g) * (g.mean() - d[col].mean()) ** 2).sum()
        return float(bet / tot)

    out = {"completo": tab(df), "test": tab(df[df.index >= test_start]),
           "eta2_vol20": eta2(df, "vol20"), "eta2_ret5": eta2(df, "ret5"),
           "nota": "HMM ajustado en toda la muestra y etiquetado por ret5: solo descriptivo (hay look-ahead)."}
    out["separa_por_vol_no_direccion"] = bool(out["eta2_vol20"] > out["eta2_ret5"])
    return out


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="M8 · stops, Kelly, veto de cola, régimen")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    np.random.seed(SEED)
    tickers = load_universe_tickers("bmv")
    if args.fast:
        tickers = dict(list(tickers.items())[:16])
    n_draws = 60 if args.fast else 200

    df = build_panel(tickers, horizon=HORIZON, score_mode="p_up", use_open_entry=True).sort_index()
    # σ̂ diaria EWMA21 por ticker (causal, hasta el cierre de la señal)
    sig = {n: daily_sigma(store_get(n)["Close"], span=21) for n in tickers if store_get(n) is not None}
    df["sig5"] = [horizon_sigma(sig[t], HORIZON).get(d, np.nan) for d, t in zip(df.index, df["ticker"])]
    df = df[df["sig5"] > 0].copy()
    df["log_sig5"] = np.log(df["sig5"])
    for k in KS:
        df[f"tail_{k}"] = tail_label(df["fwd_ret"] / 100.0, df["sig5"], k)

    cut = df.index[int(len(df) * TRAIN_RATIO)]
    train, test = df[df.index < cut].copy(), df[df.index >= cut].copy()
    np.random.seed(SEED)
    model = EnsembleModel(top_features=15, calibrate=True)
    feat_cols = model_feature_cols(train.columns)   # sin tasas/dividendos (casi todo NaN)
    model.fit(train, feature_cols=feat_cols, target_col="y")
    test["p_up"] = model.predict_proba(test, feat_cols)
    for k, s in fit_tail_scores(train, test, cut).items():
        test[f"ptail_{k}"] = s.to_numpy()
    rp = rebalance_panel(test)
    print(f"test {test.index.min().date()}→{test.index.max().date()} · fechas {rp.index.nunique()}")

    # picks semanales top-5 de p_up (desempate aleatorio fijo) para la parte (a)
    ret_w = wide(rp, "ret")
    rng = np.random.default_rng(SEED)
    jit = pd.DataFrame(rng.uniform(0, 1e-9, size=ret_w.shape), index=ret_w.index, columns=ret_w.columns)
    w5 = select_with_hysteresis(wide(rp, "p_up").reindex_like(ret_w) + jit, 5, 5, 5)
    picks = w5.stack()
    picks = picks[picks > 0].reset_index()
    picks.columns = ["date", "ticker", "w"]
    picks = picks.set_index("date")

    rep: dict = {"generated_at": datetime.now().isoformat(timespec="seconds"), "fast": args.fast,
                 "seed": SEED, "cost_per_side": COST, "veto_frac": VETO_FRAC, "n_draws": n_draws,
                 "test_start": str(test.index.min().date()), "test_end": str(test.index.max().date()),
                 "n_dates": int(rp.index.nunique())}
    rep["barreras"] = barrier_study(tickers, test.index.min(), picks)
    top_ids = picks.reset_index().set_index(["date", "ticker"]).index
    top_p = rp.reset_index(names="date").set_index(["date", "ticker"]).loc[top_ids, "p_up"]
    tr_ret = train.loc[train.index < cut - pd.Timedelta(days=10), "fwd_ret"]
    bad = tr_ret.abs() > MAX_ABS_RET_PCT
    rep["datos_basura"] = {"filas_train_descartadas_en_pagos": int(bad.sum()),
                           "filas_test_con_abs_ret_mayor_100": int((test["fwd_ret"].abs() > MAX_ABS_RET_PCT).sum()),
                           "umbral_pct": MAX_ABS_RET_PCT}
    rep["kelly"] = kelly_study(tr_ret[~bad], rp["p_up"], top_p)
    rep["cola_auc"] = tail_auc_table(rp, train_vol_sign(train, cut))
    rep["veto"] = veto_study(rp, n_draws)
    rep["regimen"] = regime_table(test.index.min())
    n_trials = len(KS) * 2 + 2 + len(KS) + len(SCHEDULES) * 4 + 1   # 4+2+2+8+1 = 17
    rep["ensayos"] = {"total": n_trials,
                      "detalle": "4 barreras (2 k × {ATR,σ̂}) + 2 Kelly + 2 modelos de cola + "
                                 "2 calendarios × 4 vetos (ninguno, cola k1, cola k1.5, vol pasada) + 1 HMM"}
    path = ROOT / "data" / "eval" / ("stops_study_fast.json" if args.fast else "stops_study.json")
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"Ensayos: {n_trials} · {path}")
    _print(rep)


def _print(rep: dict) -> None:
    b = rep["barreras"]
    print(f"\n(a) señales comunes: {b['n_signals']}")
    for key, w in b["configs"].items():
        for wn in ("todas", "top5_semanal"):
            c = w[wn]
            print(f"  {key:8}{wn:13} n={c['n']:6} tiempo={c['pct_time']:.2f} stop={c['pct_stop']:.2f} "
                  f"tgt={c['pct_target']:.2f} neto/op={c['net_per_trade_pct']:+.3f}%")
    for k, d in b["delta_sig_minus_atr"].items():
        for wn, r in d.items():
            print(f"  Δ σ̂−ATR k={k} {wn:13} {r['diff']:+.3f}pp IC[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                  f"sirve={r['sirve']}")
    print("\n(b) Kelly:", json.dumps({k: rep["kelly"][k] for k in ("fijo", "estimado", "payoffs_estimados")},
                                     default=float))
    print("\n(c) AUC cola:")
    for k, r in rep["cola_auc"].items():
        print(f"  k={k} tasa={r['tasa_cola']:.3f} modelo={r['auc_modelo']['mean']:.4f} "
              f"base_cruda={r['auc_vol_pasada_cruda']['mean']:.4f} base_orient={r['auc_baseline_vol_pasada']['mean']:.4f} Δ={r['modelo_menos_baseline']['diff']:+.4f} "
              f"IC[{r['modelo_menos_baseline']['ci_lo']:+.4f},{r['modelo_menos_baseline']['ci_hi']:+.4f}]")
    for s, d in rep["veto"].items():
        print(f"  {s}: EW neto={d['equiponderado']['net_mean']*100:+.3f}% mdd={d['equiponderado']['mdd_net']:.3f}")
        for v, m in d["strategies"].items():
            print(f"    {v:11} rot={m['turnover']:.2f} neto={m['net_mean']*100:+.3f}% (se {m['net_se']*100:.3f}) "
                  f"sh={m['sharpe_net']:+.2f} mdd={m['mdd_net']:+.3f} pctl_alea={m['percentil_en_aleatorio_neto']:.0f}")
        for v, c in d["checks"].items():
            print(f"    check {v:11} recorta_mdd={c['recorta_mdd']} neto_ok={c['neto_no_baja_mas_de_1se']} "
                  f"sirve={c['sirve']}")
    r = rep["regimen"]
    print(f"\n(d) η² vol20={r['eta2_vol20']:.3f} ret5={r['eta2_ret5']:.3f}")
    for s, v in r["completo"].items():
        print(f"  {s:8} n={v['n_dias']:5} vol20={v['vol20_anualizada_media_pct']:.1f}% ret5={v['ret5_medio_pct']:+.2f}%")


if __name__ == "__main__":
    main()
