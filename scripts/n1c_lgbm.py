"""
N1-C2 (sustituto de TabPFN, ver n1c_tabpfn.md): LightGBM pequeno, target rank cross-sectional, ranks de features, MISMOS 6 folds de 1B.
TabPFN (venv aparte en el scratchpad, torch 2.5.1+cpu, tabpfn 2.0.9) instalo y bajo pesos, pero no logro ni fit(1000x42)+predict(500)
en 300 s con la maquina en swap lleno (carga 11-20): no viable, no se insiste.
Reusa scripts/n1b_target y scripts/n1c_abstencion. Ejecucion: .venv/bin/python scripts/n1c_lgbm.py
"""
from __future__ import annotations
import gc, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
import scripts.n0a_diagnostico as A
import scripts.n0b_diagnostico as B
import scripts.n1b_target as T
import scripts.n1c_abstencion as C
from src.ml.features import FEATURE_COLS
from src.ml.trial_stats import deflated_sharpe
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, count_trials, log_trial

OUT = ROOT / "data" / "eval"
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.5, bagging_freq=1, lambda_l2=50.0, num_threads=2, verbose=-1, seed=42)
N_TREES = 150


def lgb_oos(df, X, tcol, h):
    tgt = df[tcol].to_numpy(dtype="float64"); ret = df[f"r{h}"].to_numpy(dtype="float64"); parts = []
    for fi, (tr, te) in enumerate(T.folds_iter(df, h)):
        tr = tr[np.isfinite(tgt[tr])]; te = te[np.isfinite(ret[te])]
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], tgt[tr].astype("float32")), num_boost_round=N_TREES)
        parts.append(pd.DataFrame({"date": df["date"].values[te], "ticker": df["ticker"].values[te], "ret": ret[te],
                                   "score": m.predict(X[te]), "fold": fi + 1}))
        del m; gc.collect()
    return pd.concat(parts, ignore_index=True)


def main():
    df = A.load_panel(T.SCRATCH / "panel.parquet")
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[np.isfinite(df["ret_fwd"])].reset_index(drop=True)
    df = df.merge(T.build_extra(df), on=["date", "ticker"], how="left"); gc.collect()
    df = T.add_targets(df); df = T.calendar_cols(df); df["rev1d"] = -df["ret1d"]
    X = T.rank_features(df, cols)
    rr = (df.groupby("date")["rev1d"].rank(pct=True) - 0.5).fillna(0.0).to_numpy(dtype="float32")
    X_aug = np.column_stack([X, rr, rr * df["last5"].to_numpy(dtype="float32")]).astype("float32")
    liq, _ = B.liquidity_panel(sorted(df["ticker"].unique()))
    R = {"meta": {"params": PARAMS, "n_trees": N_TREES,
                  "tabpfn": "NO VIABLE: venv scratchpad instalado (torch 2.5.1+cpu, tabpfn 2.0.9, pesos 44MB descargados) pero fit 1000x42 + predict 500 no termino en 300 s "
                            "(swap 2035/2047 MB, carga 11-20, ~1 GB libre). Contexto de 3000 filas x walk-forward trimestral inviable. Sustituido por LightGBM."},
         "cfg": {}}
    series = {}
    for h in (5, 21):
        oos = lgb_oos(df, X_aug, f"t_rank{h}", h)
        tg = df[["date", "ticker", f"t_rank{h}"]].rename(columns={f"t_rank{h}": "tgt"})
        oos = oos.merge(tg, on=["date", "ticker"], how="left")
        r, sr = T.eval_oos(oos, h, liq)
        nm = f"lgbm_rank_aug_h{h}"
        series[nm] = sr
        T.log(f"n1c_{nm}", {"model": "lgbm", "params": PARAMS, "n_trees": N_TREES, "target": "rank", "features": "ranks40+rev1d+rev1d*last5"}, r, h)
        R["cfg"][nm] = {"h": h, **B.clean(r)}
        print(nm, f"IC={T.f(r['ic'])} t={T.f(r['t_nw'],2)} aT={T.f(r.get('alpha_net_turn_pct'),3)} turn={T.f(r.get('turnover_mean'),2)} "
              f"folds+={r.get('folds_pos_turn')} IC24+={T.f(r['por_anio']['2024+']['ic'])} T3={T.f(r['por_tercil'].get('T3_liquido',{}).get('ic'))}", flush=True)
        for rule, par, rn in [("topk", {"k": 10, "b": 10}, "top10_b10"), ("gate_disp", {"q": 0.5}, "gate_disp_q50")]:
            rc, sc = C.evaluate(oos, liq, h, rule, par)
            full = f"{nm}_{rn}"; series[full] = sc
            R["cfg"][full] = {"h": h, "regla": rn, **B.clean(rc)}
            log_trial(f"n1c_{full}", {"tanda": "n1c", "model": "lgbm", "h": h, "regla": rn, "par": par, "params": PARAMS}, 
                      B.clean({k: rc.get(k) for k in ("alpha_net_pct", "turnover", "exposicion", "folds_pos", "ic_traded")}), int(rc["n_rebal"]), path=str(ROOT / TRIALS_PATH))
            print("  ", rn, f"aN={T.f(rc['alpha_net_pct'])} turn={T.f(rc['turnover'],2)} folds+={rc['folds_pos']}/6 24+={T.f(rc.get('alpha_net_2024p_pct'))} T3 aN={T.f((rc.get('T3') or {}).get('alpha_net_pct'))}", flush=True)
        json.dump(R, open(OUT / "n1c_lgbm.json", "w"), indent=2, default=str)
    n = count_trials(); R["meta"]["n_trials_global"] = n
    for k, v in series.items():
        R["cfg"][k]["dsr_global"] = B.clean(deflated_sharpe(v, n, periods=max(1, round(252 / R["cfg"][k]["h"]))))
    json.dump(R, open(OUT / "n1c_lgbm.json", "w"), indent=2, default=str)
    L = ["# N1-C2 LightGBM pequeno (sustituto de TabPFN)", "", "- " + R["meta"]["tabpfn"], f"- LightGBM {N_TREES} arboles, params {PARAMS}; target rank por fecha, features ranks40+rev1d+rev1d*last5; mismos 6 folds; N global DSR = {n}.", "",
         "| config | IC | t NW | alfa neto turn % | turnover | folds+ | IC 2024+ | IC T3 | DSR |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in R["cfg"].items():
        d = (r.get("dsr_global") or {}).get("dsr")
        if "regla" in r:
            L.append(f"| {k} | - | - | {T.f(r.get('alpha_net_pct'),3)} | {T.f(r.get('turnover'),2)} | {r.get('folds_pos')}/6 | alfa 2024+ {T.f(r.get('alpha_net_2024p_pct'),3)} | T3 alfa {T.f((r.get('T3') or {}).get('alpha_net_pct'),3)} | {T.f(d,3)} |")
        else:
            L.append(f"| {k} | {T.f(r['ic'])} | {T.f(r['t_nw'],2)} | {T.f(r.get('alpha_net_turn_pct'),3)} | {T.f(r.get('turnover_mean'),2)} | {r.get('folds_pos_turn')}/6 | {T.f(r['por_anio']['2024+']['ic'])} | {T.f(r['por_tercil'].get('T3_liquido',{}).get('ic'))} | {T.f(d,3)} |")
    (OUT / "n1c_lgbm.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("ok N=", n)

if __name__ == "__main__":
    main()
