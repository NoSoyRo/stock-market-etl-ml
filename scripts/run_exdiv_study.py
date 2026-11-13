"""N3 · estudio de captura del ex-dividendo (criterio en avances/2026-11-12_N3_preregistro.md).

Uso: .venv/bin/python scripts/run_exdiv_study.py
Salidas: data/eval/exdiv_study.json y .md; ensayos en data/eval/trial_log.jsonl (familia n3_exdiv).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest_topk import load_universe_tickers  # noqa: E402
from src.data.store import get  # noqa: E402
from src.ml.exdiv_capture import (  # noqa: E402
    causal_trades, event_returns, net_return, quarter_mean_t, top_n_share,
)
from src.ml.trials import log_trial  # noqa: E402

WINDOW = 5
COST = 0.003
TAX = 0.10
TRAIN = (2008, 2017)
TEST = (2018, 2026)
OUT_JSON = ROOT / "data" / "eval" / "exdiv_study.json"
OUT_MD = ROOT / "data" / "eval" / "exdiv_study.md"
DIV_DIR = ROOT / "data" / "dividends"


def load_prices() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Cierres y dividendos de los tickers MX: universo bmv + archivos `*.MX` de dividendos."""
    names = set(load_universe_tickers("bmv")) | {p.stem[:-3] for p in DIV_DIR.glob("*.MX.parquet")}
    closes, divs = {}, {}
    for n in sorted(names):
        f = DIV_DIR / f"{n}.MX.parquet"
        if not f.exists():
            continue
        px = get(n)
        if px is None or "Close" not in px:
            px = get(f"{n}.MX")
        if px is None or "Close" not in px or px.empty:
            continue
        d = pd.read_parquet(f)["dividend"]
        if d.empty:
            continue
        closes[n] = px["Close"].astype(float).sort_index()
        divs[n] = d
    return closes, divs


def ew_window_return(closes: dict[str, pd.Series], bmv: list[str]) -> pd.Series:
    """Retorno equiponderado de `WINDOW` días hábiles del universo bmv, por fecha de venta."""
    m = pd.DataFrame({k: closes[k] for k in bmv if k in closes}).sort_index()
    return (m / m.shift(WINDOW) - 1.0).mean(axis=1, skipna=True)


def attach(df: pd.DataFrame, ew: pd.Series, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = ticker
    df["ew"] = ew.reindex(df.index).to_numpy()
    return df


def stats_for(df: pd.DataFrame, col: str) -> dict:
    s = quarter_mean_t(df[col])
    s.update({f"top5_{k}": v for k, v in top_n_share(df[col], 5).items() if k != "n"})
    return s


def window_split(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    y = df.index.year
    return df[(y >= lo) & (y <= hi)]


def cols_for(df: pd.DataFrame) -> pd.DataFrame:
    """Añade las columnas de cada versión: A (largo-solo) y B (menos equiponderado), brutas y netas."""
    df = df.copy()
    for ver, base in (("A", df["gross"]), ("B", df["gross"] - df["ew"])):
        df[f"{ver}_gross"] = base
        df[f"{ver}_net_tax"] = net_return(base, df["yield"], tax=TAX, cost_per_side=COST)
        df[f"{ver}_net_notax"] = net_return(base, df["yield"], tax=0.0, cost_per_side=COST)
    return df


def variant_report(df: pd.DataFrame, q4_cut: float) -> dict:
    """Estadísticos por periodo para todos y cuartil alto de yield (umbral congelado de 2008-2017)."""
    out: dict = {}
    for scope, sub in (("all", df), ("q4", df[df["yield"] >= q4_cut])):
        for per, (lo, hi) in (("train", TRAIN), ("test", TEST), ("full", (2004, 2026))):
            w = window_split(sub, lo, hi)
            for ver in ("A", "B"):
                for kind in ("gross", "net_tax", "net_notax"):
                    out[f"{scope}|{per}|{ver}_{kind}"] = stats_for(w, f"{ver}_{kind}") if len(w) else {"n": 0}
    return out


def passes(rep: dict) -> dict:
    """Criterio del preregistro sobre la versión A, neto de costos y retención, en el cuartil alto."""
    te, tr = rep["q4|test|A_net_tax"], rep["q4|train|A_net_tax"]
    c1 = bool(te.get("mean", np.nan) > 0.003)
    t = te.get("t", np.nan)
    c2 = bool(t > 2) if np.isfinite(t) else False
    c3 = bool(tr.get("mean", np.nan) > 0)
    mid = bool(np.isfinite(t) and 1.5 < t <= 2)
    small = te.get("n_q", 0) < 20
    return {"c1_mean_gt_0.30pct": c1, "c2_t_gt_2": c2, "c3_same_sign_train": c3,
            "mean_test": te.get("mean"), "t_test": t, "n_q_test": te.get("n_q"),
            "mean_train": tr.get("mean"), "all": c1 and c2 and c3 and not small,
            "inconclusive": (c1 and c3) and (mid or small) and not (c2 and not small)}


def verdict(causal: dict, oracle: dict) -> tuple[str, str]:
    if causal["all"]:
        return "se sostiene", "la versión A causal en el cuartil alto cumple las tres condiciones"
    if causal["inconclusive"]:
        return "inconcluso", "la causal cumple 1 y 3 pero la t queda entre 1.5 y 2 o el n por trimestre es < 20"
    if oracle["all"]:
        return "no operable", "con la fecha real el efecto pasa, pero con la regla causal de fecha no"
    if oracle["inconclusive"]:
        return "inconcluso", "ni la causal pasa; el oráculo queda en zona gris"
    return "no se sostiene", "ni con la fecha real se cumple el criterio neto"


def pct(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}%"


def main() -> None:
    closes, divs = load_prices()
    bmv = list(load_universe_tickers("bmv"))
    ew = ew_window_return(closes, bmv)
    div_cols = sorted({tuple(pd.read_parquet(f).columns) for f in DIV_DIR.glob("*.MX.parquet")})

    ora, cau, ext = [], [], []
    for tk in closes:
        e = event_returns(closes[tk], divs[tk], WINDOW)
        if len(e):
            ora.append(attach(e, ew, tk))
        e_all = event_returns(closes[tk], divs[tk], WINDOW, max_yield=None)
        big = e_all[e_all["yield"] > 0.15]
        for d, r in big.iterrows():
            ext.append({"ticker": tk, "ex_date": str(r["ex_date"].date()), "yield": float(r["yield"]),
                        "gross": float(r["gross"])})
        c = causal_trades(closes[tk], divs[tk], WINDOW)
        if len(c):
            cau.append(attach(c, ew, tk))
    oracle = cols_for(pd.concat(ora).dropna(subset=["ew"]).sort_index())
    causal = cols_for(pd.concat(cau).dropna(subset=["ew"]).sort_index())

    # umbral del cuartil alto: percentil 75 del yield en 2008-2017 (congelado)
    cut_o = float(window_split(oracle, *TRAIN)["yield"].quantile(0.75))
    cut_c = float(window_split(causal, *TRAIN)["yield"].quantile(0.75))
    rep_o, rep_c = variant_report(oracle, cut_o), variant_report(causal, cut_c)
    pass_o, pass_c = passes(rep_o), passes(rep_c)
    ver, why = verdict(pass_c, pass_o)

    # sanidad: bruto vs equiponderado, todos los eventos 2004-2026
    san_B, san_A = rep_o["all|full|B_gross"], rep_o["all|full|A_gross"]

    # acierto de la fecha causal
    near = causal["offset"].abs() <= 2
    hit_by_trade = float(near.mean())
    ev_pos = {}
    cov = []
    for tk in closes:
        idx = closes[tk].dropna().index
        c = causal[causal["ticker"] == tk]
        o = oracle[oracle["ticker"] == tk]
        if o.empty:
            continue
        cp = idx.searchsorted(c.index) if len(c) else np.array([], dtype=int)
        for d in o.index:
            p = idx.searchsorted(d)
            cov.append(bool(len(cp) and np.abs(cp - p).min() <= 2))
    hit_by_event = float(np.mean(cov)) if cov else np.nan
    by_hit = {
        "hit_pm2": {"n": int(near.sum()), "A_gross_mean": float(causal.loc[near, "A_gross"].mean())},
        "miss": {"n": int((~near).sum()), "A_gross_mean": float(causal.loc[~near, "A_gross"].mean())},
        "captured_dividend": {"n": int((causal["n_div_captured"] > 0).sum()),
                              "share": float((causal["n_div_captured"] > 0).mean())},
    }

    # cuartiles de yield (umbrales de 2008-2017), solo versión A neta con retención, en 2018-2026
    qs = oracle_q = None
    edges = window_split(oracle, *TRAIN)["yield"].quantile([0.25, 0.5, 0.75]).tolist()
    bins = [-np.inf] + edges + [np.inf]
    te = window_split(oracle, *TEST)
    te = te.assign(q=pd.cut(te["yield"], bins=bins, labels=["Q1", "Q2", "Q3", "Q4"]))
    qs = {str(k): stats_for(g, "A_net_tax") for k, g in te.groupby("q", observed=True)}

    # por año (cuartil alto causal) y capacidad
    q4c = causal[causal["yield"] >= cut_c]
    by_year = {int(y): {"n": int(len(g)), "A_gross": float(g["A_gross"].mean()),
                        "A_net_tax": float(g["A_net_tax"].mean())} for y, g in q4c.groupby(q4c.index.year)}
    cap_all = oracle.groupby(oracle.index.year).size()
    q4o = oracle[oracle["yield"] >= cut_o]
    cap_q4 = q4o.groupby(q4o.index.year).size()
    q4c_test = window_split(q4c, *TEST)
    by_quarter = {str(k): float(v) for k, v in
                  q4c_test["A_net_tax"].groupby(q4c_test.index.to_period("Q")).mean().items()}

    # ensayos: una fila por variante × versión (métricas de 2018-2026, neto con retención)
    n_logged = 0
    for var, rep, cut in (("oraculo_todos", rep_o, None), ("oraculo_q4", rep_o, cut_o),
                          ("causal_todos", rep_c, None), ("causal_q4", rep_c, cut_c)):
        scope = "q4" if cut is not None else "all"
        for v in ("A", "B"):
            m = rep[f"{scope}|test|{v}_net_tax"]
            cfg = {"variant": var, "version": v, "window": WINDOW, "cost_per_side": COST, "tax": TAX,
                   "q4_cut": cut, "train": TRAIN, "test": TEST}
            wrote = log_trial(f"n3_exdiv:{var}_{v}", cfg,
                              {k: (None if (isinstance(x, float) and not np.isfinite(x)) else x)
                               for k, x in m.items()}, int(m.get("n", 0)))
            n_logged += int(wrote)
    from src.ml import trial_stats as ts
    n_family = ts.n_trials("n3_exdiv")

    result = {
        "params": {"window": WINDOW, "cost_per_side": COST, "tax": TAX, "train": TRAIN, "test": TEST,
                   "q4_cut_oracle": cut_o, "q4_cut_causal": cut_c, "dividend_parquet_columns": div_cols,
                   "tickers": len(closes), "execution": "Close de p-5 a Close de p"},
        "sanity": {"B_gross_full": san_B, "A_gross_full": san_A, "target_B_mean": 0.0085},
        "oracle": rep_o, "causal": rep_c,
        "criterion": {"causal": pass_c, "oracle": pass_o, "verdict": ver, "why": why},
        "causal_hit": {"share_trades_pm2": hit_by_trade, "share_events_covered_pm2": hit_by_event, **by_hit},
        "yield_quartiles_test_A_net_tax_oracle": qs,
        "q4_causal_by_year": by_year,
        "q4_causal_test_by_quarter_A_net_tax": by_quarter,
        "capacity": {"events_per_year_all": {int(k): int(v) for k, v in cap_all.items()},
                     "events_per_year_q4_oracle": {int(k): int(v) for k, v in cap_q4.items()},
                     "events_per_ticker_year_q4_test": float(
                         len(window_split(q4o, *TEST)) / max(1, window_split(q4o, *TEST)["ticker"].nunique())
                         / (TEST[1] - TEST[0] + 1))},
        "extraordinary_discarded": ext,
        "trials": {"logged_now": n_logged, "n3_exdiv_family": n_family},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=float, ensure_ascii=False), encoding="utf-8")

    def row(name, s):
        return (f"| {name} | {s.get('n', 0)} | {s.get('n_q', 0)} | {pct(s.get('mean'))} | "
                f"{pct(s.get('mean_q'))} | {s.get('t', float('nan')):.2f} |")

    md = [f"# N3 · captura del ex-dividendo — veredicto: **{ver}**", "", f"_{why}_", "",
          "Compra Close[p−5], venta Close[p] (+dividendo); costo 0.30%/lado; retención 10% (supuesto sin verificar).",
          f"Umbral Q4 de yield (p75 2008-17): oráculo {cut_o:.3%}, causal {cut_c:.3%}.", "",
          "## Sanidad (bruto, todos los eventos 2004-2026)", "",
          "| serie | n | trim. | media/evento | media/trim. | t |", "|---|---|---|---|---|---|",
          row("B bruto (vs EW)", san_B), row("A bruto (largo-solo)", san_A), "",
          "## Criterio (A, cuartil alto, neto con retención, 2018-2026)", "",
          "| fecha | media test | t test | n trim. | media 2008-17 | cumple |", "|---|---|---|---|---|---|"]
    for nm, p in (("causal", pass_c), ("oráculo", pass_o)):
        md.append(f"| {nm} | {pct(p['mean_test'])} | {p['t_test']:.2f} | {p['n_q_test']} | "
                  f"{pct(p['mean_train'])} | {'sí' if p['all'] else 'no'} |")
    md += ["", "## Detalle por variante (2018-2026)", "",
           "| variante | n | trim. | media/evento | media/trim. | t |", "|---|---|---|---|---|---|"]
    for nm, rep, sc in (("oráculo todos", rep_o, "all"), ("oráculo Q4", rep_o, "q4"),
                        ("causal todos", rep_c, "all"), ("causal Q4", rep_c, "q4")):
        for ver_ in ("A", "B"):
            for kind in ("gross", "net_tax", "net_notax"):
                md.append(row(f"{nm} {ver_} {kind}", rep[f"{sc}|test|{ver_}_{kind}"]))
    md += ["", "## Acierto de la fecha causal", "",
           f"- Operaciones previstas con evento real a ±2 días hábiles: {hit_by_trade:.1%}",
           f"- Eventos reales cubiertos por una previsión a ±2: {hit_by_event:.1%}",
           f"- Con acierto: n={by_hit['hit_pm2']['n']}, bruto A {pct(by_hit['hit_pm2']['A_gross_mean'])}; "
           f"con fallo: n={by_hit['miss']['n']}, bruto A {pct(by_hit['miss']['A_gross_mean'])}",
           f"- Capturaron algún dividendo real: {by_hit['captured_dividend']['share']:.1%}", "",
           "## Eventos por año (oráculo)", "", f"Todos: {result['capacity']['events_per_year_all']}", "",
           f"Q4: {result['capacity']['events_per_year_q4_oracle']}", "",
           "## Extraordinarios descartados (yield > 15%)", ""]
    md += [f"- {e['ticker']} {e['ex_date']}: yield {e['yield']:.1%}, bruto {pct(e['gross'])}" for e in ext] or ["- ninguno"]
    md += ["", f"Ensayos: {n_logged} escritos ahora; N familia n3_exdiv = {n_family}."]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"veredicto: {ver} — {why}")
    print(f"sanidad B bruto: media/evento {pct(san_B['mean'])}, media/trim {pct(san_B['mean_q'])}, "
          f"t {san_B['t']:.2f}, n_q {san_B['n_q']}, n {san_B['n']}")
    print(f"sanidad A bruto: media/evento {pct(san_A['mean'])}, t {san_A['t']:.2f}")
    print("causal:", {k: v for k, v in pass_c.items()})
    print("oraculo:", {k: v for k, v in pass_o.items()})
    print(f"acierto ±2: trades {hit_by_trade:.1%}, eventos {hit_by_event:.1%}; ensayos N={n_family}")


if __name__ == "__main__":
    main()
