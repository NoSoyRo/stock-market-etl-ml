"""
M4 · Estudio de la reversión de 5 días: ¿sobrevive a la ejecución real?

IC por fecha (Spearman, paso 5 = semanal sin solape) de ``score = -señal`` contra el retorno
futuro de 5 días, para 9 señales × 4 entradas × 8 subconjuntos, más una sanidad de portafolio
(top-5 de las peores de ret_5d, rotación real) y la puerta de decisión escrita de antemano.

Uso:
    .venv/bin/python scripts/run_reversal_study.py              # completo
    .venv/bin/python scripts/run_reversal_study.py --use-mask   # además aplica la máscara M3a

Universo: todos los ``data/market/*.MX.parquet`` salvo NAFTRAC*/FIBRA*, con >= 1500 filas,
desde 2008 (réplica de main). Retornos en fracciones; periodo = 5 días hábiles ⇒ 50.4/año.

Puerta (fijada antes de mirar): IC(Open t+1 → Close t+6) de -ret_5d >= 0.03 en al menos
4 de 5 bloques (5 bloques contiguos con igual nº de semanas) Y >= 0.03 en el tercio más líquido.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.quality import load_quality_mask  # noqa: E402
from src.data.total_return import ex_div_flag, normalize_dividends  # noqa: E402
from src.ml.reversal import (  # noqa: E402
    ENTRY_NAMES,
    SIGNAL_NAMES,
    abnormal_volume,
    adv,
    block_means,
    build_signals,
    forward_returns,
    tercile_by_date,
    to_panel,
    weekly_bottom_weights,
)
from src.ml.shared.evaluators.costs import (  # noqa: E402
    breakeven_cost_per_side,
    cost_sensitivity,
    net_returns,
    turnover,
)
from src.ml.shared.evaluators.cross_section import (  # noqa: E402
    equal_weight_returns,
    ic_by_date,
    ic_summary,
    percentile_in,
    random_topk_returns,
    topk_returns,
)

EVAL = ROOT / "data" / "eval"
MARKET = ROOT / "data" / "market"
DIVS = ROOT / "data" / "dividends"
HORIZON, STEP, PPY, TOP_K = 5, 5, 252 / 5, 5
COST_SIDE = 0.003
MIN_ALL, MIN_SUB = 12, 6
SEED = 0
GATE_IC, GATE_BLOCKS = 0.03, 4
START = "2008"
SUBSETS = (
    "todo", "liq_bajo", "liq_medio", "liq_alto",
    "volanor_bajo", "volanor_medio", "volanor_alto", "sin_exdiv",
)
SIGNAL_MAIN = "ret_5d"
ENTRY_GATE = "open_close"


# ── carga ────────────────────────────────────────────────────────────────────

def load_universe(use_mask: bool):
    """Matrices anchas Open/Close/Volume y dividendos por ticker."""
    mask = load_quality_mask() if use_mask else None
    O, C, V, divs = {}, {}, {}, {}
    for f in sorted(MARKET.glob("*.MX.parquet")):
        if f.stem.startswith(("NAFTRAC", "FIBRA")):
            continue
        df = pd.read_parquet(f)
        df = df[~df.index.duplicated()].sort_index()
        if len(df) < 1500:
            continue
        ok = (df["Volume"] > 0) & (df["Close"] > 0) & (df["Open"] > 0)
        if mask is not None:
            m = mask[mask["ticker"] == f.stem].drop_duplicates("date", keep="last").set_index("date")["valid"]
            ok &= m.reindex(df.index).fillna(True).astype(bool)
        O[f.stem], C[f.stem], V[f.stem] = df["Open"].where(ok), df["Close"].where(ok), df["Volume"].where(ok)
        p = DIVS / f"{f.stem}.parquet"
        divs[f.stem] = normalize_dividends(pd.read_parquet(p)["dividend"]) if p.exists() else normalize_dividends(None)
    O, C, V = (pd.DataFrame(x).loc[START:] for x in (O, C, V))
    return O, C, V, divs


def exdiv_wide(index: pd.DatetimeIndex, cols, divs, window: int) -> pd.DataFrame:
    """True si el ticker tiene un ex-dividendo en (t, t+window] (en filas de la malla común)."""
    out = {c: ex_div_flag(index, divs[c], window).astype(float) for c in cols}
    return pd.DataFrame(out, index=index)


# ── IC ───────────────────────────────────────────────────────────────────────

def ic_row(panel: pd.DataFrame, score: str, ret: str, min_names: int) -> tuple[dict, pd.Series]:
    """Resumen del IC semanal (paso 5) + los IC semanales para sacar bloques."""
    ic = ic_by_date(panel, score, ret, min_names=min_names).iloc[::STEP]
    s = ic_summary(ic, step=1, block=4, seed=SEED)
    return {k: (float(v) if k != "n" else int(v)) for k, v in s.items()}, ic


def subset_panel(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "todo":
        return panel
    if name == "sin_exdiv":
        return panel[panel["exdiv"] == 0]
    kind, lvl = name.split("_")
    col = "liq_t" if kind == "liq" else "va_t"
    return panel[panel[col] == {"bajo": 0, "medio": 1, "alto": 2}[lvl]]


def run_ic_table(panel: pd.DataFrame) -> tuple[list[dict], dict]:
    """Tabla IC: señal × entrada × subconjunto; devuelve filas y bloques de la combinación de la puerta."""
    rows, blocks = [], {}
    for sub in SUBSETS:
        sp = subset_panel(panel, sub)
        mn = MIN_ALL if sub == "todo" else MIN_SUB
        for sig in SIGNAL_NAMES:
            for ent in ENTRY_NAMES:
                s, ic = ic_row(sp, f"s_{sig}", f"f_{ent}", mn)
                rows.append({"señal": sig, "entrada": ent, "subconjunto": sub, **s})
                if sig == SIGNAL_MAIN and ent in ("close_close", "open_close", "open_open", "skip_close"):
                    blocks[f"{sub}|{ent}"] = block_means(ic, 5)
    return rows, blocks


# ── portafolio ───────────────────────────────────────────────────────────────

def portfolio_sanity(score_w: pd.DataFrame, fwd_w: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Top-5 de las peores de ret_5d: bruto/neto con rotación real vs equiponderado y top-5 aleatorio."""
    w = weekly_bottom_weights(score_w, TOP_K, STEP)
    f = fwd_w.reindex(w.index)
    bad = ((w > 0) & f.isna()).any(axis=1)
    w, f = w[~bad], f[~bad]
    gross = (w * f.fillna(0.0)).sum(axis=1)
    tov = turnover(w)
    net = net_returns(gross, tov, COST_SIDE)

    reb = panel[panel.index.isin(w.index)]
    ew = equal_weight_returns(reb, "f_open_close", min_names=MIN_ALL).reindex(gross.index)
    rnd = random_topk_returns(reb, "f_open_close", TOP_K, n_draws=1000, seed=SEED, min_names=MIN_ALL)
    rnd_mean = rnd.mean(axis=0)
    # verificación cruzada con la función genérica (difiere solo por desempates)
    tk = topk_returns(reb, "s_ret_5d", "f_open_close", TOP_K, min_names=MIN_ALL).reindex(gross.index)
    exc = (gross - ew).dropna()
    be = breakeven_cost_per_side(gross, tov)
    return {
        "periodos": int(len(gross)),
        "bruto_medio_pct": float(gross.mean() * 100),
        "neto_medio_pct_a_0.30": float(net.mean() * 100),
        "ew_medio_pct": float(ew.mean() * 100),
        "aleatorio_medio_pct": float(rnd_mean.mean() * 100),
        "percentil_bruto_vs_aleatorio": float(percentile_in(float(gross.mean()), rnd_mean.to_numpy())),
        "exceso_bruto_vs_ew_pct": float(exc.mean() * 100),
        "exceso_t": float(exc.mean() / (exc.std(ddof=1) / np.sqrt(len(exc)))),
        "exceso_neto_vs_ew_pct": float((net - ew).dropna().mean() * 100),
        "rotacion_media": float(tov.mean()),
        "costo_equilibrio_por_lado_pct": float(be * 100),
        "bruto_cubre_costo_0.30": bool(be >= COST_SIDE),
        "max_dif_vs_topk_returns": float((gross - tk).abs().max()),
        "sensibilidad": cost_sensitivity(gross, tov, periods_per_year=PPY).reset_index(drop=True).to_dict("records"),
    }


# ── puerta ───────────────────────────────────────────────────────────────────

def apply_gate(rows: list[dict], blocks: dict) -> dict:
    b = blocks[f"todo|{ENTRY_GATE}"]
    n_ok = sum(x["mean"] >= GATE_IC for x in b)
    liq = next(r for r in rows if (r["señal"], r["entrada"], r["subconjunto"]) == (SIGNAL_MAIN, ENTRY_GATE, "liq_alto"))
    c1 = n_ok >= GATE_BLOCKS
    c2 = liq["mean"] >= GATE_IC
    return {
        "criterio_1_bloques": {"requisito": f">= {GATE_IC} en {GATE_BLOCKS} de 5 bloques", "bloques_ok": n_ok,
                               "medias": [round(x["mean"], 4) for x in b], "pasa": bool(c1)},
        "criterio_2_tercio_liquido": {"requisito": f"IC medio >= {GATE_IC}", "ic": liq["mean"], "t": liq["t"],
                                      "pasa": bool(c2)},
        "pasa_puerta": bool(c1 and c2),
    }


# ── reporte ──────────────────────────────────────────────────────────────────

def render_md(res: dict) -> str:
    rows = pd.DataFrame(res["ic"])
    L = [f"# M4 · Estudio de reversión de 5 días ({res['fecha']})", "",
         f"Universo: {res['n_tickers']} tickers .MX, {res['desde']} → {res['hasta']}; máscara M3a: {res['use_mask']}. "
         f"Score = −señal; IC Spearman semanal (paso 5). Ensayos: {res['ensayos']}.", ""]

    def tabla(sub: str, titulo: str, sigs=SIGNAL_NAMES):
        d = rows[rows["subconjunto"] == sub].set_index(["señal", "entrada"])
        L.extend([f"## {titulo}", "", "| señal | " + " | ".join(ENTRY_NAMES) + " |", "|---|" + "---|" * len(ENTRY_NAMES)])
        for s in sigs:
            cells = [f"{d.loc[(s, e), 'mean']:+.4f} (t={d.loc[(s, e), 't']:.1f})" for e in ENTRY_NAMES]
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L.append("")

    tabla("todo", "Todo el universo")
    for sub in SUBSETS[1:]:
        tabla(sub, f"Subconjunto: {sub} (solo ret_5d, ret_1d, resid_5d)", ("ret_5d", "ret_1d", "resid_5d"))
    L += ["## Bloques (ret_5d, todo)", "", "| entrada | " + " | ".join(f"B{i+1}" for i in range(5)) + " |", "|---|---|---|---|---|---|"]
    for e in ENTRY_NAMES:
        L.append(f"| {e} | " + " | ".join(f"{x['mean']:+.3f}" for x in res["bloques"][f"todo|{e}"]) + " |")
    L.append("")
    b0 = res["bloques"]["todo|close_close"]
    L.append("Bloques: " + "; ".join(f"B{i+1} {x['desde']}→{x['hasta']}" for i, x in enumerate(b0)) + "\n")
    g, p = res["puerta"], res["portafolio"]
    L += ["## Puerta", "",
          f"- Criterio 1 (bloques): {'PASA' if g['criterio_1_bloques']['pasa'] else 'NO PASA'} — "
          f"{g['criterio_1_bloques']['bloques_ok']}/5 con IC≥{GATE_IC}; medias {g['criterio_1_bloques']['medias']}",
          f"- Criterio 2 (tercio más líquido): {'PASA' if g['criterio_2_tercio_liquido']['pasa'] else 'NO PASA'} — "
          f"IC {g['criterio_2_tercio_liquido']['ic']:+.4f} (t={g['criterio_2_tercio_liquido']['t']:.1f})",
          f"- **Puerta: {'PASA' if g['pasa_puerta'] else 'NO PASA'}**", "",
          "## Portafolio top-5 de las peores de ret_5d (Open t+1 → Close t+6)", ""]
    for k in ("periodos", "bruto_medio_pct", "neto_medio_pct_a_0.30", "ew_medio_pct", "aleatorio_medio_pct",
              "percentil_bruto_vs_aleatorio", "exceso_bruto_vs_ew_pct", "exceso_t", "exceso_neto_vs_ew_pct",
              "rotacion_media", "costo_equilibrio_por_lado_pct", "bruto_cubre_costo_0.30", "max_dif_vs_topk_returns"):
        v = p[k]
        L.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    L += ["", "| costo/lado | neto medio | Sharpe | retorno total |", "|---|---|---|---|"]
    for r in p["sensibilidad"]:
        vals = [r["cost_per_side"], r["mean_net"], r["sharpe_net"], r["total_return_net"]]
        L.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--use-mask", action="store_true", help="aplica la máscara de calidad M3a (valid)")
    args = ap.parse_args()

    O, C, V, divs = load_universe(args.use_mask)
    print(f"{C.shape[1]} tickers | {C.index.min().date()} → {C.index.max().date()} | máscara={args.use_mask}")

    sig = build_signals(O, C)
    fwd = forward_returns(O, C, HORIZON)
    liq_t = tercile_by_date(adv(C, V))
    va_t = tercile_by_date(abnormal_volume(V))
    exd = exdiv_wide(C.index, C.columns, divs, HORIZON + 1)

    wide = {f"s_{k}": -v for k, v in sig.items()}  # score = −señal
    wide.update({f"f_{k}": v for k, v in fwd.items()})
    wide.update({"liq_t": liq_t, "va_t": va_t, "exdiv": exd})
    panel = to_panel(wide)

    rows, blocks = run_ic_table(panel)
    ensayos = len(SIGNAL_NAMES) * len(ENTRY_NAMES) * len(SUBSETS)
    port = portfolio_sanity(wide["s_ret_5d"], fwd[ENTRY_GATE], panel)
    res = {
        "fecha": datetime.now().strftime("%Y-%m-%d"), "use_mask": bool(args.use_mask),
        "n_tickers": int(C.shape[1]), "tickers": list(C.columns),
        "desde": str(C.index.min().date()), "hasta": str(C.index.max().date()),
        "ensayos": ensayos, "ic": rows, "bloques": blocks, "puerta": apply_gate(rows, blocks),
        "portafolio": port,
    }
    suf = "_mask" if args.use_mask else ""
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / f"reversal_study{suf}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    (EVAL / f"reversal_study{suf}.md").write_text(render_md(res))
    md = render_md(res)
    print(md)


if __name__ == "__main__":
    main()
