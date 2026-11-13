"""
M1d · Tabla única de M1: métricas corregidas, IC por fecha, baselines, costo por
rotación real, atribución del Sharpe bruto y sesgo de supervivencia grueso.

Uso:
    .venv/bin/python scripts/run_m1_report.py            # completo
    .venv/bin/python scripts/run_m1_report.py --fast     # subconjunto de tickers, pocos sorteos

Convenciones: ``fwd_ret`` del panel está en % (aquí se pasa a fracciones); periodo = 5 días
hábiles ⇒ ``periods_per_year = 50.4``; semilla fija en todo; ``cost_mode`` siempre explícito.
Todo número del .md sale del .json (el .md se genera desde el dict).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backtest_topk import (  # noqa: E402
    HORIZON,
    MIN_NAMES,
    SEED,
    _naftrac_return,
    build_test_panel,
    load_universe_tickers,
    run_topk_backtest,
)
from scripts.evaluate_portfolio import build_portfolio_export  # noqa: E402
from src.ml.shared.evaluators.cross_section import (  # noqa: E402
    auc_by_date,
    equal_weight_returns,
    ic_by_date,
    ic_summary,
    percentile_in,
    random_topk_returns,
    topk_returns,
)
from src.ml.shared.evaluators.metrics import Metrics  # noqa: E402

EVAL = ROOT / "data" / "eval"
PPY = 252 / HORIZON  # 50.4
TOP_K = 5
FAST_TICKERS = 16


# ── funciones puras ──────────────────────────────────────────────────────────

def rebalance_panel(test: pd.DataFrame, horizon: int = HORIZON, min_names: int = MIN_NAMES) -> pd.DataFrame:
    """Panel de rebalanceo: la última fila de cada ticker por semana (igual que el backtest).

    Índice = última fecha de la semana; ``ret`` = ``fwd_ret`` en fracción. Solo semanas con
    ≥ ``min_names`` tickers; se toma una de cada ``round(horizon/5)`` semanas.
    """
    t = test.copy()
    t["_week"] = t.index.to_period("W")
    step = max(1, round(horizon / 5))
    partes = []
    for wi, week in enumerate(sorted(t["_week"].unique())):
        if wi % step != 0:
            continue
        last = t[t["_week"] == week].sort_index().groupby("ticker").tail(1)
        if len(last) < min_names:
            continue
        last = last.copy()
        last.index = pd.DatetimeIndex([last.index.max()] * len(last))
        partes.append(last)
    rp = pd.concat(partes).drop(columns="_week")
    rp["ret"] = rp["fwd_ret"] / 100.0
    return rp


def mark_backtest_picks(rp: pd.DataFrame, score_col: str, k: int) -> tuple[pd.Series, int]:
    """Marca los k picks por fecha con la MISMA llamada que el backtest (``sort_values`` desc).

    Devuelve (máscara booleana alineada a ``rp``, nº de fechas con empate en la frontera k/k+1).
    Con empates (p. ej. calibración isotónica) el orden de ``sort_values`` decide quién entra.
    """
    mask = np.zeros(len(rp), dtype=bool)
    empates = 0
    for _, idx in rp.groupby(level=0).indices.items():
        sub = rp.iloc[idx].reset_index(drop=True)
        orden = sub.sort_values(score_col, ascending=False)
        mask[idx[orden.head(k).index.to_numpy()]] = True
        if len(orden) > k and orden[score_col].iloc[k - 1] == orden[score_col].iloc[k]:
            empates += 1
    return pd.Series(mask, index=rp.index), empates


def sharpe(r, ppy: float = PPY) -> float:
    """Sharpe anualizado por periodo (sin tasa libre); NaN si no hay dispersión."""
    r = np.asarray(r, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def sharpe_cols(m: pd.DataFrame, ppy: float = PPY) -> np.ndarray:
    """Sharpe anualizado de cada columna de una matriz periodo × sorteo."""
    x = m.to_numpy(dtype=float)
    sd = x.std(axis=0, ddof=1)
    return np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1) * np.sqrt(ppy), np.nan)


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS con constante: devuelve alfa, betas, sus t y R². ``X`` sin constante (n × p)."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float).reshape(len(y), -1)])
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    res = y - X @ coef
    dof = len(y) - X.shape[1]
    s2 = float(res @ res) / dof
    se = np.sqrt(np.diag(xtx_inv) * s2)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "alpha": float(coef[0]),
        "alpha_t": float(coef[0] / se[0]),
        "betas": [float(b) for b in coef[1:]],
        "beta_t": [float(b / s) for b, s in zip(coef[1:], se[1:])],
        "r2": float(1 - (res @ res) / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y)),
    }


def vol_matched_random(
    rp: pd.DataFrame,
    pick_col: str,
    k: int,
    *,
    vol_col: str = "atr_pct",
    n_draws: int = 1000,
    n_keep: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Top-k aleatorio emparejado por volatilidad (ex-ante, ``vol_col``) al canasto del modelo.

    En cada fecha: se sortean ``n_draws`` canastos de k nombres y se conservan los
    ``n_keep`` cuya vol media queda más cerca de la de los k nombres marcados en ``pick_col``.
    Devuelve fecha × ``n_keep`` (la columna j une el j-ésimo más cercano de cada fecha).
    """
    rng = np.random.default_rng(seed)
    fechas, filas = [], []
    for d, g in rp.groupby(level=0):
        g = g.dropna(subset=[vol_col, "ret"])
        if len(g) < max(k, 2) or not g[pick_col].any():
            continue
        v, r = g[vol_col].to_numpy(), g["ret"].to_numpy()
        objetivo = v[g[pick_col].to_numpy()].mean()
        idx = rng.random((n_draws, len(g))).argpartition(k - 1, axis=1)[:, :k]
        cerca = np.argsort(np.abs(v[idx].mean(axis=1) - objetivo))[:n_keep]
        filas.append(r[idx[cerca]].mean(axis=1))
        fechas.append(d)
    return pd.DataFrame(np.vstack(filas), index=pd.Index(fechas))


def yearly_table(ew: pd.Series, naf: pd.Series) -> list[dict]:
    """Retorno compuesto por año de EW (lista actual) y NAFTRAC, y su diferencia."""
    df = pd.DataFrame({"ew": ew, "naf": naf}).dropna()
    out = []
    for y, g in df.groupby(df.index.year):
        e, n = float((1 + g["ew"]).prod() - 1), float((1 + g["naf"]).prod() - 1)
        out.append({"year": int(y), "n": int(len(g)), "ew": e, "naftrac": n, "diff": e - n})
    return out


def summarize_series(r: pd.Series | np.ndarray) -> dict:
    """Media por periodo, Sharpe anualizado y retorno total de una serie de retornos."""
    a = np.asarray(r, dtype=float)
    return {
        "mean": float(a.mean()),
        "sharpe": sharpe(a),
        "total": float((1 + a).prod() - 1),
        "n": int(len(a)),
    }


def _num(x):
    """Convierte numpy/NaN a tipos JSON."""
    if isinstance(x, dict):
        return {k: _num(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def compact_backtest(bt: dict) -> dict:
    """Resumen de un backtest sin curvas ni series largas."""
    s = bt["strategy"]
    return {
        "cost_mode": bt["config"]["cost_mode"],
        "sharpe_gross": s["sharpe_gross"],
        "sharpe_net": s["sharpe_net"],
        "total_return_gross": s["total_return_gross"],
        "total_return_net": s["total_return_net"],
        "avg_period_gross_pct": s["avg_period_gross_pct"],
        "avg_period_net_pct": s["avg_period_net_pct"],
        "max_drawdown": s["max_drawdown"],
        "calmar_ratio": s["calmar_ratio"],
        "naftrac_sharpe": bt["benchmark_naftrac"]["sharpe_ratio"],
        "naftrac_total_return": bt["benchmark_naftrac"]["total_return"],
        "n_periods": s["n_periods"],
        "turnover_mean": bt["turnover_mean"],
        "breakeven_cost_per_side": bt["breakeven_cost_per_side"],
        "cost_sensitivity": bt["cost_sensitivity"],
    }


def _read_json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _read_json_head(rel: str) -> dict | None:
    """Versión de un JSON en git HEAD (el "antes" estable, aunque ya se haya regenerado)."""
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


# ── armado del reporte ───────────────────────────────────────────────────────

def ic_block(test: pd.DataFrame, score_col: str, y_col: str, n_boot: int) -> dict:
    """IC (Spearman vs fwd_ret) y AUC por fecha de un score, con ic_summary (step = horizonte)."""
    ic = ic_by_date(test, score_col, "fwd_ret", min_names=MIN_NAMES)
    auc = auc_by_date(test, y_col, score_col, min_names=MIN_NAMES)
    return {
        "ic": ic_summary(ic, step=HORIZON, block=4, n_boot=n_boot, seed=SEED),
        "auc_by_date": ic_summary(auc - 0.5, step=HORIZON, block=4, n_boot=n_boot, seed=SEED),
        "auc_mean": float(auc.mean()),
        "n_dates_daily": int(len(ic)),
    }


def build_report(fast: bool, n_draws: int, n_boot: int) -> tuple[dict, dict, dict]:
    """Corre todo. Devuelve (reporte, bt_turnover, bt_fixed)."""
    t0 = time.time()
    tickers = load_universe_tickers("bmv")
    if fast:
        tickers = dict(list(tickers.items())[:FAST_TICKERS])

    # Dos modelos: objetivo absoluto (y) y relativo (y_rel). 2 entrenamientos.
    test_up, info_up = build_test_panel("bmv", "p_up", HORIZON, "open", tickers=tickers)
    test_rel, info_rel = build_test_panel("bmv", "p_rel", HORIZON, "open", tickers=tickers)

    # Backtest K=5, p_up, equal, min_p_up=0, cost_mode EXPLÍCITO en ambos modos.
    common = dict(horizon=HORIZON, top_k=TOP_K, score_mode="p_up", weight_mode="equal", min_p_up=0.0)
    bt_fixed = run_topk_backtest(test_up, cost_mode="fixed", **common)
    bt_turn = run_topk_backtest(test_up, cost_mode="turnover", **common)
    for bt, info in ((bt_fixed, info_up), (bt_turn, info_up)):
        bt["model"], bt["universe"] = info, "bmv"
        bt["generated_at"] = datetime.now().isoformat(timespec="seconds")

    # 2. IC y AUC por fecha (días hábiles del test; step = horizonte).
    ics = {
        "p_up": ic_block(test_up, "p_up", "y", n_boot),
        "p_rel": ic_block(test_rel, "p_rel", "y_rel", n_boot),
    }

    # 3. Baselines sobre las MISMAS fechas de rebalanceo del backtest.
    rp = rebalance_panel(test_up)
    rp["neg_ret5"] = -rp["ret_5d"]
    rp["is_pick"], n_empates = mark_backtest_picks(rp, "p_up", TOP_K)
    top5 = rp[rp["is_pick"]].groupby(level=0)["ret"].mean().rename("topk")
    ew = equal_weight_returns(rp, "ret", min_names=MIN_NAMES)
    peores = topk_returns(rp, "neg_ret5", "ret", TOP_K, min_names=MIN_NAMES)
    naf = pd.Series([_naftrac_return(d, HORIZON) for d in top5.index], index=top5.index)
    chequeo = bool(
        len(top5) == len(bt_turn["period_returns"]["gross"])
        and np.allclose(top5.to_numpy(), bt_turn["period_returns"]["gross"], atol=1e-9)
    )
    rnd = random_topk_returns(rp, "ret", TOP_K, n_draws=n_draws, seed=SEED, min_names=MIN_NAMES)
    rnd_sh = sharpe_cols(rnd)
    rnd_mean = rnd.mean(axis=0)
    base = {
        "top5_modelo": summarize_series(top5),
        "equiponderado": summarize_series(ew),
        "top5_aleatorio": {
            **summarize_series(rnd.mean(axis=1)),
            "n_draws": n_draws,
            "sharpe_mean": float(np.nanmean(rnd_sh)),
            "sharpe_p5": float(np.nanpercentile(rnd_sh, 5)),
            "sharpe_p95": float(np.nanpercentile(rnd_sh, 95)),
            "model_percentile_sharpe": percentile_in(sharpe(top5), rnd_sh),
            "model_percentile_mean": percentile_in(float(top5.mean()), rnd_mean),
        },
        "cinco_peores_ret5d": summarize_series(peores),
        "ic_neg_ret5d": ic_summary(
            ic_by_date(test_up.assign(neg=-test_up["ret_5d"]), "neg", "fwd_ret", min_names=MIN_NAMES),
            step=HORIZON, block=4, n_boot=n_boot, seed=SEED),
        "naftrac": summarize_series(naf),
        "top5_menos_ew": ols(top5.to_numpy() - ew.to_numpy(), np.zeros((len(top5), 0))),
        "check_top5_igual_backtest": chequeo,
        "fechas_con_empate_en_frontera_top5": n_empates,
    }

    # 4. Rotación real y costos (M1c).
    cost = {"fixed": compact_backtest(bt_fixed), "turnover": compact_backtest(bt_turn)}

    # 5. Atribución del Sharpe bruto.
    d_idx = top5.index
    pick_vol = rp[rp["is_pick"]].groupby(level=0)["atr_pct"].mean()
    uni_vol = rp.groupby(level=0)["atr_pct"].mean()
    reg_ew = ols(top5.to_numpy(), ew.to_numpy())
    reg_both = ols(top5.to_numpy(), np.column_stack([ew.to_numpy(), naf.to_numpy()]))
    vm = vol_matched_random(rp, "is_pick", TOP_K, n_draws=n_draws, n_keep=min(200, n_draws), seed=SEED)
    vm_sh = sharpe_cols(vm)
    attrib = {
        "atr_pct_top5": float(pick_vol.mean()),
        "atr_pct_universo": float(uni_vol.mean()),
        "atr_ratio": float(pick_vol.mean() / uni_vol.mean()),
        "vol_anual_top5": float(top5.std(ddof=1) * np.sqrt(PPY)),
        "vol_anual_ew": float(ew.std(ddof=1) * np.sqrt(PPY)),
        "vol_anual_naftrac": float(naf.std(ddof=1) * np.sqrt(PPY)),
        "beta_top5_vs_ew": reg_ew["betas"][0],
        "beta_top5_vs_naftrac": ols(top5.to_numpy(), naf.to_numpy())["betas"][0],
        "reg_sobre_ew": reg_ew,
        "reg_sobre_ew_y_naftrac": reg_both,
        "alfa_anual_vs_ew": reg_ew["alpha"] * PPY,
        "vol_matched": {
            "n_cols": int(vm.shape[1]),
            "sharpe_mean": float(np.nanmean(vm_sh)),
            "sharpe_p5": float(np.nanpercentile(vm_sh, 5)),
            "sharpe_p95": float(np.nanpercentile(vm_sh, 95)),
            "model_percentile_sharpe": percentile_in(sharpe(top5), vm_sh),
        },
        "nota": "betas de muestra completa: descriptivas, no una señal operable",
    }

    # 6. Sesgo de supervivencia grueso.
    surv = {"por_anio": yearly_table(ew, naf)}
    surv["diff_media_anual"] = float(np.mean([r["diff"] for r in surv["por_anio"]]))

    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fast": fast,
        "config": {
            "universe": "bmv", "top_k": TOP_K, "score": "p_up", "weight": "equal",
            "min_p_up": 0.0, "horizon": HORIZON, "periods_per_year": PPY, "seed": SEED,
            "cost_mode_canonico_archivos": "turnover",
            "cost_per_side": Metrics.DEFAULT_COMMISSION_PCT + Metrics.DEFAULT_SLIPPAGE_PCT,
            "n_draws": n_draws, "n_boot": n_boot,
        },
        "datos": {
            "n_tickers": int(test_up["ticker"].nunique()),
            "tickers": sorted(test_up["ticker"].unique().tolist()),
            "modelo_p_up": info_up, "modelo_p_rel": info_rel,
            "rebalanceos": int(len(top5)),
            "primer_rebalanceo": str(top5.index.min().date()),
            "ultimo_rebalanceo": str(top5.index.max().date()),
        },
        "ic_por_fecha": ics,
        "baselines": base,
        "costos": cost,
        "atribucion": attrib,
        "supervivencia": surv,
        "ensayos": {
            "modelos_entrenados": 2,
            "corridas_backtest": 2,
            "configuraciones": 1,
            "nota": "p_up y p_rel; backtest fixed y turnover; sin búsqueda de hiperparámetros",
        },
        "segundos": round(time.time() - t0, 1),
    }
    return _num(rep), bt_turn, bt_fixed


# ── markdown (todo sale del dict) ────────────────────────────────────────────

def render_md(r: dict) -> str:
    """Genera m1_report.md a partir del dict del reporte (≤ 60 líneas)."""
    b, c, a, s = r["baselines"], r["costos"], r["atribucion"], r["supervivencia"]
    d, cfg = r["datos"], r["config"]
    f, t = c["fixed"], c["turnover"]
    ra, rr = r["ic_por_fecha"]["p_up"], r["ic_por_fecha"]["p_rel"]
    pct = lambda x: f"{x * 100:+.2f}%"  # noqa: E731
    row = lambda n, m, sh, tot: f"| {n} | {m * 100:+.3f}% | {sh:+.2f} | {pct(tot)} |"  # noqa: E731
    ale, tm = b["top5_aleatorio"], b["top5_modelo"]
    vm, reg = a["vol_matched"], a["reg_sobre_ew"]
    L = [
        "# M1 · Tabla única (modelo top-5 vs baselines)",
        f"Test {d['modelo_p_up']['test_start']} → {d['modelo_p_up']['test_end']} · {d['rebalanceos']} rebalanceos de "
        f"{HORIZON} días · {d['n_tickers']} tickers · semilla {cfg['seed']} · AUC p_up {d['modelo_p_up']['auc']}"
        + (" · **MODO FAST (subconjunto)**" if r["fast"] else ""),
        "",
        "| Estrategia (por periodo de 5d) | Media | Sharpe anual | Total |",
        "|---|---|---|---|",
        row("Top-5 del modelo, bruto", tm["mean"], tm["sharpe"], tm["total"]),
        f"| Top-5 neto, costo fijo (100% rotación) | {f['avg_period_net_pct']:+.3f}% | {f['sharpe_net']:+.2f} | {pct(f['total_return_net'])} |",
        f"| Top-5 neto, rotación real ({t['turnover_mean'] * 100:.0f}%) | {t['avg_period_net_pct']:+.3f}% | {t['sharpe_net']:+.2f} | {pct(t['total_return_net'])} |",
        row("Equiponderado del universo", b["equiponderado"]["mean"], b["equiponderado"]["sharpe"], b["equiponderado"]["total"]),
        row(f"Top-5 aleatorio (media de {ale['n_draws']})", ale["mean"], ale["sharpe_mean"], ale["total"]),
        row("5 peores del ret_5d (sin modelo)", b["cinco_peores_ret5d"]["mean"], b["cinco_peores_ret5d"]["sharpe"], b["cinco_peores_ret5d"]["total"]),
        row("NAFTRAC", b["naftrac"]["mean"], b["naftrac"]["sharpe"], b["naftrac"]["total"]),
        "",
        f"IC por fecha (Spearman, step={HORIZON}): p_up {ra['ic']['mean']:+.4f} (t={ra['ic']['t']:.2f}, "
        f"IC95 [{ra['ic']['ci_lo']:+.4f}, {ra['ic']['ci_hi']:+.4f}], n={ra['ic']['n']}) · "
        f"p_rel {rr['ic']['mean']:+.4f} (t={rr['ic']['t']:.2f}, n={rr['ic']['n']}) · "
        f"−ret_5d {b['ic_neg_ret5d']['mean']:+.4f} (t={b['ic_neg_ret5d']['t']:.2f}).",
        f"AUC medio por fecha: p_up {ra['auc_mean']:.4f} · p_rel {rr['auc_mean']:.4f}.",
        "",
        "## Conclusiones",
    ]
    L.append(
        f"1. **¿Es mercado?** El top-5 aleatorio rinde Sharpe {ale['sharpe_mean']:+.2f} y el equiponderado "
        f"{b['equiponderado']['sharpe']:+.2f}, contra {tm['sharpe']:+.2f} del modelo (percentil {ale['model_percentile_sharpe']:.0f} "
        f"entre sorteos). " + (
            "Casi todo el bruto es mercado y elegir al azar da algo parecido."
            if ale["model_percentile_sharpe"] < 75 else
            "Queda arriba del azar, pero sin llegar al percentil 95: evidencia débil."
            if ale["model_percentile_sharpe"] < 95 else "El modelo queda arriba del azar.")
    )
    L.append(
        f"2. **¿Hay señal?** IC medio {ra['ic']['mean']:+.4f} con t={ra['ic']['t']:.2f}: "
        + ("indistinguible de cero." if abs(ra["ic"]["t"]) < 2 else "distinto de cero, pero chico.")
        + f" Alfa del top-5 sobre el equiponderado: {a['alfa_anual_vs_ew'] * 100:+.1f}% anual (t={reg['alpha_t']:.2f})."
    )
    L.append(
        f"3. **Atribución:** el top-5 elige acciones más nerviosas (ATR {a['atr_ratio']:.2f}× el universo; vol anual "
        f"{a['vol_anual_top5'] * 100:.0f}% vs {a['vol_anual_ew'] * 100:.0f}%), beta {a['beta_top5_vs_ew']:.2f} sobre EW. "
        f"Un top-5 aleatorio con la misma vol da Sharpe {vm['sharpe_mean']:+.2f} (modelo en percentil {vm['model_percentile_sharpe']:.0f})."
    )
    L.append(
        f"4. **Costos:** la rotación real es {t['turnover_mean'] * 100:.0f}%, no 100%: neto {t['sharpe_net']:+.2f} (rotación real) vs "
        f"{f['sharpe_net']:+.2f} (fijo). Costo de equilibrio {t['breakeven_cost_per_side'] * 100:.3f}% por lado; el base es "
        f"{cfg['cost_per_side'] * 100:.2f}%, {'más' if cfg['cost_per_side'] > t['breakeven_cost_per_side'] else 'menos'} que el equilibrio."
    )
    L.append(
        f"5. **Supervivencia:** EW de la lista actual − NAFTRAC = {s['diff_media_anual'] * 100:+.1f} pp en el año promedio. "
        + ("Hay un colchón positivo (sesgo de lista fija + tamaño) dentro del bruto y de los baselines."
           if s["diff_media_anual"] > 0.005 else
           "Colchón ≈ 0 o negativo: con esta estimación gruesa, el sesgo no explica el bruto.")
    )
    L += [
        "",
        "Detalle por año, sensibilidad de costos, tickers y fechas: `data/eval/m1_report.json`.",
        f"Ensayos: {r['ensayos']['modelos_entrenados']} modelos, {r['ensayos']['corridas_backtest']} backtests, "
        f"{r['ensayos']['configuraciones']} configuración (sin búsqueda).",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Reporte integrado M1")
    ap.add_argument("--fast", action="store_true", help="16 tickers, 200 sorteos, 500 bootstrap; no regenera portfolio_metrics/backtest_topk")
    args = ap.parse_args()
    n_draws, n_boot = (200, 500) if args.fast else (1000, 2000)

    # "Antes": versiones versionadas en git HEAD (previas a la corrección y al reentrenamiento).
    pm_head = _read_json_head("data/eval/portfolio_metrics.json") or {}
    antes = {
        "portfolio_metrics": pm_head.get("portfolio_dual"),
        "portfolio_metrics_naftrac": pm_head.get("benchmark_naftrac"),
        "backtest_topk": (_read_json_head("data/eval/backtest_topk.json") or {}).get("strategy"),
    }

    rep, bt_turn, bt_fixed = build_report(args.fast, n_draws, n_boot)
    rep["antes_git_head"] = _num(antes)
    rep["despues"] = _num({
        "fixed": {k: bt_fixed["strategy"][k] for k in ("sharpe_gross", "sharpe_net", "total_return_net")}
        | {"naftrac_sharpe": bt_fixed["benchmark_naftrac"]["sharpe_ratio"]},
        "turnover": {k: bt_turn["strategy"][k] for k in ("sharpe_gross", "sharpe_net", "total_return_net")}
        | {"naftrac_sharpe": bt_turn["benchmark_naftrac"]["sharpe_ratio"]},
    })

    suf = "_fast" if args.fast else ""
    (EVAL / f"m1_report{suf}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    (EVAL / f"m1_report{suf}.md").write_text(render_md(rep))

    if not args.fast:
        # Archivos canónicos regenerados con cost_mode explícito = turnover (ver config.cost_mode).
        exp = build_portfolio_export(bt_turn)
        exp["config"]["cost_mode"] = "turnover"
        exp["config"]["seed"] = SEED
        exp["config"]["min_p_up"] = 0.0
        (EVAL / "portfolio_metrics.json").write_text(json.dumps(exp, indent=2, ensure_ascii=False, default=str))
        (EVAL / "backtest_topk.json").write_text(json.dumps(bt_turn, indent=2, ensure_ascii=False, default=str))

    print((EVAL / f"m1_report{suf}.md").read_text())
    print(f"Listo en {rep['segundos']} s -> {EVAL / f'm1_report{suf}.json'}")


if __name__ == "__main__":
    main()
