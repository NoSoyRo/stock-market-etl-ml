"""
Backtest top-K long-only estilo Ghosh 2022 (KB-081).

Cada rebalance (5d): selecciona top-K por P(up) o score vs mediana cross-seccional.
Long-only, equal-weight o convicción-weight, con costos BMV (KB-003).
Benchmark: buy-and-hold NAFTRAC.

Uso:
    ./venv/bin/python scripts/backtest_topk.py --universe bmv --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.data.quality import apply_mask, load_quality_mask
from src.data.store import UNIVERSE_GROUPS, get as store_get, MACRO_TICKERS
from src.ml.features import (
    FEATURE_COLS,
    add_cross_section_features,
    add_macro_features,
    compute_features,
    compute_signal_score,
    compute_target,
)
from src.ml.ensemble import EnsembleModel
from src.ml.shared.evaluators.costs import (
    breakeven_cost_per_side,
    cost_sensitivity,
    net_returns,
    turnover as compute_turnover,
)
from src.ml.shared.evaluators.metrics import Metrics

HORIZON = 5
TRAIN_RATIO = 0.80
MIN_NAMES = 10
CAPITAL = 100_000
DEFAULT_TOP_K = 2
DEFAULT_MIN_P_UP = 0.56
SEED = 42

ScoreMode = Literal["p_up", "p_rel", "signal"]
WeightMode = Literal["equal", "conviction"]
CostMode = Literal["turnover", "fixed"]
EntryMode = Literal["open", "close"]
SENSITIVITY_COSTS = (0.001, 0.0015, 0.003, 0.005)
MAX_ABS_FWD_RET = 100.0   # guardián de M3c: |fwd_ret| > 100 % es dato basura, no mercado
DROP_RULES = ("mask_invalid", "open_nonpos", "fwd_ret_outlier")


def load_universe_tickers(universe: str) -> dict[str, str]:
    """Carga tickers del universo desde store/universes."""
    tickers = UNIVERSE_GROUPS.get(universe)
    if not tickers:
        raise ValueError(
            f"Universo desconocido: {universe!r}. "
            f"Opciones: {list(UNIVERSE_GROUPS)}"
        )
    return tickers


def clean_ticker_rows(
    feat: pd.DataFrame, raw_open: pd.Series, mask: pd.DataFrame, symbol: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quita las filas basura de un ticker (M3c). Devuelve (filas_conservadas, descartes).

    Reglas, evaluadas por separado (una fila puede caer en varias):
      - ``mask_invalid``: la máscara de M3a la marca no válida (``illiquid`` NO excluye).
      - ``open_nonpos``: Open ≤ 0 en esa fecha (guardián independiente de la máscara).
      - ``fwd_ret_outlier``: ``fwd_ret`` no finito o |fwd_ret| > 100 %.
    Un ``fwd_ret`` NaN (últimas filas sin futuro) no es outlier. ``descartes`` tiene una
    fila por fecha eliminada y una columna booleana por regla, pero solo de filas que SÍ
    habrían entrado al panel (features propias y ``fwd_ret`` no nulos); las de calentamiento
    o sin futuro se eliminan igual pero no se cuentan.
    """
    kept_by_mask = feat.index.isin(apply_mask(feat, mask, symbol).index)
    open_ = raw_open.reindex(feat.index)
    fr = feat["fwd_ret"]
    flags = pd.DataFrame(
        {
            "mask_invalid": ~kept_by_mask,
            "open_nonpos": (open_ <= 0).to_numpy(),
            "fwd_ret_outlier": (fr.notna() & ~(fr.abs() <= MAX_ABS_FWD_RET)).to_numpy(),
        },
        index=feat.index,
    )
    bad = flags.any(axis=1).to_numpy()
    own = [c for c in FEATURE_COLS if c in feat.columns] + ["fwd_ret"]
    # inf cuenta como presente (notna), así que el outlier por Open≈0 sí se reporta
    relevant = feat[own].notna().all(axis=1).to_numpy()
    return feat[~bad], flags[bad & relevant]


def summarize_mask_drops(drops: pd.DataFrame, cut: pd.Timestamp) -> dict:
    """Filas y tickers descartados por regla, separando train (< cut) y test (≥ cut)."""
    out: dict = {}
    for tag, part in (
        ("train", drops[drops["date"] < cut]),
        ("test", drops[drops["date"] >= cut]),
    ):
        block = {
            r: {"rows": int(part[r].sum()), "tickers": int(part.loc[part[r], "ticker"].nunique())}
            for r in DROP_RULES
        }
        block["rows_total"] = int(len(part))
        block["tickers_total"] = int(part["ticker"].nunique())
        out[tag] = block
    return out


def build_panel(
    tickers: dict[str, str],
    horizon: int = HORIZON,
    score_mode: ScoreMode = "p_up",
    use_open_entry: bool = True,
    use_mask: bool = False,
    mask: pd.DataFrame | None = None,
    drops_out: list | None = None,
) -> pd.DataFrame:
    """Dataset panel con features, retorno futuro y targets.

    ``use_open_entry``: True entra al open del día siguiente; False, al close.
    ``use_mask``: True descarta filas basura (máscara de M3a + Open ≤ 0 + |fwd_ret| > 100 %)
    ANTES de las features cross-seccionales. Default False = comportamiento previo.
    ``mask``: máscara ya cargada (por defecto ``load_quality_mask()``).
    ``drops_out``: lista a la que se agrega un DataFrame con el detalle de lo descartado
    (columnas ticker, date y una booleana por regla). No va en ``df.attrs`` porque un
    DataFrame ahí rompe ``pd.concat`` aguas abajo.
    """
    if use_mask and mask is None:
        mask = load_quality_mask()
    drops_parts: list[pd.DataFrame] = []
    macro = {n: store_get(n) for n in MACRO_TICKERS}
    macro = {k: v for k, v in macro.items() if v is not None}

    feat_frames: dict[str, pd.DataFrame] = {}
    for name in tickers:
        raw = store_get(name)
        if raw is None or len(raw) < 200:
            continue
        try:
            feat = compute_features(raw)
            feat = add_macro_features(feat, macro)
            feat["fwd_ret"] = compute_target(raw, horizon=horizon, use_open_entry=use_open_entry)
            if use_mask:
                feat, dropped = clean_ticker_rows(feat, raw["Open"], mask, tickers.get(name, name))
                if len(dropped):
                    drops_parts.append(
                        dropped.rename_axis("date").reset_index().assign(ticker=name)
                    )
            feat["y"] = (feat["fwd_ret"] > 0).astype(int)
            feat["ticker"] = name
            if score_mode == "signal":
                scores = feat.apply(compute_signal_score, axis=1)
                feat["signal_score"] = scores.apply(lambda x: x[0])
            feat_frames[name] = feat
        except Exception as exc:
            print(f"  {name}: {exc}")

    if not feat_frames:
        raise RuntimeError("Sin datos suficientes para el universo solicitado")

    feat_frames = add_cross_section_features(feat_frames)
    df = pd.concat(feat_frames.values()).sort_index()
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=feat_cols + ["fwd_ret"])

    df["uni_n"] = df.groupby(df.index)["fwd_ret"].transform("count")
    df["uni_med"] = df.groupby(df.index)["fwd_ret"].transform("median")
    df = df[df["uni_n"] >= MIN_NAMES]
    df["y_rel"] = (df["fwd_ret"] > df["uni_med"]).astype(int)
    if use_mask and drops_out is not None:
        cols = ["ticker", "date", *DROP_RULES]
        drops_out.append(
            pd.concat(drops_parts, ignore_index=True)[cols]
            if drops_parts else pd.DataFrame(columns=cols).astype({r: bool for r in DROP_RULES})
        )
    return df


def _score_column(score_mode: ScoreMode) -> str:
    if score_mode == "p_up":
        return "p_up"
    if score_mode == "p_rel":
        return "p_rel"
    return "signal_score"


def _pick_weights(
    picks: pd.DataFrame, weight_mode: WeightMode, score_col: str
) -> pd.Series:
    """Pesos del basket (suman 1) indexados por ticker."""
    if weight_mode == "conviction":
        scores = picks[score_col].values.astype(float)
        scores = np.maximum(scores - scores.min(), 1e-9)
        w = scores / scores.sum()
    else:
        w = np.full(len(picks), 1.0 / len(picks))
    return pd.Series(w, index=picks["ticker"].values)


def _period_returns(
    picks: pd.DataFrame,
    weight_mode: WeightMode,
    score_col: str,
    commission_pct: float,
    slippage_pct: float,
) -> tuple[float, float]:
    """Retorno bruto y neto (costo fijo de ida y vuelta) del período."""
    rets = picks["fwd_ret"].values / 100.0
    weights = _pick_weights(picks, weight_mode, score_col).values
    gross = float(np.dot(weights, rets))
    active = len(picks) > 0
    net = Metrics.net_return(gross, commission_pct, slippage_pct) if active else 0.0
    return gross, net


def _naftrac_return(d0: pd.Timestamp, horizon: int) -> float:
    naf = store_get("NAFTRAC")
    if naf is None:
        return 0.0
    fut = naf["Close"][naf.index > d0]
    base = naf["Close"][naf.index <= d0]
    if len(fut) >= horizon and len(base) > 0:
        return float(fut.iloc[horizon - 1] / base.iloc[-1] - 1)
    return 0.0


def run_topk_backtest(
    test: pd.DataFrame,
    horizon: int = HORIZON,
    top_k: int = DEFAULT_TOP_K,
    score_mode: ScoreMode = "p_up",
    weight_mode: WeightMode = "equal",
    commission_pct: float = Metrics.DEFAULT_COMMISSION_PCT,
    slippage_pct: float = Metrics.DEFAULT_SLIPPAGE_PCT,
    capital: float = CAPITAL,
    min_p_up: float = 0.0,
    cost_mode: CostMode = "turnover",
    tiebreak: bool = True,
) -> dict:
    """
    Backtest top-K long-only sobre el set de test.

    ``cost_mode="turnover"`` cobra ``cost_por_lado × Σ|Δw|`` (solo lo que cambia);
    ``"fixed"`` cobra siempre el costo de ida y vuelta completo (100% de rotación).
    ``tiebreak=True`` ordena por ``p_rank`` (mesetas de la isotónica desempatadas
    por el score crudo) cuando está presente, manteniendo ``min_p_up`` y el AUC
    sobre ``p_up``.

    Returns:
        Dict con métricas duales, benchmark y curvas de equity.
    """
    score_col = _score_column(score_mode)
    if score_col not in test.columns:
        raise ValueError(f"Columna de score {score_col!r} no encontrada en test")
    rank_col = (
        "p_rank"
        if (tiebreak and score_col in ("p_up", "p_rel") and "p_rank" in test.columns)
        else score_col
    )

    test = test.copy()
    test["week"] = test.index.to_period("W")
    weeks = sorted(test["week"].unique())
    step = max(1, round(horizon / 5))

    eq_gross, eq_net, eq_bh = capital, capital, capital
    curve_gross, curve_net, curve_bh = [eq_gross], [eq_net], [eq_bh]
    r_gross, r_net, r_bh = [], [], []
    weights_rows: list[pd.Series] = []

    for wi, week in enumerate(weeks):
        if wi % step != 0:
            continue
        grp = test[test["week"] == week]
        last = grp.sort_index().groupby("ticker").tail(1)
        if len(last) < MIN_NAMES:
            continue

        if tiebreak and "ticker" in last.columns:
            ranked = last.sort_values(
                [rank_col, "ticker"], ascending=[False, True], kind="mergesort"
            )
        else:
            ranked = last.sort_values(rank_col, ascending=False)
        if min_p_up > 0 and score_col in ("p_up", "p_rel"):
            ranked = ranked[ranked[score_col] >= min_p_up]
        picks = ranked.head(top_k)
        if len(picks) == 0:
            continue

        gross, net = _period_returns(
            picks, weight_mode, score_col, commission_pct, slippage_pct
        )
        weights_rows.append(_pick_weights(picks, weight_mode, score_col))
        bh = _naftrac_return(last.index.max(), horizon)

        eq_gross *= 1 + gross
        eq_net *= 1 + net
        eq_bh *= 1 + bh
        r_gross.append(gross)
        r_net.append(net)
        r_bh.append(bh)
        curve_gross.append(eq_gross)
        curve_net.append(eq_net)
        curve_bh.append(eq_bh)

    # Rotación real y costo por rotación (los pesos no derivan entre rebalanceos).
    cost_per_side = commission_pct + slippage_pct
    gross_s = pd.Series(r_gross, dtype=float)
    if weights_rows:
        turn_s = compute_turnover(pd.DataFrame(weights_rows).reset_index(drop=True))
    else:
        turn_s = pd.Series(dtype=float)
    if cost_mode == "turnover" and weights_rows:
        r_net = net_returns(gross_s, turn_s, cost_per_side).tolist()
        eq_net = capital
        curve_net = [eq_net]
        for r in r_net:
            eq_net *= 1 + r
            curve_net.append(eq_net)

    periods_per_year = 252 / horizon
    dual = Metrics.calculate_dual_metrics(
        np.array(r_gross),
        np.array(r_net),
        periods_per_year=periods_per_year,
        y_true_binary=test["y"].values if "y" in test.columns else None,
        y_prob=test["p_up"].values if "p_up" in test.columns else None,
    )
    bench = Metrics.portfolio_metrics_from_returns(
        np.array(r_bh), periods_per_year=periods_per_year
    )

    return {
        "strategy": {
            **dual,
            "n_periods": len(r_gross),
            "avg_period_gross_pct": round(float(np.mean(r_gross) * 100), 3) if r_gross else 0,
            "avg_period_net_pct": round(float(np.mean(r_net) * 100), 3) if r_net else 0,
            "final_equity_gross": round(eq_gross, 2),
            "final_equity_net": round(eq_net, 2),
        },
        "benchmark_naftrac": {
            "total_return": bench["total_return"],
            "sharpe_ratio": bench["sharpe_ratio"],
            "max_drawdown": bench["max_drawdown"],
            "win_rate": bench["win_rate"],
            "n_periods": len(r_bh),
            "final_equity": round(eq_bh, 2),
        },
        "vs_benchmark": {
            "excess_return_net": round(dual["total_return_net"] - bench["total_return"], 4),
            "sharpe_delta_net": round(dual["sharpe_net"] - bench["sharpe_ratio"], 3),
        },
        "config": {
            "horizon": horizon,
            "top_k": top_k,
            "score_mode": score_mode,
            "weight_mode": weight_mode,
            "commission_pct": commission_pct,
            "slippage_pct": slippage_pct,
            "round_trip_cost": Metrics.round_trip_cost(commission_pct, slippage_pct),
            "capital": capital,
            "min_p_up": min_p_up,
            "cost_mode": cost_mode,
            "tiebreak": tiebreak,
        },
        "turnover_mean": round(float(turn_s.mean()), 4) if len(turn_s) else 0.0,
        "turnover_series": [round(float(x), 4) for x in turn_s],
        "breakeven_cost_per_side": (
            float(breakeven_cost_per_side(gross_s, turn_s)) if len(turn_s) else None
        ),
        "cost_sensitivity": (
            cost_sensitivity(
                gross_s, turn_s, SENSITIVITY_COSTS, periods_per_year
            ).to_dict(orient="records")
            if len(turn_s)
            else []
        ),
        "curves": {
            "gross": curve_gross,
            "net": curve_net,
            "naftrac": curve_bh,
        },
        "period_returns": {
            "gross": r_gross,
            "net": r_net,
            "naftrac": r_bh,
        },
    }


def build_test_panel(
    universe: str = "bmv",
    score_mode: ScoreMode = "p_up",
    horizon: int = HORIZON,
    entry: EntryMode = "open",
    tickers: dict[str, str] | None = None,
    use_mask: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Entrena el ensemble (semillas fijas) y devuelve el panel de test con ``p_up``/``p_rel``.

    ``use_mask`` (default False) pasa a ``build_panel``; con True, ``info["mask_drops"]``
    resume las filas descartadas por regla en train y test.

    Returns:
        (test, info): ``test`` = filas del 20% final con las columnas de score; ``info`` =
        AUC, columna objetivo, fechas y tamaños de train/test.
    """
    np.random.seed(SEED)
    tickers = tickers if tickers is not None else load_universe_tickers(universe)
    drops_out: list[pd.DataFrame] = []
    df = build_panel(
        tickers,
        horizon=horizon,
        score_mode=score_mode,
        use_open_entry=(entry == "open"),
        use_mask=use_mask,
        drops_out=drops_out,
    ).sort_index()

    cut = df.index[int(len(df) * TRAIN_RATIO)]
    train, test = df[df.index < cut], df[df.index >= cut].copy()

    feat_cols = [c for c in FEATURE_COLS if c in train.columns]
    target_col = "y_rel" if score_mode == "p_rel" else "y"
    model = EnsembleModel(top_features=15, calibrate=True)
    model.fit(train, feature_cols=feat_cols, target_col=target_col)

    p_test = model.predict_proba(test, feat_cols)
    p_rank = model.predict_proba_tiebroken(test, feat_cols)
    test["p_up"] = p_test
    test["p_rel"] = p_test
    test["p_rank"] = p_rank

    auc_target = test[target_col]
    auc = (
        float(roc_auc_score(auc_target, p_test))
        if auc_target.nunique() > 1
        else 0.5
    )

    if score_mode == "signal" and "signal_score" not in test.columns:
        scores = test.apply(compute_signal_score, axis=1)
        test["signal_score"] = scores.apply(lambda x: x[0])

    info = {
        "auc": round(auc, 4),
        "target_col": target_col,
        "train_start": str(train.index.min().date()),
        "train_end": str(train.index.max().date()),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seed": SEED,
    }
    if drops_out:
        info["mask_drops"] = summarize_mask_drops(drops_out[0], cut)
    return test, info


def train_and_backtest(
    universe: str = "bmv",
    top_k: int = DEFAULT_TOP_K,
    score_mode: ScoreMode = "p_up",
    weight_mode: WeightMode = "equal",
    horizon: int = HORIZON,
    commission_pct: float = Metrics.DEFAULT_COMMISSION_PCT,
    slippage_pct: float = Metrics.DEFAULT_SLIPPAGE_PCT,
    min_p_up: float = DEFAULT_MIN_P_UP,
    cost_mode: CostMode = "turnover",
    entry: EntryMode = "open",
    tiebreak: bool = True,
) -> dict:
    """Pipeline completo: cargar datos, entrenar ensemble, backtest top-K."""
    test, info = build_test_panel(universe, score_mode, horizon, entry)

    bt = run_topk_backtest(
        test,
        horizon=horizon,
        top_k=top_k,
        score_mode=score_mode,
        weight_mode=weight_mode,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        min_p_up=min_p_up,
        cost_mode=cost_mode,
        tiebreak=tiebreak,
    )

    bt["model"] = info
    bt["universe"] = universe
    bt["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return bt


def print_summary(result: dict) -> None:
    """Imprime resumen legible en consola."""
    s = result["strategy"]
    b = result["benchmark_naftrac"]
    v = result["vs_benchmark"]
    cfg = result["config"]
    m = result.get("model", {})

    print("\n" + "=" * 72)
    print("  BACKTEST TOP-K LONG-ONLY (Ghosh 2022 style)")
    print("=" * 72)
    print(f"  Universo: {result.get('universe', '?')}  |  K={cfg['top_k']}  |  "
          f"score={cfg['score_mode']}  |  weight={cfg['weight_mode']}")
    if m:
        print(f"  AUC={m.get('auc', 'n/a')}  |  test {m.get('test_start')} → {m.get('test_end')}")
    print(f"  Costos: {cfg['commission_pct']*100:.2f}% comisión/lado + "
          f"{cfg['slippage_pct']*100:.3f}% slippage/lado  "
          f"(round-trip ≈ {cfg['round_trip_cost']*100:.2f}%)")
    print("-" * 72)
    print(f"  {'':20} {'Estrategia (neto)':>18} {'NAFTRAC B&H':>14}")
    print(f"  {'Sharpe':20} {s['sharpe_net']:>18.3f} {b['sharpe_ratio']:>14.3f}")
    print(f"  {'Max Drawdown':20} {s['max_drawdown']*100:>17.2f}% {b['max_drawdown']*100:>13.2f}%")
    print(f"  {'Total Return':20} {s['total_return_net']*100:>17.2f}% {b['total_return']*100:>13.2f}%")
    print(f"  {'Win Rate (neto)':20} {s['win_rate_net']*100:>17.1f}% {b['win_rate']*100:>13.1f}%")
    print(f"  {'Calmar':20} {s['calmar_ratio']:>18.3f}")
    print(f"  {'Excess vs NAFTRAC':20} {v['excess_return_net']*100:>17.2f}%")
    print(f"  Rotación media: {result['turnover_mean']*100:.1f}%  |  "
          f"costo de equilibrio: {(result['breakeven_cost_per_side'] or 0)*100:.3f}%/lado  |  "
          f"modo={cfg.get('cost_mode', 'fixed')}")
    print(f"  Períodos: {s['n_periods']}  |  Sharpe bruto={s.get('sharpe_gross', 0):.3f}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest top-K long-only BMV")
    parser.add_argument("--universe", default="bmv", choices=list(UNIVERSE_GROUPS))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--min-p-up",
        type=float,
        default=DEFAULT_MIN_P_UP,
        help="Filtrar picks con P(up) >= umbral antes de top-K",
    )
    parser.add_argument(
        "--score",
        default="p_up",
        choices=["p_up", "p_rel", "signal"],
        help="Criterio de ranking: P(up), vs mediana, o signal score",
    )
    parser.add_argument(
        "--weight",
        default="equal",
        choices=["equal", "conviction"],
        help="Ponderación: equal-weight o convicción por score",
    )
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--commission", type=float, default=Metrics.DEFAULT_COMMISSION_PCT)
    parser.add_argument("--slippage", type=float, default=Metrics.DEFAULT_SLIPPAGE_PCT)
    parser.add_argument(
        "--cost-mode",
        default="turnover",
        choices=["turnover", "fixed"],
        help="turnover: costo según rotación real; fixed: 100%% de rotación cada periodo",
    )
    parser.add_argument(
        "--entry",
        default="open",
        choices=["open", "close"],
        help="Precio de entrada del retorno futuro: open (t+1) o close",
    )
    parser.add_argument(
        "--no-tiebreak",
        dest="tiebreak",
        action="store_false",
        default=True,
        help="Desactivar desempate de mesetas de p_up con el score crudo (p_rank)",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        default=str(ROOT / "data" / "backtest_topk.json"),
        help="Ruta JSON de salida",
    )
    args = parser.parse_args()

    result = train_and_backtest(
        universe=args.universe,
        top_k=args.top_k,
        score_mode=args.score,
        weight_mode=args.weight,
        horizon=args.horizon,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
        min_p_up=args.min_p_up,
        cost_mode=args.cost_mode,
        entry=args.entry,
        tiebreak=args.tiebreak,
    )

    print_summary(result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
