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

from src.data.store import UNIVERSE_GROUPS, get as store_get, MACRO_TICKERS
from src.ml.features import (
    FEATURE_COLS,
    add_cross_section_features,
    add_macro_features,
    compute_features,
    compute_signal_score,
    compute_target,
)
from src.ml.rates_features import (
    build_rates_panel,
    compute_rates_features,
)
from src.data.market_quality import (
    prepare_market_data,
    quality_summary,
)

from src.agent.trade_execution import execute_trade
from src.ml.ensemble import EnsembleModel
from src.ml.shared.evaluators.metrics import Metrics

BANXICO_DIR = ROOT / "data" / "rates" / "banxico"

DIVIDEND_COLS = {
    "dias_a_ex_date",
    "ex_date_window",
    "div_yield_fwd",
}

TOPK_FEATURE_COLS = [
    col for col in FEATURE_COLS
    if col not in DIVIDEND_COLS
]

HORIZON = 5
TRAIN_RATIO = 0.80
MIN_NAMES = 10
CAPITAL = 100_000
DEFAULT_TOP_K = 2
DEFAULT_MIN_P_UP = 0.56

ScoreMode = Literal["p_up", "p_rel", "signal"]
WeightMode = Literal["equal", "conviction"]


def load_universe_tickers(universe: str) -> dict[str, str]:
    """Carga tickers del universo desde store/universes."""
    tickers = UNIVERSE_GROUPS.get(universe)
    if not tickers:
        raise ValueError(
            f"Universo desconocido: {universe!r}. "
            f"Opciones: {list(UNIVERSE_GROUPS)}"
        )
    return tickers


def build_panel(
    tickers: dict[str, str],
    horizon: int = HORIZON,
    score_mode: ScoreMode = "p_up",
) -> pd.DataFrame:
    """Dataset panel con features, retorno futuro y targets."""
    macro = {n: store_get(n) for n in MACRO_TICKERS}
    macro = {k: v for k, v in macro.items() if v is not None}

    rates = build_rates_panel(rates_dir=BANXICO_DIR)

    if rates.empty:
        raise RuntimeError("No hay datos historicos de Banxico")

    rates = rates.copy()
    rates.index += pd.Timedelta(days=1)

    benchmark_calendar = _load_naftrac().index

    feat_frames: dict[str, pd.DataFrame] = {}
    for name in tickers:
        raw = store_get(name)
        if raw is None or len(raw) < 200:
            continue
        try:
            raw = raw.sort_index()

            calendar = benchmark_calendar[
                (benchmark_calendar >= raw.index.min())
                & (benchmark_calendar <= raw.index.max())
            ]

            if len(calendar) < 200:
                continue

            quality = quality_summary(
                raw, calendar=calendar
            )

            print(
                f"  {name}: "
                f"invalidos={quality['invalid_records']}, "
                f"ausentes={quality['missing_sessions']}, "
                f"volumen_cero={quality['zero_volume']}"
            )

            raw = prepare_market_data(
                raw, calendar=calendar
            )

            feat = compute_features(raw)

            feat["signal_tradable"] = raw["tradable"]

            feat["entry_tradable"] = (
                raw["tradable"]
                .shift(-1, fill_value=False)
            )

            feat["exit_tradable"] = (
                raw["tradable"]
                .shift(-(horizon + 1), fill_value=False)
            )
            feat = add_macro_features(feat, macro)

            fechas = pd.DatetimeIndex(feat.index)

            alineado = (
                rates
                .reindex(rates.index.union(fechas))
                .sort_index()
                .ffill()
                .reindex(fechas)
            )

            tasas_feat = compute_rates_features(alineado)

            for col in tasas_feat.columns:
                feat[col] = tasas_feat[col]
            feat["fwd_ret"] = compute_target(raw, horizon=horizon, use_open_entry=True)
            calendario = pd.Series(
                raw.index, index=raw.index
            )

            feat["entry_date"] = calendario.shift(-1)
            feat["exit_date"] = calendario.shift(-(horizon + 1))

            valid_target = (
                feat["entry_tradable"]
                & feat["exit_tradable"]
                & np.isfinite(feat["fwd_ret"])
            )

            feat["fwd_ret"] = feat["fwd_ret"].where(
                valid_target
            )

            feat["y"] = (
                (feat["fwd_ret"] > 0)
                .astype(float)
                .where(valid_target)
            )
            feat["ticker"] = name
            if score_mode == "signal":
                scores = feat.apply(compute_signal_score, axis=1)
                feat["signal_score"] = scores.apply(lambda x: x[0])
            feat_frames[name] = feat
        except Exception as exc:
            print(f"  {name}: {exc}")

    if not feat_frames:
        raise RuntimeError("Sin datos suficientes para el universo solicitado")

    feat_frames = {
        name: frame.loc[frame["signal_tradable"]].copy()
        for name, frame in feat_frames.items()
    }

    feat_frames = add_cross_section_features(feat_frames)
    df = pd.concat(feat_frames.values()).sort_index()
    df[TOPK_FEATURE_COLS] = df[
        TOPK_FEATURE_COLS
    ].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=TOPK_FEATURE_COLS)

    df["uni_n"] = df.groupby(
        df.index
    )["ticker"].transform("count")

    df["uni_med"] = df.groupby(
        df.index
    )["fwd_ret"].transform("median")

    df["uni_labeled"] = df.groupby(
        df.index
    )["fwd_ret"].transform("count")

    df["y_rel"] = (
        (df["fwd_ret"] > df["uni_med"])
        .astype(float)
        .where(
            df["fwd_ret"].notna()
            & (df["uni_labeled"] >= MIN_NAMES)
        )
    )

    return df


def _score_column(score_mode: ScoreMode) -> str:
    if score_mode == "p_up":
        return "p_up"
    if score_mode == "p_rel":
        return "p_rel"
    return "signal_score"


def _period_returns(
    picks: pd.DataFrame,
    weight_mode: WeightMode,
    score_col: str,
    commission_pct: float,
    slippage_pct: float,
    calendar: pd.DatetimeIndex,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    market_cache: dict,
) -> dict:
    """Simula las operaciones de una cartera seleccionada."""

    if picks.empty:
        return {
            "gross": 0.0,
            "net": 0.0,
            "actual_exit": exit_date,
            "trades": [],
            "open_positions": []
        }

    if weight_mode == "conviction":
        scores = picks[score_col].to_numpy(dtype=float)
        scores = np.maximum(
            scores - scores.min(), 1e-9
        )
        weights = scores / scores.sum()
    else:
        weights = np.full(
            len(picks), 1.0 / len(picks)
        )

    gross = 0.0
    net = 0.0
    actual_exit = exit_date
    trades = []
    open_positions = []

    for (_, row), weight in zip(
        picks.iterrows(), weights
    ):
        ticker = row["ticker"]

        if ticker not in market_cache:
            market_cache[ticker] = store_get(ticker)

        market = market_cache[ticker]

        if market is None or market.empty:
            raise RuntimeError(
                f"Sin datos de mercado para {ticker}"
            )

        trade = execute_trade(
            market=market,
            calendar=calendar,
            entry_date=entry_date,
            exit_date=exit_date,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct
        )

        trade["ticker"] = ticker
        trade["weight"] = float(weight)

        trades.append(trade)

        if trade["status"] == "open_position":
            open_positions.append(trade)
            continue

        gross += weight * trade["gross_return"]
        net += weight * trade["net_return"]

        if trade["actual_exit"] is not None:
            actual_exit = max(
                actual_exit,
                pd.Timestamp(trade["actual_exit"])
            )

    return {
        "gross": float(gross),
        "net": float(net),
        "actual_exit": actual_exit,
        "trades": trades,
        "open_positions": open_positions
    }


def _load_naftrac() -> pd.DataFrame:
    """Carga el ETF utilizado como benchmark."""
    ruta = ROOT / "data" / "market" / "NAFTRACISHRS.MX.parquet"

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el benchmark: {ruta}"
        )

    naf = pd.read_parquet(ruta).sort_index()

    if naf.empty or naf.index.has_duplicates:
        raise ValueError("Datos invalidos de NAFTRAC")

    return naf


def _naftrac_return(
    naf: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> float:
    """Retorno del ETF entre fechas exactas de operacion."""

    if entry_date not in naf.index or exit_date not in naf.index:
        raise ValueError(
            f"NAFTRAC sin datos entre {entry_date} y {exit_date}"
        )

    entrada = float(naf.loc[entry_date, "Open"])
    salida = float(naf.loc[exit_date, "Close"])

    if not np.isfinite([entrada, salida]).all():
        raise ValueError("Precios no finitos en NAFTRAC")

    if entrada <= 0 or salida <= 0:
        raise ValueError("Precios invalidos en NAFTRAC")

    return salida / entrada - 1


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
) -> dict:
    """
    Backtest top-K long-only sobre el set de test.

    Returns:
        Dict con métricas duales, benchmark y curvas de equity.
    """
    score_col = _score_column(score_mode)
    if score_col not in test.columns:
        raise ValueError(f"Columna de score {score_col!r} no encontrada en test")

    if horizon < 1 or top_k < 1:
        raise ValueError("Horizonte y top_k deben ser positivos")

    test = test.copy().sort_index()
    naf = _load_naftrac()

    eq_gross, eq_net, eq_bh = capital, capital, capital

    curve_gross = [capital]
    curve_net = [capital]
    curve_bh = [capital]

    r_gross, r_net, r_bh = [], [], []
    operations = []

    previous_exit = None
    previous_benchmark_price = None
    first_entry = None

    calendario = naf.index
    fechas = test.index.unique().sort_values()

    evaluation_end = min(
        calendario.max(),
        test.index.max()
    )

    execution_calendar = calendario[
        calendario <= evaluation_end
    ]

    market_cache = {}

    for fecha in fechas:

        if fecha not in calendario:
            continue

        posicion = calendario.get_loc(fecha)

        if posicion + horizon + 1 >= len(calendario):
            break

        entry_date = calendario[posicion + 1]
        exit_date = calendario[posicion + horizon + 1]

        if exit_date > evaluation_end:
            break

        planned_exit = exit_date

        # No utilizar capital comprometido
        if previous_exit is not None:
            if entry_date <= previous_exit:
                continue

        # Todas las acciones deben compartir fechas
        last = test.loc[[fecha]].copy()

        last = last.loc[
            np.isfinite(last[score_col])
        ]

        last = last.drop_duplicates(subset=["ticker"])

        ranked = last.sort_values(
            [score_col, "ticker"],
            ascending=[False, True]
        )

        if len(last) < MIN_NAMES:
            ranked = ranked.iloc[0:0]

        if min_p_up > 0 and score_col in ("p_up", "p_rel"):
            ranked = ranked[
                ranked[score_col] >= min_p_up
            ]

        picks = ranked.head(top_k)

        # Si no hay acciones elegibles, mantener efectivo
        execution = _period_returns(
            picks,
            weight_mode,
            score_col,
            commission_pct,
            slippage_pct,
            execution_calendar,
            entry_date,
            exit_date,
            market_cache
        )

        if execution["open_positions"]:
            pendientes = [
                t["ticker"]
                for t in execution["open_positions"]
            ]

            raise RuntimeError(
                "Posiciones sin liquidar al finalizar "
                f"la evaluacion: {pendientes}. "
                "No se pueden calcular metricas definitivas."
            )

        gross = execution["gross"]
        net = execution["net"]
        exit_date = execution["actual_exit"]

        executed = [
            t for t in execution["trades"]
            if t["status"] == "closed"
        ]

        # Benchmark continuo: comprar una sola vez
        if previous_benchmark_price is None:
            bh = _naftrac_return(
                naf, entry_date, exit_date
            )
            first_entry = entry_date
        else:
            precio_actual = float(
                naf.loc[exit_date, "Close"]
            )

            if not np.isfinite(precio_actual) or precio_actual <= 0:
                raise ValueError("Precio invalido del benchmark")

            bh = precio_actual / previous_benchmark_price - 1

        previous_benchmark_price = float(
            naf.loc[exit_date, "Close"]
        )

        eq_gross *= 1 + gross
        eq_net *= 1 + net
        eq_bh *= 1 + bh

        r_gross.append(gross)
        r_net.append(net)
        r_bh.append(bh)

        curve_gross.append(eq_gross)
        curve_net.append(eq_net)
        curve_bh.append(eq_bh)

        operations.append({
            "signal_date": str(fecha.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "tickers": picks["ticker"].tolist(),
            "n_positions": len(executed),
            "n_selected": int(len(picks)),
            "planned_exit": str(planned_exit.date()),
            "trades": execution["trades"],
            "cancelled_entries": sum(
                t["status"] == "cancelled_entry"
                for t in execution["trades"]
            ),
            "delayed_exits": sum(
                t["delayed_exit"]
                for t in execution["trades"]
            ),
            "gross_return": float(gross),
            "net_return": float(net),
            "benchmark_return": float(bh),
            "cash_only": not bool(executed),
        })

        previous_exit = exit_date

    if not operations:
        raise RuntimeError(
            "No se encontraron periodos validos de evaluacion"
        )

    # Anualizacion segun la duracion efectiva
    sesiones = (
        calendario.get_loc(previous_exit)
        - calendario.get_loc(first_entry)
        + 1
    )

    periods_per_year = 252 * len(operations) / sesiones

    dual = Metrics.calculate_dual_metrics(
        np.array(r_gross),
        np.array(r_net),
        periods_per_year=periods_per_year,
        y_true_binary=None,
        y_prob=None,
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
        },
        "curves": {
            "gross": curve_gross,
            "net": curve_net,
            "naftrac": curve_bh,
        },
        "operations": operations,
        "evaluation_period": {
            "start": str(first_entry.date()),
            "end": str(previous_exit.date()),
            "trading_sessions": int(sesiones),
            "periods_per_year": float(periods_per_year),
            "active_periods": sum(
                not op["cash_only"] for op in operations
            ),
            "cash_periods": sum(
                op["cash_only"] for op in operations
            ),
            "benchmark_symbol": "NAFTRACISHRS.MX",
            "benchmark_method": "continuous_buy_and_hold",
            "execution_policy": "wait_until_all_positions_close",
            "evaluation_cutoff": str(evaluation_end.date()),
            "cancelled_entries": sum(
                op["cancelled_entries"]
                for op in operations
            ),
            "delayed_exits": sum(
                op["delayed_exits"]
                for op in operations
            ),
        },
        "period_returns": {
            "gross": r_gross,
            "net": r_net,
            "naftrac": r_bh,
        },
    }


def train_and_backtest(
    universe: str = "bmv",
    top_k: int = DEFAULT_TOP_K,
    score_mode: ScoreMode = "p_up",
    weight_mode: WeightMode = "equal",
    horizon: int = HORIZON,
    commission_pct: float = Metrics.DEFAULT_COMMISSION_PCT,
    slippage_pct: float = Metrics.DEFAULT_SLIPPAGE_PCT,
    min_p_up: float = DEFAULT_MIN_P_UP,
) -> dict:
    """Pipeline completo: cargar datos, entrenar ensemble, backtest top-K."""
    tickers = load_universe_tickers(universe)
    df = build_panel(tickers, horizon=horizon, score_mode=score_mode).sort_index()

    fechas = df.index.unique().sort_values()
    cut = fechas[int(len(fechas) * TRAIN_RATIO)]

    train_original = df[df.index < cut]

    target_col = "y_rel" if score_mode == "p_rel" else "y"

    train = train_original[
        (train_original["exit_date"] < cut)
        & train_original[target_col].notna()
    ].copy()

    test = df[df.index >= cut].copy()

    n_purged = len(train_original) - len(train)

    if train.empty or test.empty:
        raise RuntimeError(
            "Datos insuficientes despues de la separacion temporal"
        )

    assert train["exit_date"].max() < test.index.min()

    print("Inicio de prueba:", cut)
    print("Observaciones excluidas:", n_purged)
    print("Ultima salida del entrenamiento:",
          train["exit_date"].max())

    target_col = "y_rel" if score_mode == "p_rel" else "y"
    model = EnsembleModel(top_features=15, calibrate=True)
    model.fit(train, feature_cols=TOPK_FEATURE_COLS, target_col=target_col)

    p_test = model.predict_proba(test, TOPK_FEATURE_COLS)
    test["p_up"] = p_test
    test["p_rel"] = p_test

    auc_mask = test[target_col].notna()
    auc_target = test.loc[auc_mask, target_col]

    auc = (
        float(roc_auc_score(
            auc_target,
            test.loc[auc_mask, "p_up"]
        ))
        if auc_target.nunique() > 1
        else None
    )

    print("Predicciones totales:", len(test))
    print("Predicciones evaluables:", int(auc_mask.sum()))
    print("Predicciones sin etiqueta:", int((~auc_mask).sum()))

    if score_mode == "signal" and "signal_score" not in test.columns:
        scores = test.apply(compute_signal_score, axis=1)
        test["signal_score"] = scores.apply(lambda x: x[0])

    bt = run_topk_backtest(
        test,
        horizon=horizon,
        top_k=top_k,
        score_mode=score_mode,
        weight_mode=weight_mode,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        min_p_up=min_p_up,
    )

    bt["strategy"]["auc"] = auc

    bt["model"] = {
        "auc": round(auc, 4) if auc is not None else None,
        "target_col": target_col,
        "train_start": str(train.index.min().date()),
        "train_end": str(train.index.max().date()),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "n_purged": int(n_purged),
        "split_method": "chronological_purged",
        "train_last_exit": str(
            train["exit_date"].max().date()
        ),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_test_labeled": int(auc_mask.sum()),
        "n_test_unlabeled": int((~auc_mask).sum()),
    }
    bt["data_sources"] = {
        "market": "Yahoo Finance (local market cache)",
        "rates": "Banco de Mexico - SIE API",
        "rates_directory": "data/rates/banxico",
        "rates_lag_days": 1,
        "rates_alignment": "forward_fill",
    }

    bt["model"]["feature_cols"] = TOPK_FEATURE_COLS
    bt["model"]["selected_features"] = model.top_feats
    bt["model"]["excluded_features"] = sorted(DIVIDEND_COLS)

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
        "--output",
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
    )

    print_summary(result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
