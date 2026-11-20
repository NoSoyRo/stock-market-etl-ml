"""
Evaluación del EnsembleModel con Purged Walk-Forward CV + Embargo.

Metodología de López de Prado:
  - 6 folds de walk-forward
  - Purging por FECHAS: elimina del train observaciones cuyo label se solapa con test
  - Embargo: buffer de 10 días entre train y test
  - Entrada causal: Open de t+1
  - Salida: Close de t+1+horizon
  - Métricas: AUC, Sharpe, Profit Factor y Deflated Sharpe Ratio

Ejecución:
    python scripts/run_purged_cv.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.data.store import BMV_TICKERS, MACRO_TICKERS, get as store_get
from src.ml.ensemble import EnsembleModel
from src.ml.features import (
    FEATURE_COLS,
    add_cross_section_features,
    add_macro_features,
    compute_features,
)
from src.ml.purged_cv import (
    deflated_sharpe_ratio,
    purged_walk_forward_date_splits,
)
from src.ml.targets import binary_label, forward_return_open, label_span
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, count_trials, log_trial

HORIZON = 5
N_SPLITS = 6
EMBARGO = 10
PROB_THRESH = 0.53

# Con entrada en Open t+1, la etiqueta usa información hasta t+1+horizon.
# Para HORIZON=5, el span real que debe purgarse es 6 días.
LABEL_SPAN = label_span(
    f"dir_{HORIZON}",
    entry="next_open",
)

# FEATURE_COLS ya incluye las columnas cross-section del proyecto.
all_feature_cols = FEATURE_COLS


# ─── 1. Cargar y preparar dataset ─────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    macro = {name: store_get(name) for name in MACRO_TICKERS}
    macro = {k: v for k, v in macro.items() if v is not None}

    feat_frames: dict[str, pd.DataFrame] = {}

    for ticker in BMV_TICKERS:
        raw = store_get(ticker)

        if raw is None or len(raw) < 150:
            continue

        try:
            feat = compute_features(raw)
            feat = add_macro_features(feat, macro)

            # Retorno forward real, en porcentaje:
            # entrada = Open[t+1]
            # salida  = Close[t+1+HORIZON]
            #
            # forward_return_open devuelve una fracción; se multiplica por 100
            # para conservar la convención histórica del script.
            feat["ret_fwd"] = (
                forward_return_open(
                    raw["Open"],
                    raw["Close"],
                    HORIZON,
                )
                * 100
            )

            # binary_label conserva NaN cuando no existe suficiente futuro.
            feat["y"] = binary_label(feat["ret_fwd"])

            feat["ticker"] = ticker
            feat_frames[ticker] = feat

        except Exception as exc:
            print(f"  {ticker}: {exc}")

    feat_frames = add_cross_section_features(feat_frames)

    rows: list[pd.DataFrame] = []

    for feat in feat_frames.values():
        available = [
            col
            for col in all_feature_cols
            if col in feat.columns
        ]

        clean = feat.dropna(
            subset=available + ["y", "ret_fwd"]
        ).copy()

        clean = clean[np.isfinite(clean["ret_fwd"])].copy()

        # El target se conserva como float mientras puede contener NaN;
        # después del dropna ya es seguro convertirlo a entero.
        clean["y"] = clean["y"].astype(int)

        if len(clean) > 30:
            rows.append(
                clean[
                    available
                    + ["y", "ret_fwd", "ticker"]
                ]
            )

    if not rows:
        raise RuntimeError("Sin datos")

    df = pd.concat(rows).sort_index()

    print(
        f"Dataset: {len(df):,} obs  |  "
        f"{df['ticker'].nunique()} tickers  |  "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )

    return df


# ─── 2. Utilidades de evaluación ──────────────────────────────────────────────

def _profit_factor(returns: np.ndarray) -> float | None:
    """Profit Factor = ganancias brutas / pérdidas brutas.

    Devuelve None cuando no hay pérdidas, porque el cociente no es finito.
    """
    returns = np.asarray(returns, dtype=float)

    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(-returns[returns < 0].sum())

    if gross_loss <= 0:
        return None

    return gross_profit / gross_loss


def _safe_distribution_stats(returns: np.ndarray) -> tuple[float, float]:
    """Skew y kurtosis no excedente con defaults seguros."""
    returns = np.asarray(returns, dtype=float)

    skew = (
        float(stats.skew(returns))
        if len(returns) > 2
        else 0.0
    )
    kurt = (
        float(stats.kurtosis(returns, fisher=False))
        if len(returns) > 3
        else 3.0
    )

    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 3.0

    return skew, kurt


# ─── 3. Evaluar con Purged CV ─────────────────────────────────────────────────

def evaluate_purged_cv(df: pd.DataFrame) -> dict:
    # Reset index para usar posiciones iloc.
    df_reset = df.reset_index(drop=False)

    if "Date" in df_reset.columns:
        dates = pd.to_datetime(df_reset["Date"])
    else:
        # El primer campo proviene del índice temporal después de reset_index.
        dates = pd.to_datetime(df_reset.iloc[:, 0])

    X = df_reset[
        [
            col
            for col in all_feature_cols
            if col in df_reset.columns
        ]
    ]
    y = df_reset["y"]

    folds_results: list[dict] = []
    all_rets: list[np.ndarray] = []

    # IMPORTANTE:
    # este dataset es un panel (muchos tickers por fecha).
    # Por eso el purge/embargo debe hacerse por fechas únicas, no por filas.
    for fold_i, (tr_idx, te_idx) in enumerate(
        purged_walk_forward_date_splits(
            dates,
            n_splits=N_SPLITS,
            horizon=LABEL_SPAN,
            embargo=EMBARGO,
        )
    ):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_te = y.iloc[te_idx]

        if len(np.unique(y_te)) < 2:
            continue

        # DataFrames esperados por EnsembleModel.
        train_df = X_tr.copy()
        train_df["y"] = y_tr.values

        test_df = X_te.copy()
        test_df["y"] = y_te.values

        available = [
            col
            for col in all_feature_cols
            if col in train_df.columns
        ]

        model = EnsembleModel(top_features=15)
        model.fit(
            train_df,
            feature_cols=available,
            target_col="y",
        )

        p_te = model.predict_proba(
            test_df,
            available,
        )
        auc = roc_auc_score(
            y_te.values,
            p_te,
        )

        # Estrategia simple: señal cuando la probabilidad supera el umbral.
        sigs = p_te >= PROB_THRESH

        if sigs.sum() > 0:
            # ret_fwd está almacenado en porcentaje; convertir a fracción
            # para Sharpe, skew, kurtosis y Profit Factor.
            rets = (
                df_reset.iloc[te_idx]["ret_fwd"]
                .to_numpy(dtype=float)[sigs]
                / 100
            )

            all_rets.append(rets)

            # Con una sola señal (o desviación ~0), Sharpe no es estimable.
            if len(rets) > 1:
                ret_std = float(np.std(rets, ddof=1))
            else:
                ret_std = 0.0

            if ret_std > 1e-12:
                sharpe = (
                    float(np.mean(rets))
                    / ret_std
                    * np.sqrt(252 / HORIZON)
                )
            else:
                sharpe = 0.0

            win_rate = float((rets > 0).mean())
            n_signals = int(sigs.sum())
            profit_factor = _profit_factor(rets)

        else:
            sharpe = 0.0
            win_rate = 0.0
            n_signals = 0
            profit_factor = None

        test_start = dates.iloc[te_idx].min()
        test_end = dates.iloc[te_idx].max()

        fold_res = {
            "fold": fold_i + 1,
            "test_start": str(test_start)[:10],
            "test_end": str(test_end)[:10],
            "n_test": int(len(te_idx)),
            "n_train": int(len(tr_idx)),
            "auc": round(float(auc), 4),
            "sharpe": round(float(sharpe), 3),
            "profit_factor": (
                round(float(profit_factor), 3)
                if profit_factor is not None
                else None
            ),
            "win_rate": round(win_rate, 3),
            "n_signals": n_signals,
            "w_lr": round(float(model.w_lr), 3),
            "w_lgbm": round(float(model.w_lgbm), 3),
        }

        folds_results.append(fold_res)

        pf_text = (
            f"{profit_factor:.2f}"
            if profit_factor is not None
            else "N/A"
        )

        print(
            f"  Fold {fold_i + 1}  "
            f"[{fold_res['test_start']}→{fold_res['test_end']}]"
            f"  AUC={auc:.4f}"
            f"  Sharpe={sharpe:.2f}"
            f"  PF={pf_text}"
            f"  WR={win_rate:.0%}"
            f"  signals={n_signals}"
            f"  n_train={len(tr_idx):,}"
        )

    if not folds_results:
        return {"error": "Sin folds válidos"}

    aucs = [row["auc"] for row in folds_results]
    sharpes = [row["sharpe"] for row in folds_results]
    wrs = [row["win_rate"] for row in folds_results]

    mean_sharpe = float(np.mean(sharpes))

    # DSR: los folds no son ensayos distintos.
    # Se usa el número medio de señales que respalda el Sharpe de un fold.
    n_obs = int(
        np.mean(
            [row["n_signals"] for row in folds_results]
        )
    )

    trial_config = {
        "model": "ensemble",
        "n_splits": N_SPLITS,
        "embargo": EMBARGO,
        "horizon": HORIZON,
        "label_span": LABEL_SPAN,
        "entry": "next_open",
        "split_unit": "dates",
        "prob_thresh": PROB_THRESH,
        "n_features": len(all_feature_cols),
    }

    n_trials = trials_for_dsr(
        trial_config,
        {
            "mean_sharpe": mean_sharpe,
            "mean_auc": float(np.mean(aucs)),
        },
        n_obs,
    )

    pooled = (
        np.concatenate(all_rets)
        if all_rets
        else np.array([], dtype=float)
    )

    skew, kurt = _safe_distribution_stats(pooled)

    dsr = dsr_from_fold_sharpes(
        sharpes,
        HORIZON,
        n_trials,
        n_obs,
        skewness=skew,
        kurtosis=kurt,
    )

    pooled_profit_factor = (
        _profit_factor(pooled)
        if len(pooled) > 0
        else None
    )

    aggregate = {
        "mean_auc": round(float(np.mean(aucs)), 4),
        "std_auc": round(float(np.std(aucs)), 4),
        "mean_sharpe": round(mean_sharpe, 3),
        "profit_factor": (
            round(float(pooled_profit_factor), 3)
            if pooled_profit_factor is not None
            else None
        ),
        "mean_win_rate": round(float(np.mean(wrs)), 3),
        "deflated_sharpe_ratio": round(float(dsr), 4),
        "dsr_interpretation": (
            "✅ Edge estadísticamente significativo"
            if dsr > 0.95
            else "⚡ Edge moderado, monitorear"
            if dsr > 0.75
            else "⚠️ Edge débil, posible sobreajuste"
        ),
        "total_signals": int(
            sum(
                row["n_signals"]
                for row in folds_results
            )
        ),
        "n_trials": int(n_trials),
        "skew": round(skew, 3),
        "kurtosis": round(kurt, 3),
    }

    return {
        "folds": folds_results,
        "aggregate": aggregate,
        "meta": {
            "model": "EnsembleModel (WoE+LR+LightGBM)",
            "n_splits": N_SPLITS,
            "embargo": EMBARGO,
            "horizon": HORIZON,
            "label_span": LABEL_SPAN,
            "entry": "next_open",
            "exit": f"close_t+1+{HORIZON}",
            "split_unit": "dates",
            "n_features": len(all_feature_cols),
        },
    }


def trials_for_dsr(
    config: dict,
    metrics: dict,
    n_obs: int,
    *,
    path: str | None = None,
) -> int:
    """Registra esta corrida y devuelve el número de ensayos únicos.

    El DSR descuenta la suerte de haber probado varias configuraciones.
    El conteo sale del registro único data/eval/trial_log.jsonl y no de
    folds × features. Repetir exactamente la misma configuración no debe
    inflar el conteo.
    """
    path = path or str(ROOT / TRIALS_PATH)

    log_trial(
        "purged_cv:ensemble",
        config,
        metrics,
        n_obs,
        path=path,
    )

    return max(
        count_trials(path=path),
        1,
    )


def dsr_from_fold_sharpes(
    annual_sharpes,
    horizon: int,
    n_trials: int,
    n_obs: int,
    *,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    use_fold_sr_std: bool = False,
) -> float:
    """DSR a partir de Sharpes de fold anualizados.

    deflated_sharpe_ratio exige Sharpe por periodo, por lo que aquí se
    desanualiza con el mismo factor usado al calcular el Sharpe.

    `n_obs` aproxima el número de observaciones detrás del Sharpe de cada fold.
    `skewness` y `kurtosis` son los de los retornos reales de las señales.
    """
    per = (
        np.asarray(
            annual_sharpes,
            dtype=float,
        )
        / np.sqrt(252 / horizon)
    )

    sr_std = None

    if use_fold_sr_std and len(per) > 1:
        sr_std = float(
            per.std(ddof=1)
        )

        if sr_std <= 0:
            sr_std = None

    return float(
        deflated_sharpe_ratio(
            float(per.mean()),
            n_trials,
            max(int(n_obs), 2),
            skewness=skewness,
            kurtosis=kurtosis,
            sr_std=sr_std,
        )
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 72)
    print(
        "  Purged Walk-Forward CV  |  "
        "EnsembleModel (WoE+LR+LightGBM)"
    )
    print(
        f"  {N_SPLITS} folds"
        f"  |  embargo={EMBARGO}d"
        f"  |  horizonte={HORIZON}d"
        f"  |  label_span={LABEL_SPAN}d"
        "  |  split=fechas"
    )
    print("=" * 72 + "\n")

    df = build_dataset()

    print(
        f"\nEjecutando {N_SPLITS} folds "
        "Purged CV por fechas...\n"
    )

    results = evaluate_purged_cv(df)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    agg = results["aggregate"]

    print(f"\n{'=' * 72}")
    print("  RESULTADOS PURGED CV")
    print(f"{'=' * 72}")
    print(
        f"  AUC:              "
        f"{agg['mean_auc']:.4f} ± {agg['std_auc']:.4f}"
    )
    print(
        f"  Sharpe:           "
        f"{agg['mean_sharpe']:.2f}"
    )

    pf = agg["profit_factor"]
    pf_text = (
        f"{pf:.2f}"
        if pf is not None
        else "N/A"
    )
    print(f"  Profit Factor:     {pf_text}")

    print(
        f"  Win rate:         "
        f"{agg['mean_win_rate']:.0%}"
    )
    print(
        f"  DSR:              "
        f"{agg['deflated_sharpe_ratio']:.4f}"
        f"  → {agg['dsr_interpretation']}"
    )
    print(
        f"  Total señales:    "
        f"{agg['total_signals']}"
    )
    print(f"{'=' * 72}\n")

    out_path = ROOT / "data" / "purged_cv_results.json"

    out_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
