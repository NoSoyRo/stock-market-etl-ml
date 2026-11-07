"""
Evaluación del EnsembleModel con Purged Walk-Forward CV + Embargo.

Metodología de López de Prado:
  - 6 folds de walk-forward
  - Purging: elimina filas del train cuyo label se solapa con el test
  - Embargo: buffer de 10 días entre train y test
  - Métricas: AUC, Sharpe, Profit Factor, Deflated Sharpe Ratio

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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.data.store  import get as store_get, BMV_TICKERS, MACRO_TICKERS
from src.ml.features import (compute_features, add_macro_features,
                              add_cross_section_features,
                              FEATURE_COLS, CROSS_SECTION_COLS)
from src.ml.ensemble import EnsembleModel
from src.ml.purged_cv import purged_walk_forward_splits, deflated_sharpe_ratio
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, count_trials, log_trial

HORIZON    = 5
N_SPLITS   = 6
EMBARGO    = 10
PROB_THRESH = 0.53
all_feature_cols = FEATURE_COLS  # ya incluye CROSS_SECTION_COLS


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
            feat["y"]      = (raw["Close"].shift(-HORIZON) / raw["Close"] - 1 > 0).astype(int)
            feat["ticker"] = ticker
            feat_frames[ticker] = feat
        except Exception as e:
            print(f"  {ticker}: {e}")

    feat_frames = add_cross_section_features(feat_frames)

    rows = []
    for name, feat in feat_frames.items():
        available = [c for c in all_feature_cols if c in feat.columns]
        clean = feat.dropna(subset=available + ["y"])
        if len(clean) > 30:
            rows.append(clean[available + ["y", "ticker"]])

    if not rows:
        raise RuntimeError("Sin datos")

    df = pd.concat(rows).sort_index()
    print(f"Dataset: {len(df):,} obs  |  {df['ticker'].nunique()} tickers  "
          f"|  {df.index.min().date()} → {df.index.max().date()}")
    return df


# ─── 2. Evaluar con Purged CV ─────────────────────────────────────────────────

def evaluate_purged_cv(df: pd.DataFrame) -> dict:
    # Reset index para usar iloc
    df_reset = df.reset_index(drop=False)
    dates    = df_reset["Date"] if "Date" in df_reset.columns else df_reset.index

    X = df_reset[[c for c in all_feature_cols if c in df_reset.columns]]
    y = df_reset["y"]

    folds_results = []

    for fold_i, (tr_idx, te_idx) in enumerate(
        purged_walk_forward_splits(df_reset, n_splits=N_SPLITS, horizon=HORIZON, embargo=EMBARGO)
    ):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        if len(np.unique(y_te)) < 2:
            continue

        # Armar DataFrames para EnsembleModel
        train_df = X_tr.copy(); train_df["y"] = y_tr.values
        test_df  = X_te.copy(); test_df["y"]  = y_te.values

        available = [c for c in all_feature_cols if c in train_df.columns]

        model = EnsembleModel(top_features=15)
        model.fit(train_df, feature_cols=available, target_col="y")

        p_te = model.predict_proba(test_df, available)
        auc  = roc_auc_score(y_te.values, p_te)

        # Estrategia simple: señal si P > umbral
        sigs = p_te >= PROB_THRESH
        if sigs.sum() > 0:
            returns_te = df_reset.iloc[te_idx]
            # Retorno forward real si está disponible
            if "ret_fwd" in df_reset.columns:
                rets = df_reset.iloc[te_idx]["ret_fwd"].values[sigs] / 100
            else:
                # Proxy: y=1 → +1%, y=0 → -1%
                rets = np.where(y_te.values[sigs] == 1, 0.01, -0.01)
            # Convención: se escala por √(252/HORIZON) y dsr_from_fold_sharpes des-anualiza
            # con el mismo factor para obtener el Sharpe por señal/periodo que exige el DSR.
            sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252 / HORIZON)
            win_rate = (y_te.values[sigs] == 1).mean()
            n_signals = int(sigs.sum())
        else:
            sharpe, win_rate, n_signals = 0.0, 0.0, 0

        test_start = dates.iloc[te_idx[0]]  if hasattr(dates, 'iloc') else te_idx[0]
        test_end   = dates.iloc[te_idx[-1]] if hasattr(dates, 'iloc') else te_idx[-1]

        fold_res = {
            "fold":       fold_i + 1,
            "test_start": str(test_start)[:10],
            "test_end":   str(test_end)[:10],
            "n_test":     len(te_idx),
            "n_train":    len(tr_idx),
            "auc":        round(auc, 4),
            "sharpe":     round(sharpe, 3),
            "win_rate":   round(win_rate, 3),
            "n_signals":  n_signals,
            "w_lr":       round(model.w_lr, 3),
            "w_lgbm":     round(model.w_lgbm, 3),
        }
        folds_results.append(fold_res)

        print(f"  Fold {fold_i+1}  [{fold_res['test_start']}→{fold_res['test_end']}]"
              f"  AUC={auc:.4f}  Sharpe={sharpe:.2f}  WR={win_rate:.0%}  "
              f"signals={n_signals}  n_train={len(tr_idx):,}")

    if not folds_results:
        return {"error": "Sin folds válidos"}

    aucs    = [f["auc"] for f in folds_results]
    sharpes = [f["sharpe"] for f in folds_results]
    wrs     = [f["win_rate"] for f in folds_results]

    # Deflated Sharpe: n_trials = ensayos únicos del registro (no folds × features: los folds
    # no son ensayos distintos).
    mean_sharpe = np.mean(sharpes)
    n_obs       = int(np.mean([f["n_signals"] for f in folds_results]))
    n_trials    = trials_for_dsr(
        {"n_splits": N_SPLITS, "embargo": EMBARGO, "horizon": HORIZON,
         "prob_thresh": PROB_THRESH, "n_features": len(all_feature_cols)},
        {"mean_sharpe": float(mean_sharpe), "mean_auc": float(np.mean(aucs))},
        n_obs,
    )
    dsr         = dsr_from_fold_sharpes(sharpes, HORIZON, n_trials, n_obs)

    aggregate = {
        "mean_auc":   round(np.mean(aucs), 4),
        "std_auc":    round(np.std(aucs),  4),
        "mean_sharpe":  round(mean_sharpe, 3),
        "mean_win_rate": round(np.mean(wrs), 3),
        "deflated_sharpe_ratio": round(dsr, 4),
        "dsr_interpretation": (
            "✅ Edge estadísticamente significativo" if dsr > 0.95 else
            "⚡ Edge moderado, monitorear" if dsr > 0.75 else
            "⚠️ Edge débil, posible sobreajuste"
        ),
        "total_signals": sum(f["n_signals"] for f in folds_results),
        "n_trials": n_trials,
    }

    return {"folds": folds_results, "aggregate": aggregate,
            "meta": {"n_splits": N_SPLITS, "embargo": EMBARGO, "horizon": HORIZON,
                     "n_features": len(all_feature_cols)}}


def trials_for_dsr(config: dict, metrics: dict, n_obs: int, *, path: str | None = None) -> int:
    """Registra esta corrida en el registro de ensayos y devuelve cuántos ensayos únicos hay.

    El DSR descuenta la suerte de haber probado varias configuraciones; el conteo sale del
    registro (`data/eval/trials.jsonl`) y no de folds × features. Es una cota conservadora:
    cuenta todos los ensayos registrados, no solo los de esta estrategia. Es idempotente:
    repetir la misma configuración no infla el conteo. Mínimo 1.
    """
    path = path or str(ROOT / TRIALS_PATH)
    log_trial("purged_cv:ensemble", config, metrics, n_obs, path=path)
    return max(count_trials(path=path), 1)


def dsr_from_fold_sharpes(annual_sharpes, horizon: int, n_trials: int, n_obs: int) -> float:
    """DSR a partir de Sharpes de fold ANUALIZADOS (×√(252/horizon)).

    `deflated_sharpe_ratio` exige Sharpe POR PERIODO: se des-anualiza aquí, y
    `sr_std` es la desviación (ddof=1) del Sharpe por periodo entre folds.
    `n_obs` = nº de observaciones (señales) detrás de cada Sharpe.
    """
    per = np.asarray(annual_sharpes, dtype=float) / np.sqrt(252 / horizon)
    sr_std = float(per.std(ddof=1)) if len(per) > 1 else None
    if sr_std is not None and sr_std <= 0:
        sr_std = None
    return float(deflated_sharpe_ratio(float(per.mean()), n_trials, max(int(n_obs), 2),
                                       sr_std=sr_std))


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  Purged Walk-Forward CV  |  EnsembleModel (WoE+LR+LightGBM)")
    print(f"  {N_SPLITS} folds  |  embargo={EMBARGO}d  |  horizonte={HORIZON}d")
    print("=" * 65 + "\n")

    df = build_dataset()

    print(f"\nEjecutando {N_SPLITS} folds purged CV...\n")
    results = evaluate_purged_cv(df)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    agg = results["aggregate"]
    print(f"\n{'='*65}")
    print(f"  RESULTADOS PURGED CV")
    print(f"{'='*65}")
    print(f"  AUC:              {agg['mean_auc']:.4f} ± {results['aggregate'].get('std_auc',0):.4f}")
    print(f"  Sharpe:           {agg['mean_sharpe']:.2f}")
    print(f"  Win rate:         {agg['mean_win_rate']:.0%}")
    print(f"  DSR:              {agg['deflated_sharpe_ratio']:.4f}  → {agg['dsr_interpretation']}")
    print(f"  Total señales:    {agg['total_signals']}")
    print(f"{'='*65}\n")

    out_path = ROOT / "data" / "purged_cv_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
