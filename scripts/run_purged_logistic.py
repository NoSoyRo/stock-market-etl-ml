"""
Evaluación de WoE + Logistic Regression con Purged Walk-Forward CV y Embargo.

Metodología:
  - Horizonte nominal: 5 días
  - Entrada causal: Open de t+1
  - Salida: Close de t+1+horizon (t+6)
  - Label span para purging: 6 días
  - Purged Walk-Forward por FECHAS (panel multi-ticker)
  - Folds: 6
  - Embargo: 10 días entre train y test
  - Modelo: LogisticRegression(C=0.1, max_iter=1000, random_state=42)
  - Transformación: WoE ajustado estrictamente en el set de entrenamiento de cada fold
  - Selección de features: Top 15 ordenadas por IV en el set de entrenamiento de cada fold
  - Valores fuera de rango o NaN en test tratados como WoE neutral (0.0)
  - Retornos forward reales para cálculo de Sharpe, IC y Profit Factor (sin proxy +-1%)

Ejecución:
    python scripts/run_purged_logistic.py
"""
from __future__ import annotations

import datetime
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from scripts.run_purged_cv import (
    _profit_factor,
    _safe_distribution_stats,
    build_dataset,
    dsr_from_fold_sharpes,
)
from src.ml.features import FEATURE_COLS
from src.ml.purged_cv import purged_walk_forward_date_splits
from src.ml.targets import label_span
from src.ml.trials import DEFAULT_PATH as TRIALS_PATH, count_trials, log_trial
from src.ml.woe import FeatureWoE, compute_all_woe

# ─── Hiperparámetros y configuración metodológica ────────────────────────────

HORIZON = 5
N_SPLITS = 6
EMBARGO = 10
PROB_THRESH = 0.50
C_REG = 0.1
TOP_FEATURES = 15
N_BINS_WOE = 8

# Con entrada en Open t+1 y salida en Close t+6, el span para purging es 6 días.
LABEL_SPAN = label_span("dir_5", entry="next_open")

all_candidate_cols = FEATURE_COLS


# ─── Transformación WoE causal y anti-leakage ─────────────────────────────────

def transform_woe_features(
    df: pd.DataFrame,
    features: list[str],
    woe_map: dict[str, FeatureWoE],
) -> np.ndarray:
    """Aplica el mapa WoE aprendido en entrenamiento a un DataFrame.

    Restricciones causales:
      - Valores fuera del rango observado en train quedan con WoE neutral (0.0).
      - Valores faltantes (NaN) se imputan con WoE neutral (0.0 = odds ratio 1.0).
    """
    transformed = []
    for feat in features:
        fw = woe_map[feat]
        s_woe = fw.transform(df[feat]).fillna(0.0).astype(float)
        transformed.append(s_woe.to_numpy())

    if not transformed:
        return np.empty((len(df), 0), dtype=float)

    return np.column_stack(transformed)


# ─── Registro en bitácora de trials ──────────────────────────────────────────

def trials_for_dsr(
    config: dict,
    metrics: dict,
    n_obs: int,
    *,
    path: str | None = None,
) -> int:
    """Registra el ensayo de forma idempotente y retorna el total de trials únicos."""
    path = path or str(ROOT / TRIALS_PATH)
    log_trial(
        "purged_cv:woe_logistic",
        config,
        metrics,
        n_obs,
        path=path,
    )
    return max(count_trials(path=path), 1)


# ─── Evaluación Purged CV ─────────────────────────────────────────────────────

def evaluate_purged_logistic(df: pd.DataFrame) -> dict:
    # Excluir observaciones con ret_fwd no finito (divisiones por cero cuando Open[t+1] == 0 en datos crudos)
    df = df[np.isfinite(df["ret_fwd"])].copy()
    df_reset = df.reset_index(drop=False)

    if "Date" in df_reset.columns:
        dates = pd.to_datetime(df_reset["Date"])
    else:
        dates = pd.to_datetime(df_reset.iloc[:, 0])

    available_candidates = [
        col for col in all_candidate_cols if col in df_reset.columns
    ]

    folds_results: list[dict] = []
    all_rets: list[np.ndarray] = []

    for fold_i, (tr_idx, te_idx) in enumerate(
        purged_walk_forward_date_splits(
            dates,
            n_splits=N_SPLITS,
            horizon=LABEL_SPAN,
            embargo=EMBARGO,
        )
    ):
        X_tr = df_reset.iloc[tr_idx]
        y_tr = df_reset.iloc[tr_idx]["y"].to_numpy(dtype=int)
        X_te = df_reset.iloc[te_idx]
        y_te = df_reset.iloc[te_idx]["y"].to_numpy(dtype=int)
        rets_te_all = df_reset.iloc[te_idx]["ret_fwd"].to_numpy(dtype=float) / 100

        # Verificar que test y train contengan ambas clases
        if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
            continue

        # 1. Ajustar WoE estrictamente en train del fold
        train_feat_df = X_tr[available_candidates].copy()
        train_feat_df["y"] = y_tr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            woe_map = compute_all_woe(
                train_feat_df,
                feature_cols=available_candidates,
                target_col="y",
                n_bins=N_BINS_WOE,
            )

        # 2. Selección de features estrictamente en train:
        #    - Bins válidos (n_bins >= 2)
        #    - IV positivo y finito
        #    - Orden descendente por IV
        #    - Top 15 features
        valid_features = [
            feat
            for feat, fw in woe_map.items()
            if fw.n_bins >= 2 and fw.iv > 0.0 and np.isfinite(fw.iv)
        ]
        valid_features.sort(key=lambda f: woe_map[f].iv, reverse=True)
        selected_features = valid_features[:TOP_FEATURES]

        if not selected_features:
            continue

        # 3. Transformación a WoE aplicando mapa de train
        X_tr_woe = transform_woe_features(X_tr, selected_features, woe_map)
        X_te_woe = transform_woe_features(X_te, selected_features, woe_map)

        # 4. Entrenar Logistic Regression determinista
        lr = LogisticRegression(
            C=C_REG,
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )
        lr.fit(X_tr_woe, y_tr)

        # 5. Predecir probabilidades sobre test
        p_te = lr.predict_proba(X_te_woe)[:, 1]

        # 6. Calcular métricas del fold
        auc = float(roc_auc_score(y_te, p_te))

        # IC: Spearman rank correlation entre probabilidad y retorno forward real
        if len(p_te) > 1 and np.std(p_te) > 1e-12 and np.std(rets_te_all) > 1e-12:
            ic = float(stats.spearmanr(p_te, rets_te_all).statistic)
            if not np.isfinite(ic):
                ic = 0.0
        else:
            ic = 0.0

        # Señales de la estrategia con threshold fijo
        sigs = p_te >= PROB_THRESH
        n_signals = int(sigs.sum())

        if n_signals > 0:
            rets = rets_te_all[sigs]
            all_rets.append(rets)

            if len(rets) > 1:
                ret_std = float(np.std(rets, ddof=1))
            else:
                ret_std = 0.0

            if ret_std > 1e-12:
                sharpe = float(np.mean(rets)) / ret_std * np.sqrt(252 / HORIZON)
            else:
                sharpe = 0.0

            win_rate = float((rets > 0).mean())
            profit_factor = _profit_factor(rets)
        else:
            sharpe = 0.0
            win_rate = 0.0
            profit_factor = None

        test_start = str(dates.iloc[te_idx].min())[:10]
        test_end = str(dates.iloc[te_idx].max())[:10]

        fold_res = {
            "fold": fold_i + 1,
            "test_start": test_start,
            "test_end": test_end,
            "n_test": int(len(te_idx)),
            "n_train": int(len(tr_idx)),
            "auc": round(float(auc), 4),
            "ic": round(float(ic), 4),
            "sharpe": round(float(sharpe), 3),
            "profit_factor": (
                round(float(profit_factor), 3) if profit_factor is not None else None
            ),
            "win_rate": round(float(win_rate), 3),
            "n_signals": n_signals,
            "n_features": len(selected_features),
            "selected_features": selected_features,
        }
        folds_results.append(fold_res)

        pf_text = f"{profit_factor:.2f}" if profit_factor is not None else "N/A"
        print(
            f"  Fold {fold_i + 1}  "
            f"[{test_start} -> {test_end}]  "
            f"AUC={auc:.4f}  "
            f"IC={ic:.4f}  "
            f"Sharpe={sharpe:.2f}  "
            f"PF={pf_text}  "
            f"WR={win_rate:.0%}  "
            f"signals={n_signals}  "
            f"features={len(selected_features)}  "
            f"train={len(tr_idx):,}  "
            f"test={len(te_idx):,}"
        )

    if not folds_results:
        return {"error": "Sin folds validos"}

    aucs = [row["auc"] for row in folds_results]
    ics = [row["ic"] for row in folds_results]
    sharpes = [row["sharpe"] for row in folds_results]
    wrs = [row["win_rate"] for row in folds_results]

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics))
    mean_sharpe = float(np.mean(sharpes))
    mean_wr = float(np.mean(wrs))

    pooled = (
        np.concatenate(all_rets) if all_rets else np.array([], dtype=float)
    )
    pooled_profit_factor = _profit_factor(pooled) if len(pooled) > 0 else None
    total_signals = int(sum(row["n_signals"] for row in folds_results))

    skew, kurt = _safe_distribution_stats(pooled)

    n_obs = int(np.mean([row["n_signals"] for row in folds_results])) if folds_results else 0

    trial_config = {
        "model": "woe_logistic",
        "C": C_REG,
        "n_splits": N_SPLITS,
        "embargo": EMBARGO,
        "horizon": HORIZON,
        "label_span": LABEL_SPAN,
        "entry": "next_open",
        "split_unit": "dates",
        "threshold": PROB_THRESH,
        "n_features": TOP_FEATURES,
        "n_bins": N_BINS_WOE,
    }

    trial_metrics = {
        "mean_sharpe": mean_sharpe,
        "mean_auc": mean_auc,
        "mean_ic": mean_ic,
    }

    n_trials = trials_for_dsr(
        trial_config,
        trial_metrics,
        n_obs,
    )

    dsr = dsr_from_fold_sharpes(
        sharpes,
        HORIZON,
        n_trials,
        n_obs,
        skewness=skew,
        kurtosis=kurt,
    )

    dsr_interpretation = (
        "Edge estadisticamente significativo"
        if dsr > 0.95
        else "Edge moderado, monitorear"
        if dsr > 0.75
        else "Edge debil, posible sobreajuste"
    )

    aggregate = {
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(std_auc, 4),
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "mean_sharpe": round(mean_sharpe, 3),
        "profit_factor": (
            round(float(pooled_profit_factor), 3)
            if pooled_profit_factor is not None
            else None
        ),
        "mean_win_rate": round(mean_wr, 3),
        "deflated_sharpe_ratio": round(float(dsr), 4),
        "dsr_interpretation": dsr_interpretation,
        "total_signals": total_signals,
        "n_trials": int(n_trials),
        "skew": round(skew, 3),
        "kurtosis": round(kurt, 3),
    }

    meta = {
        "model": "WoE + Logistic Regression",
        "execution_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target": "binary_label(forward_return_open)",
        "entry": "next_open",
        "exit": f"close_t+1+{HORIZON}",
        "horizon": HORIZON,
        "label_span": LABEL_SPAN,
        "embargo": EMBARGO,
        "folds": N_SPLITS,
        "split_unit": "dates",
        "C": C_REG,
        "threshold": PROB_THRESH,
        "bins_woe": N_BINS_WOE,
        "feature_selection_criteria": (
            f"Top {TOP_FEATURES} features por Information Value (IV) en train; "
            "bins_validos >= 2; valores fuera de rango o NaN tratados como WoE neutral 0.0"
        ),
    }

    return {
        "meta": meta,
        "folds": folds_results,
        "aggregate": aggregate,
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 72)
    print("  Purged Walk-Forward CV  |  WoE + Logistic Regression (C=0.1)")
    print(
        f"  {N_SPLITS} folds  |  embargo={EMBARGO}d  |  horizonte={HORIZON}d  |  "
        f"label_span={LABEL_SPAN}d  |  split=fechas"
    )
    print("=" * 72 + "\n")

    df = build_dataset()

    print(
        f"\nEjecutando {N_SPLITS} folds Purged CV por fechas...\n"
    )

    results = evaluate_purged_logistic(df)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    agg = results["aggregate"]

    print(f"\n{'=' * 72}")
    print("  RESULTADOS PURGED CV (WoE + Logistic Regression)")
    print(f"{'=' * 72}")
    print(f"  AUC:              {agg['mean_auc']:.4f} +- {agg['std_auc']:.4f}")
    print(f"  IC (Spearman):    {agg['mean_ic']:.4f} +- {agg['std_ic']:.4f}")
    print(f"  Sharpe:           {agg['mean_sharpe']:.2f}")

    pf = agg["profit_factor"]
    pf_text = f"{pf:.2f}" if pf is not None else "N/A"
    print(f"  Profit Factor:     {pf_text}")

    print(f"  Win rate:         {agg['mean_win_rate']:.0%}")
    print(
        f"  DSR:              {agg['deflated_sharpe_ratio']:.4f}  -> "
        f"{agg['dsr_interpretation']}"
    )
    print(f"  Total senales:    {agg['total_signals']}")
    print(f"  Trials DSR:       {agg['n_trials']}")
    print(f"  Skew / Kurt:      {agg['skew']:.2f} / {agg['kurtosis']:.2f}")
    print(f"{'=' * 72}\n")

    out_dir = ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "purged_logistic_results.json"

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