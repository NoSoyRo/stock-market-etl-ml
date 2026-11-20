"""
Evaluación: Ensemble base  vs  Ensemble + Meta-Labeling + Triple-Barrier.

Compara en hold-out temporal (80/20):
  - Modelo A: EnsembleModel solo (umbral fijo sobre P)
  - Modelo B: EnsembleModel + MetaModel (meta decide si actuar)

Ambos con labels triple-barrier (alineados con stops/targets reales).

Ejecución:
    python scripts/run_meta.py
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

from src.data.store   import get as store_get, BMV_TICKERS, MACRO_TICKERS
from src.ml.features  import (compute_features, add_macro_features,
                              add_cross_section_features, FEATURE_COLS)
from src.ml.labeling  import triple_barrier_labels, sample_uniqueness_weights, barrier_stats
from src.ml.ensemble  import EnsembleModel
from src.ml.meta_model import MetaModel

HORIZON      = 5
MULT_TARGET  = 1.5
MULT_STOP    = 1.5
TRAIN_RATIO  = 0.80
PRIMARY_THRESH = 0.53     # umbral del modelo primario
META_THRESH    = 0.50     # umbral del meta-modelo
COMM         = 0.0025


def build_dataset(horizon: int = HORIZON,
                  mult_target: float = MULT_TARGET,
                  mult_stop: float = MULT_STOP) -> pd.DataFrame:
    """Construye dataset con features + labels triple-barrier por ticker."""
    macro = {n: store_get(n) for n in MACRO_TICKERS}
    macro = {k: v for k, v in macro.items() if v is not None}

    feat_frames = {}
    for ticker in BMV_TICKERS:
        raw = store_get(ticker)
        if raw is None or len(raw) < 200:
            continue
        try:
            feat = compute_features(raw)
            feat = add_macro_features(feat, macro)

            # Triple-barrier labels (alineados con stops/targets reales)
            # atr_pct viene en % (ej. 2.44) → convertir a fracción (0.0244)
            vol = (feat["atr_pct"] / 100.0) if "atr_pct" in feat.columns else \
                  raw["Close"].pct_change().rolling(14).std()
            tb = triple_barrier_labels(
                close=raw["Close"], high=raw["High"], low=raw["Low"],
                vol=vol, horizon=horizon,
                mult_target=mult_target, mult_stop=mult_stop,
            )
            feat["y"]       = tb["label"]
            feat["tb_ret"]  = tb["ret"]
            feat["tb_bars"] = tb["bars"]
            feat["touch"]   = tb["touch"]
            feat["ticker"]  = ticker
            feat_frames[ticker] = feat
        except Exception as e:
            print(f"  {ticker}: {e}")

    # Features cross-seccionales (ranking vs universo)
    feat_frames = add_cross_section_features(feat_frames)

    df = pd.concat(feat_frames.values()).sort_index()
    df = df.dropna(subset=FEATURE_COLS + ["y"])
    return df


def strategy_metrics(rets: np.ndarray) -> dict:
    """Métricas de estrategia sobre los retornos netos de los trades tomados.

    `rets` son retornos por operación individual (múltiples señales por semana
    solapadas entre tickers), no periodos disjuntos de 5 días: se reporta
    `sharpe_per_trade` (media/std sin anualizar) y se mantiene `sharpe` por
    compatibilidad con salidas previas.
    """
    if len(rets) == 0:
        return {"n": 0, "win_rate": 0, "avg_ret": 0, "sharpe_per_trade": 0,
                "sharpe": 0, "profit_factor": 0, "total_ret": 0}
    rets = np.asarray(rets) / 100  # a fracción
    rets_net = rets - COMM * 2
    wins = (rets_net > 0).sum()
    pf_pos = rets_net[rets_net > 0].sum()
    pf_neg = abs(rets_net[rets_net < 0].sum())
    sr_per_trade = float(rets_net.mean() / (rets_net.std() + 1e-9))
    return {
        "n":                int(len(rets)),
        "win_rate":         round(float(wins / len(rets)), 4),
        "avg_ret":          round(float(rets_net.mean() * 100), 4),
        "sharpe_per_trade": round(sr_per_trade, 4),
        "sharpe":           round(float(sr_per_trade * np.sqrt(252 / HORIZON)), 3),
        "profit_factor":    round(float(pf_pos / pf_neg) if pf_neg > 0 else 0.0, 3),
        "total_ret":        round(float(rets_net.sum() * 100), 2),
    }


def main():
    print("\n" + "=" * 66)
    print("  Meta-Labeling vs Ensemble base  |  Triple-Barrier + Calibración")
    print("=" * 66 + "\n")

    print("[1] Construyendo dataset con labels triple-barrier...")
    df = build_dataset()
    bstats = barrier_stats(df.rename(columns={"y": "label", "tb_ret": "ret", "tb_bars": "bars"}))
    print(f"    {len(df):,} obs  |  win_rate={bstats.get('win_rate',0):.1%}  "
          f"target={bstats.get('pct_target',0):.0%}  "
          f"stop={bstats.get('pct_stop',0):.0%}  "
          f"time={bstats.get('pct_time',0):.0%}")

    # ── Corte temporal 80/20 ───────────────────────────────────────────────────
    df = df.sort_index()
    cut = int(len(df) * TRAIN_RATIO)
    # Cortar por fecha para no partir un día
    cut_date = df.index[cut]
    train = df[df.index < cut_date].copy()
    test  = df[df.index >= cut_date].copy()
    print(f"\n[2] Train: {len(train):,} ({train.index.min().date()}→{train.index.max().date()})")
    print(f"    Test:  {len(test):,} ({test.index.min().date()}→{test.index.max().date()})")

    # ── 3. Entrenar modelo primario (Ensemble) ────────────────────────────────
    print("\n[3] Entrenando modelo primario (EnsembleModel calibrado)...")
    primary = EnsembleModel(top_features=15, calibrate=True)
    primary.fit(train, feature_cols=FEATURE_COLS, target_col="y")
    print(f"    AUC LR={primary.auc_lr:.4f}  LGBM={primary.auc_lgbm:.4f}")

    p_train = primary.predict_proba(train, FEATURE_COLS)
    p_test  = primary.predict_proba(test,  FEATURE_COLS)
    auc_primary = roc_auc_score(test["y"], p_test)
    print(f"    AUC primario en test: {auc_primary:.4f}")

    # ── 4. Entrenar meta-modelo ────────────────────────────────────────────────
    print("\n[4] Entrenando meta-modelo (decide SI tomar la señal)...")
    # Pesos de unicidad
    w_train = sample_uniqueness_weights(train["tb_bars"], horizon=HORIZON)

    meta = MetaModel()
    meta.fit(
        train, feature_cols=FEATURE_COLS,
        primary_proba=p_train,
        meta_label=train["y"].values,
        sample_weight=w_train.values,
    )
    print(f"    {meta.summary()}")

    meta_p_test = meta.predict_proba(test, FEATURE_COLS, primary_proba=p_test)

    # ── 5. Comparar estrategias ────────────────────────────────────────────────
    print("\n[5] Comparando estrategias en test...\n")

    test = test.reset_index(drop=True)

    # Modelo A: Ensemble solo (P >= umbral primario)
    mask_A = np.asarray(p_test >= PRIMARY_THRESH)
    rets_A = test.loc[mask_A, "tb_ret"].values
    metrics_A = strategy_metrics(rets_A)

    # Umbral del meta = mediana de los candidatos primarios (toma la mitad superior por confianza)
    mask_A = np.asarray(mask_A)
    meta_among_candidates = meta_p_test[mask_A]
    if len(meta_among_candidates) > 0:
        meta_cut = float(np.quantile(meta_among_candidates, 0.50))
    else:
        meta_cut = META_THRESH

    # Modelo B: Ensemble + Meta (señal primaria Y meta en mitad superior de confianza)
    mask_B = mask_A & (meta_p_test >= meta_cut)
    rets_B = test.loc[mask_B, "tb_ret"].values
    metrics_B = strategy_metrics(rets_B)

    def _print(name, m):
        print(f"  {name}")
        print(f"    Trades: {m['n']:<5}  Win rate: {m['win_rate']*100:5.1f}%  "
              f"Sharpe: {m['sharpe']:5.2f}  PF: {m['profit_factor']:.2f}  "
              f"Avg ret: {m['avg_ret']:+.3f}%  Total: {m['total_ret']:+.1f}%")

    _print("A) Ensemble base   (P≥0.53)", metrics_A)
    _print(f"B) Ensemble + Meta (P≥0.53 & meta≥{meta_cut:.3f})", metrics_B)

    # Mejora
    print(f"\n  {'─'*60}")
    wr_delta = (metrics_B["win_rate"] - metrics_A["win_rate"]) * 100
    pf_delta = metrics_B["profit_factor"] - metrics_A["profit_factor"]
    print(f"  Δ Win rate:      {wr_delta:+.1f} puntos")
    print(f"  Δ Profit Factor: {pf_delta:+.2f}")
    print(f"  Trades filtrados: {metrics_A['n'] - metrics_B['n']} "
          f"({(1 - metrics_B['n']/max(metrics_A['n'],1))*100:.0f}% menos)")

    # ── Barrido de selectividad del meta-modelo ────────────────────────────────
    print(f"\n  {'─'*60}")
    print("  Barrido de selectividad (solo los trades de MÁS confianza del meta):")
    print(f"  {'Percentil':>10}  {'Trades':>7}  {'WinRate':>8}  {'Sharpe':>7}  {'PF':>6}  {'TotalRet':>9}")
    sweep = []
    for q in [0.0, 0.50, 0.70, 0.80, 0.90, 0.95]:
        if mask_A.sum() == 0:
            break
        cut = float(np.quantile(meta_p_test[mask_A], q)) if q > 0 else 0.0
        m = mask_A & (meta_p_test >= cut)
        mm = strategy_metrics(test.loc[m, "tb_ret"].values)
        sweep.append({"quantile": q, "cut": round(cut, 4), **mm})
        print(f"  {q*100:>9.0f}%  {mm['n']:>7}  {mm['win_rate']*100:>7.1f}%  "
              f"{mm['sharpe']:>7.2f}  {mm['profit_factor']:>6.2f}  {mm['total_ret']:>+8.1f}%")

    # ── 6. Guardar resultados ──────────────────────────────────────────────────
    out = {
        "meta": {
            "horizon": HORIZON, "mult_target": MULT_TARGET, "mult_stop": MULT_STOP,
            "train_ratio": TRAIN_RATIO,
            "primary_thresh": PRIMARY_THRESH, "meta_thresh": META_THRESH,
            "train_start": str(train.index.min().date()),
            "train_end":   str(train.index.max().date()),
            "test_start":  str(test.index.min()) if len(test) else None,
            "n_train": len(train), "n_test": len(test),
        },
        "barrier_stats":   bstats,
        "auc_primary":     round(auc_primary, 4),
        "auc_meta":        round(meta.auc, 4),
        "meta_precision":  round(meta.precision, 4),
        "meta_recall":     round(meta.recall, 4),
        "meta_cut":         round(meta_cut, 4),
        "strategy_base":   metrics_A,
        "strategy_meta":   metrics_B,
        "selectivity_sweep": sweep,
        "improvement": {
            "win_rate_delta":      round(wr_delta, 2),
            "profit_factor_delta": round(pf_delta, 3),
            "trades_filtered":     metrics_A["n"] - metrics_B["n"],
        },
        "meta_feat_importance": meta.feat_imp.head(15).to_dict("records") if meta.feat_imp is not None else [],
    }
    out_path = ROOT / "data" / "meta_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\n{'='*66}")
    print(f"  Guardado: {out_path}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
