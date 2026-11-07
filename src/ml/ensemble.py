"""
EnsembleModel — WoE + LR + LightGBM combinado.

Este es el modelo de producción (Ensemble v1):
  - WoE transforma las features tabulares
  - LR captura relaciones lineales
  - LightGBM captura interacciones no lineales
  - Voto ponderado por AUC individual

Diseñado para ser rápido (~3-5 seg para fit) y reutilizable en:
  - generate_signals.py  (señales del día)
  - simulate_week.py     (backtesting por semana)
  - run_ensemble.py      (evaluación holdout 80/20)

Uso:
    from src.ml.ensemble import EnsembleModel

    model = EnsembleModel()
    model.fit(train_df, feature_cols=FEATURE_COLS)

    # Predice P(subida en 5 días) para cada fila de test_df
    p = model.predict_proba(test_df, feature_cols=FEATURE_COLS)

    # Guardar y cargar (serializa todo el pipeline)
    model.save("data/ensemble_model.pkl")
    model2 = EnsembleModel.load("data/ensemble_model.pkl")
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Parámetros fijos de LightGBM (buenos sin necesitar Optuna)
# Obtenidos del mejor trial de Optuna en run_ensemble.py
_LGBM_PARAMS = dict(
    objective         = "binary",
    verbosity         = -1,
    n_estimators      = 300,
    num_leaves        = 55,
    learning_rate     = 0.026,
    min_child_samples = 59,
    feature_fraction  = 0.57,
    bagging_fraction  = 0.92,
    bagging_freq      = 1,
    reg_alpha         = 0.0002,
    reg_lambda        = 8.6,
    n_jobs            = 2,
    random_state      = 42,
)

TOP_FEATURES = 12   # features WoE a usar


def break_ties(p_cal: np.ndarray, p_raw: np.ndarray) -> np.ndarray:
    """Desempata `p_cal` (mesetas de la isotónica) con el score crudo `p_raw`.

    Devuelve p_cal + un desplazamiento < la mitad de la menor brecha entre
    valores distintos de p_cal, ordenado por p_raw dentro de cada meseta. Así el
    orden entre valores distintos de p_cal no cambia, y solo se reordenan los
    empates. Determinista: no depende del orden de las filas (empates exactos
    en ambos scores siguen empatados).
    """
    p_cal = np.asarray(p_cal, dtype=float)
    p_raw = np.asarray(p_raw, dtype=float)
    u = np.unique(p_cal)
    gap = np.diff(u).min() if len(u) > 1 else 1.0
    r = pd.Series(p_raw).groupby(p_cal).rank(method="average", pct=True).to_numpy()
    return p_cal + (r - 0.5) * 0.9 * gap / 2 * 0.999


class EnsembleModel:
    """
    Ensemble v1: WoE + Logistic Regression + LightGBM.

    Atributos públicos tras fit():
        auc_lr    (float): AUC del LR en datos de validación interna
        auc_lgbm  (float): AUC del LightGBM en datos de validación interna
        w_lr      (float): peso del LR en el voto final
        w_lgbm    (float): peso del LightGBM en el voto final
        top_feats (list):  features seleccionadas por IV
        iv_df     (DataFrame): tabla IV de todos los features
    """

    def __init__(self, top_features: int = TOP_FEATURES, calibrate: bool = True,
                 use_mlp: bool = False, mlp_seed: int = 42):
        self.top_features = top_features
        self.calibrate    = calibrate
        self.use_mlp      = use_mlp
        self.mlp_seed     = mlp_seed
        self._woe_maps    = None
        self._scaler      = None
        self._lr          = None
        self._lgbm        = None
        self._mlp         = None
        self._calibrator  = None   # isotónica P_raw → P_calibrada
        self.top_feats: list[str] = []
        self.auc_lr   = 0.5
        self.auc_lgbm = 0.5
        self.auc_mlp  = 0.5
        self.w_lr     = 0.5
        self.w_lgbm   = 0.5
        self.w_mlp    = 0.0
        self.iv_df: pd.DataFrame | None = None
        self._fitted = False

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "y",
        val_ratio: float = 0.15,
    ) -> "EnsembleModel":
        """
        Entrena WoE + LR + LightGBM sobre df.

        Args:
            df:           DataFrame con features y target (índice temporal).
            feature_cols: Lista de columnas a usar como features.
            target_col:   Columna binaria 0/1 (target).
            val_ratio:    Fracción final del train para estimar AUC internamente.
        """
        from src.ml.woe import compute_all_woe, woe_transform, iv_summary

        df = df.dropna(subset=feature_cols + [target_col]).copy()
        # Reset index para evitar duplicados por múltiples tickers en la misma fecha
        df = df.reset_index(drop=True)

        # Split interno temporal para estimar pesos
        n_val   = int(len(df) * val_ratio)
        df_tr   = df.iloc[:-n_val] if n_val > 0 else df
        df_val  = df.iloc[-n_val:] if n_val > 0 else df

        # ── WoE ──────────────────────────────────────────────────────────────
        self._woe_maps = compute_all_woe(df_tr, feature_cols, target_col=target_col)
        self.iv_df     = iv_summary(self._woe_maps)
        self.top_feats = self.iv_df.head(self.top_features)["feature"].tolist()

        X_tr_woe  = woe_transform(df_tr[feature_cols],  self._woe_maps)[self.top_feats]
        X_val_woe = woe_transform(df_val[feature_cols], self._woe_maps)[self.top_feats]

        mask_tr  = X_tr_woe.notna().all(axis=1)
        mask_val = X_val_woe.notna().all(axis=1)
        X_tr_woe  = X_tr_woe[mask_tr];   y_tr  = df_tr[target_col][mask_tr]
        X_val_woe = X_val_woe[mask_val];  y_val = df_val[target_col][mask_val]

        # ── LR ───────────────────────────────────────────────────────────────
        self._scaler = StandardScaler()
        X_tr_sc  = self._scaler.fit_transform(X_tr_woe)
        X_val_sc = self._scaler.transform(X_val_woe)

        self._lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self._lr.fit(X_tr_sc, y_tr)
        p_lr_val   = self._lr.predict_proba(X_val_sc)[:, 1]
        self.auc_lr = roc_auc_score(y_val, p_lr_val) if len(np.unique(y_val)) > 1 else 0.5

        # ── LightGBM ─────────────────────────────────────────────────────────
        try:
            import lightgbm as lgb
            self._lgbm = lgb.LGBMClassifier(**_LGBM_PARAMS)
            self._lgbm.fit(X_tr_woe.values, y_tr.values)
            p_lgbm_val   = self._lgbm.predict_proba(X_val_woe.values)[:, 1]
            self.auc_lgbm = roc_auc_score(y_val, p_lgbm_val) if len(np.unique(y_val)) > 1 else 0.5
        except ImportError:
            self._lgbm    = None
            self.auc_lgbm = 0.5

        # ── MLP (red neuronal regularizada) — votante opcional ───────────────
        p_mlp_val = None
        if self.use_mlp:
            try:
                from sklearn.neural_network import MLPClassifier
                self._mlp = MLPClassifier(
                    hidden_layer_sizes=(128, 64), alpha=1e-1,
                    max_iter=150, early_stopping=True, n_iter_no_change=8,
                    random_state=self.mlp_seed,
                )
                self._mlp.fit(X_tr_sc, y_tr)
                p_mlp_val = self._mlp.predict_proba(X_val_sc)[:, 1]
                self.auc_mlp = roc_auc_score(y_val, p_mlp_val) if len(np.unique(y_val)) > 1 else 0.5
            except Exception:
                self._mlp = None
                self.auc_mlp = 0.5

        # ── Pesos proporcionales al edge (AUC - 0.5) ─────────────────────────
        w_lr    = max(self.auc_lr    - 0.50, 0.001)
        w_lgbm  = max(self.auc_lgbm  - 0.50, 0.001)
        w_mlp   = max(self.auc_mlp   - 0.50, 0.0) if self._mlp is not None else 0.0
        total   = w_lr + w_lgbm + w_mlp
        self.w_lr   = w_lr   / total
        self.w_lgbm = w_lgbm / total
        self.w_mlp  = w_mlp  / total

        # ── Calibración isotónica ────────────────────────────────────────────
        # Entrena sobre el set de validación (no visto por LR/LGBM en su fit)
        # para que P=0.55 signifique de verdad 55% de aciertos.
        self._calibrator = None
        if self.calibrate and len(np.unique(y_val)) > 1:
            try:
                from sklearn.isotonic import IsotonicRegression
                p_ens_val = p_lr_val * self.w_lr
                if self._lgbm is not None:
                    p_ens_val = p_ens_val + p_lgbm_val * self.w_lgbm
                if p_mlp_val is not None:
                    p_ens_val = p_ens_val + p_mlp_val * self.w_mlp
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(p_ens_val, y_val.values)
                self._calibrator = iso
            except Exception:
                self._calibrator = None

        self._fitted = True
        return self

    # ── Predicción ────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        """
        Devuelve P(subida) para cada fila de df.
        Shape: (n_rows,)
        """
        if not self._fitted:
            raise RuntimeError("Llama fit() antes de predict_proba()")

        from src.ml.woe import woe_transform

        X_woe = woe_transform(df[feature_cols], self._woe_maps)[self.top_feats]
        X_woe = X_woe.fillna(0.0)  # imputar con WoE neutro

        X_sc  = self._scaler.transform(X_woe)
        p_lr  = self._lr.predict_proba(X_sc)[:, 1]

        if self._lgbm is not None:
            p_lgbm = self._lgbm.predict_proba(X_woe.values)[:, 1]
        else:
            p_lgbm = p_lr

        p_ens = p_lr * self.w_lr + p_lgbm * self.w_lgbm

        if self._mlp is not None:
            p_mlp = self._mlp.predict_proba(X_sc)[:, 1]
            p_ens = p_ens + p_mlp * self.w_mlp

        # Aplicar calibración isotónica si está disponible
        if self._calibrator is not None:
            p_ens = self._calibrator.predict(p_ens)

        return p_ens

    def predict_proba_raw(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        """P(subida) SIN calibrar — útil para meta-labeling o diagnóstico."""
        if not self._fitted:
            raise RuntimeError("Llama fit() antes de predict_proba_raw()")
        from src.ml.woe import woe_transform
        X_woe = woe_transform(df[feature_cols], self._woe_maps)[self.top_feats].fillna(0.0)
        X_sc  = self._scaler.transform(X_woe)
        p_lr  = self._lr.predict_proba(X_sc)[:, 1]
        p_lgbm = self._lgbm.predict_proba(X_woe.values)[:, 1] if self._lgbm is not None else p_lr
        p_ens = p_lr * self.w_lr + p_lgbm * self.w_lgbm
        if self._mlp is not None:
            p_ens = p_ens + self._mlp.predict_proba(X_sc)[:, 1] * self.w_mlp
        return p_ens

    def predict_proba_tiebroken(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        """`predict_proba` con las mesetas de la calibración desempatadas por el
        score crudo (ver `break_ties`). Úsalo para rankear (top-K); NO para leer
        probabilidades. `predict_proba` por defecto no cambia."""
        return break_ties(self.predict_proba(df, feature_cols),
                          self.predict_proba_raw(df, feature_cols))

    def predict_ticker(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        date: pd.Timestamp | None = None,
    ) -> float:
        """
        Devuelve P(subida) para una acción en una fecha específica (o la última).
        """
        if date is not None:
            row = df.loc[[date]]
        else:
            row = df.iloc[[-1]]
        p = self.predict_proba(row, feature_cols)
        return float(p[0])

    # ── Serialización ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Guarda el modelo completo en disco (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleModel":
        """Carga un modelo previamente guardado."""
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Resumen ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            "EnsembleModel (WoE + LR + LightGBM)",
            f"  Features:   {len(self.top_feats)} de {len(self.iv_df) if self.iv_df is not None else '?'}",
            f"  AUC LR:     {self.auc_lr:.4f}  (peso {self.w_lr:.0%})",
            f"  AUC LGBM:   {self.auc_lgbm:.4f}  (peso {self.w_lgbm:.0%})",
            f"  Top feats:  {', '.join(self.top_feats[:5])}...",
        ]
        return "\n".join(lines)
