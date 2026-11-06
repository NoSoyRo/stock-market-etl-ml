"""M9 · Brazos del bake-off de modelos (interfaz común) y utilidades de combinación.

Todos los brazos comparten la interfaz ``fit(train)`` / ``predict(df) -> np.ndarray`` sobre un
DataFrame con índice de fechas (repetidas por ticker), columnas ``rn_*`` (percentil por fecha de
cada feature), ``pct_fwd`` (rank por fecha del retorno futuro) y ``fwd_ret``. Los scores son
continuos (sin calibración isotónica) para evitar empates.

Funciones puras y testeables: etiquetas por quintil, grupos por fecha para el ranker, split
interno purgado para el early stopping, pesos de combinación (1/N, AUC, IC con shrinkage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.shared.evaluators.cross_section import ic_by_date

HORIZON = 5
EMBARGO = 5

# Base del xs_lgbm actual (copia de super_model._LGBM_BASE; no se importa para no cargar todo).
LGBM_BASE = dict(
    n_estimators=500, num_leaves=31, learning_rate=0.02, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, reg_lambda=3.0,
    n_jobs=4, random_state=42, verbosity=-1,
)

# Signos a priori (teoría de reversión de corto plazo), FIJADOS ANTES de mirar ningún IC.
# -1: más alto el indicador (sobrecompra / retorno reciente) => menor retorno esperado.
# Lo no listado queda sin restricción (0).
MONOTONE_SIGNS = {
    "rn_ret_5d": -1, "rn_ret_10d": -1, "rn_cs_ret5_rank": -1,
    "rn_rsi_14": -1, "rn_stoch_k": -1, "rn_bb_pct": -1, "rn_williams_r": -1,
}


# ── Utilidades puras ─────────────────────────────────────────────────────────

def rank_features(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Copia del panel con ``rn_<col>`` = percentil por fecha de cada feature."""
    out = panel.copy()
    g = out.groupby(out.index)
    for c in cols:
        out[f"rn_{c}"] = g[c].rank(pct=True)
    return out


def quintile_labels(dates: pd.Index, ret: np.ndarray) -> np.ndarray:
    """Etiqueta entera 0–4 por quintil del retorno DENTRO de cada fecha (4 = mejor).

    Desempata por orden de aparición (rank "first") para repartir parejo.
    """
    s = pd.Series(np.asarray(ret, dtype=float), index=pd.Index(dates))
    g = s.groupby(level=0)
    r = g.rank(method="first")
    n = g.transform("size")
    return (((r - 1) * 5) // n).clip(upper=4).astype(int).to_numpy()


def date_groups(dates: pd.Index) -> np.ndarray:
    """Tamaños de grupo (un grupo = una fecha) para el ranker. Exige filas contiguas por fecha."""
    d = pd.DatetimeIndex(dates)
    if len(d) == 0:
        return np.array([], dtype=int)
    if not d.is_monotonic_increasing:
        raise ValueError("las filas deben estar ordenadas por fecha (grupos contiguos)")
    _, counts = np.unique(d.values, return_counts=True)
    return counts.astype(int)


def inner_purged_split(
    dates: pd.Index, val_frac: float = 0.15, horizon: int = HORIZON, embargo: int = EMBARGO,
) -> tuple[np.ndarray, np.ndarray]:
    """Split interno del train: validación = últimas ``val_frac`` de las fechas únicas.

    El train interno termina ``horizon + embargo`` fechas antes de la validación (purga), de modo
    que ninguna etiqueta del train interno se traslape con la ventana de validación.
    Devuelve posiciones de fila (tr_idx, va_idx).
    """
    d = pd.DatetimeIndex(dates)
    uniq = np.sort(d.unique())
    m = len(uniq)
    v0 = int(m * (1 - val_frac))
    pos = pd.Series(np.arange(m), index=uniq).reindex(d).to_numpy()
    tr = np.flatnonzero(pos < v0 - horizon - embargo)
    va = np.flatnonzero(pos >= v0)
    return tr, va


def mean_ic(dates: pd.Index, score: np.ndarray, ret: np.ndarray, min_names: int = 10) -> float:
    """IC medio de Spearman por fecha de un vector de scores (NaN si no hay fechas válidas)."""
    df = pd.DataFrame({"s": np.asarray(score, float), "r": np.asarray(ret, float)}, index=pd.Index(dates))
    ic = ic_by_date(df, "s", "r", min_names=min_names)
    return float(ic.mean()) if len(ic) else float("nan")


def tercile_buckets(dates: pd.Index, sigma: np.ndarray) -> np.ndarray:
    """Tercil (0 = menor σ̂, 2 = mayor) de σ̂ dentro de cada fecha; NaN => -1."""
    s = pd.Series(np.asarray(sigma, float), index=pd.Index(dates))
    g = s.groupby(level=0)
    r = g.rank(method="first")
    n = g.transform("count")
    b = (((r - 1) * 3) // n).clip(upper=2)
    return b.fillna(-1).astype(int).to_numpy()


def ic_by_tercile(panel: pd.DataFrame, score_col: str, ret_col: str, sigma_col: str,
                  *, min_names: int = 6) -> dict[int, pd.Series]:
    """IC por fecha dentro de cada tercil (por fecha) de σ̂. Devuelve {0,1,2: serie de IC}."""
    b = tercile_buckets(panel.index, panel[sigma_col].to_numpy())
    return {t: ic_by_date(panel[b == t], score_col, ret_col, min_names=min_names) for t in (0, 1, 2)}


# ── Brazos ───────────────────────────────────────────────────────────────────

class Arm:
    """Interfaz común: ``fit(train)`` y ``predict(df)``; ``config`` describe el ensayo."""

    name = "arm"

    def __init__(self, cols: list[str]):
        self.cols = list(cols)

    @property
    def config(self) -> dict:
        return {"arm": self.name, "n_cols": len(self.cols)}

    def fit(self, train: pd.DataFrame) -> "Arm":  # pragma: no cover - interfaz
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - interfaz
        raise NotImplementedError


class XsLgbmArm(Arm):
    """Modelo actual: LGBMRegressor sobre rn_* con objetivo ``pct_fwd`` (MSE)."""

    name = "xs_lgbm"

    def __init__(self, cols, params: dict | None = None):
        super().__init__(cols)
        self.params = {**LGBM_BASE, **(params or {})}

    @property
    def config(self) -> dict:
        return {**super().config, **self.params}

    def fit(self, train):
        import lightgbm as lgb
        self.model_ = lgb.LGBMRegressor(**self.params)
        self.model_.fit(train[self.cols].values, train["pct_fwd"].values)
        return self

    def predict(self, df):
        return self.model_.predict(df[self.cols].values)


class MonotoneArm(XsLgbmArm):
    """Igual que xs_lgbm pero con ``monotone_constraints`` (signos a priori de MONOTONE_SIGNS)."""

    name = "monotone_lgbm"

    def __init__(self, cols, params: dict | None = None):
        super().__init__(cols, params)
        self.signs = [MONOTONE_SIGNS.get(c, 0) for c in self.cols]
        self.params["monotone_constraints"] = self.signs


class RankerArm(Arm):
    """LGBMRanker con grupo = fecha y etiqueta entera 0–4 por quintil del retorno de la fecha."""

    def __init__(self, cols, objective: str = "lambdarank", params: dict | None = None):
        super().__init__(cols)
        if objective not in ("lambdarank", "rank_xendcg"):
            raise ValueError("objective debe ser lambdarank o rank_xendcg")
        self.objective = objective
        self.name = f"ranker_{objective}"
        self.params = {**LGBM_BASE, **(params or {})}

    @property
    def config(self) -> dict:
        return {**super().config, "objective": self.objective, **self.params}

    def fit(self, train):
        import lightgbm as lgb
        y = quintile_labels(train.index, train["fwd_ret"].to_numpy())
        grp = date_groups(train.index)
        self.model_ = lgb.LGBMRanker(objective=self.objective, **self.params)
        self.model_.fit(train[self.cols].values, y, group=grp)
        return self

    def predict(self, df):
        return self.model_.predict(df[self.cols].values)


class RidgeArm(Arm):
    """Piso lineal: Ridge (alpha fija, sin ajustar) sobre los ranks, objetivo ``pct_fwd``."""

    name = "ridge_rank"

    def __init__(self, cols, alpha: float = 100.0):
        super().__init__(cols)
        self.alpha = alpha

    @property
    def config(self) -> dict:
        return {**super().config, "alpha": self.alpha}

    def fit(self, train):
        from sklearn.linear_model import Ridge
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(train[self.cols].fillna(0.5).values - 0.5, train["pct_fwd"].values)
        return self

    def predict(self, df):
        return self.model_.predict(df[self.cols].fillna(0.5).values - 0.5)


class AntiNoiseArm(Arm):
    """Árboles chicos + varias semillas promediadas + early stopping por IC en bloque interno.

    El nº de árboles se elige SOLO con el train: se entrena en el train interno purgado, se mide el
    IC medio por fecha del promedio de semillas en el bloque de validación (últimas ~15% de fechas
    del train) en puntos de control, y se reentrena en el train completo con el mejor nº de árboles.
    """

    def __init__(self, cols, num_leaves: int = 8, min_child_samples: int = 300,
                 seeds: tuple[int, ...] = (0, 1, 2, 3, 4), n_max: int = 300,
                 learning_rate: float = 0.05, check_every: int = 25, val_frac: float = 0.15):
        super().__init__(cols)
        self.num_leaves, self.min_child_samples = num_leaves, min_child_samples
        self.seeds, self.n_max, self.lr = tuple(seeds), n_max, learning_rate
        self.check_every, self.val_frac = check_every, val_frac
        self.name = f"antinoise_l{num_leaves}_m{min_child_samples}"

    @property
    def config(self) -> dict:
        return {**super().config, "num_leaves": self.num_leaves, "min_child_samples": self.min_child_samples,
                "seeds": list(self.seeds), "n_max": self.n_max, "learning_rate": self.lr,
                "check_every": self.check_every, "val_frac": self.val_frac}

    def _params(self, seed: int, n: int) -> dict:
        return dict(n_estimators=n, num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
                    learning_rate=self.lr, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
                    reg_lambda=3.0, n_jobs=4, random_state=seed, verbosity=-1)

    def best_iteration(self, train: pd.DataFrame) -> int:
        """Mejor nº de árboles por IC en validación interna (usa únicamente ``train``)."""
        import lightgbm as lgb
        tr, va = inner_purged_split(train.index, self.val_frac)
        X, y = train[self.cols].values, train["pct_fwd"].values
        checkpoints = list(range(self.check_every, self.n_max + 1, self.check_every))
        preds = np.zeros((len(checkpoints), len(va)))
        for s in self.seeds:
            m = lgb.LGBMRegressor(**self._params(s, self.n_max)).fit(X[tr], y[tr])
            for i, c in enumerate(checkpoints):
                preds[i] += m.predict(X[va], num_iteration=c) / len(self.seeds)
        ics = [mean_ic(train.index[va], preds[i], train["fwd_ret"].to_numpy()[va]) for i in range(len(checkpoints))]
        return checkpoints[int(np.nanargmax(ics))]

    def fit(self, train):
        import lightgbm as lgb
        self.best_n_ = self.best_iteration(train)
        X, y = train[self.cols].values, train["pct_fwd"].values
        self.models_ = [lgb.LGBMRegressor(**self._params(s, self.best_n_)).fit(X, y) for s in self.seeds]
        return self

    def predict(self, df):
        X = df[self.cols].values
        return np.mean([m.predict(X) for m in self.models_], axis=0)


# ── Combinación de modelos ───────────────────────────────────────────────────

def _trailing(hist: pd.DataFrame, i: int, window: int, lag: int) -> pd.DataFrame:
    """Filas de ``hist`` realizadas al llegar a la posición i: posiciones ≤ i − lag, últimas ``window``."""
    end = i - lag + 1
    return hist.iloc[max(0, end - window):max(0, end)]


def shrunk_ic_weights(mean_ic_by_arm: pd.Series, lam: float = 0.25) -> pd.Series:
    """Pesos = (1−λ)/N + λ·w_IC, con w_IC ∝ max(IC,0); si ningún IC es positivo, 1/N."""
    n = len(mean_ic_by_arm)
    pos = mean_ic_by_arm.clip(lower=0).fillna(0.0)
    w_ic = pos / pos.sum() if pos.sum() > 0 else pd.Series(1.0 / n, index=pos.index)
    return (1 - lam) / n + lam * w_ic


def auc_weights(mean_auc_by_arm: pd.Series) -> pd.Series:
    """Pesos ∝ max(AUC−0.5, 0) (como el selector actual); si todos ≤ 0.5, 1/N."""
    pos = (mean_auc_by_arm - 0.5).clip(lower=0).fillna(0.0)
    return pos / pos.sum() if pos.sum() > 0 else pd.Series(1.0 / len(pos), index=pos.index)


def weights_path(hist: pd.DataFrame, scheme: str, *, lam: float = 0.25, window: int = 520,
                 lag: int = HORIZON + EMBARGO, every: int = HORIZON, min_hist: int = 130) -> pd.DataFrame:
    """Pesos por fecha a partir de una historia realizada (IC o AUC por fecha × brazo).

    En la posición i solo se usan filas con posición ≤ i − ``lag`` (el IC de la fecha j se conoce
    hasta j + horizonte; ``lag`` = horizonte + embargo), últimas ``window`` fechas. Se recalcula cada
    ``every`` fechas; con menos de ``min_hist`` fechas de historia, 1/N.
    scheme: "equal" | "ic" (shrinkage λ) | "auc".
    """
    n = hist.shape[1]
    eq = pd.Series(1.0 / n, index=hist.columns)
    out = np.tile(eq.to_numpy(), (len(hist), 1))
    cur = eq
    for i in range(len(hist)):
        if i % every == 0 and scheme != "equal":
            h = _trailing(hist, i, window, lag)
            if len(h) >= min_hist:
                cur = shrunk_ic_weights(h.mean(), lam) if scheme == "ic" else auc_weights(h.mean())
            else:
                cur = eq
        out[i] = cur.to_numpy() if scheme != "equal" else eq.to_numpy()
    return pd.DataFrame(out, index=hist.index, columns=hist.columns)


def blend_scores(arm_scores: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Score combinado por fila: Σ_arm peso(fecha, arm) · score(fila, arm).

    ``arm_scores``: filas del panel (índice fecha) × brazo, ya normalizados por fecha.
    """
    w = weights.reindex(arm_scores.index)
    return (arm_scores * w.to_numpy()).sum(axis=1)
