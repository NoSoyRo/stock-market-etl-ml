"""
Regresores de volatilidad con información externa: calendario y macro (M7b).

HAR solo ve volatilidad pasada. Aquí se añade lo que se sabe DE ANTEMANO:
resultados trimestrales (fecha estimada), ex-dividendo, decisiones de Banxico y
FOMC, publicación del INPC, más el clima global (VIX, DXY) con rezago de 1 día.

Causalidad:
  - Calendarios publicados (Banxico, FOMC, INEGI): se conocen con anticipación;
    la ventana se cuenta en días hábiles de CALENDARIO (sin mirar las filas
    futuras del ticker).
  - Resultados: fecha esperada = pico de reacción del MISMO trimestre del año
    anterior (+364 días). Cada evento esperado solo es usable desde
    `known_from` = fin de la ventana de detección de ese pico. Nunca se usa el
    pico del trimestre que se predice.
  - Ex-dividendo: `exdiv_declared` usa la fecha REAL del ex-dividendo (supone
    que ya fue declarada; no tenemos la fecha de anuncio ⇒ posible fuga leve).
    `exdiv_expected` usa solo los ex-dividendos del año anterior (+364 días).
  - Macro: `shift(1)` sobre la serie de EE.UU. y luego ffill al calendario BMV.

Los regresores se registran en `vol_regressors.REGRESSORS` al importar el módulo.
Los ex-dividendos necesitan saber de qué ticker es cada DataFrame: `prepare(raws)`
anota `df.attrs["ticker"]` (el script de estudio lo llama con `--extra-module`).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import vol_regressors as vr

ROOT = Path(__file__).parent.parent.parent
CAL_DIR = ROOT / "data" / "calendars"
EVENTS_CSV = CAL_DIR / "events.csv"
MACRO_PARQUET = CAL_DIR / "macro_vix_dxy.parquet"
DIVIDENDS_DIR = ROOT / "data" / "dividends"

CAP = 30                    # tope de días hábiles hasta el próximo evento
TOL_EARN = 2                # ± días hábiles de incertidumbre en la fecha de resultados
TOL_DIV = 5                 # ídem para ex-dividendo esperado
REPORT_WINDOW = (10, 55)    # días naturales tras cierre de trimestre donde buscar el pico
SHIFT_DAYS = 364            # mismo día de la semana un año después

_MONTHS_EN = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"])}
_MONTHS_ES = {m: i + 1 for i, m in enumerate(
    ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
     "septiembre", "octubre", "noviembre", "diciembre"])}


# ─── Parseo de fuentes oficiales (puro; la red vive en build_event_calendar.py) ─

def _plain(html: str) -> str:
    """HTML → líneas de texto."""
    t = re.sub(r"<[^>]+>", "\n", html)
    t = re.sub(r"[ \t\xa0]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t)


def _month_num(token: str) -> int | None:
    """'January' / 'Jan' / 'Jan/Feb' / 'April/May' → mes (si es 'A/B', el SEGUNDO: el día es de B)."""
    last = token.split("/")[-1]
    for name, n in _MONTHS_EN.items():
        if name.lower().startswith(last.lower()[:3]) and len(last) >= 3:
            return n
    return None


def parse_fomc_year_page(html: str, year: int) -> list[pd.Timestamp]:
    """
    Reuniones PROGRAMADAS del FOMC de una página histórica (`fomchistoricalYYYY`):
    "January 27-28 Meeting - 2015" → día de la decisión = último día de la reunión
    (en "Jan/Feb 31-1" es el 1 de febrero). Excluye llamadas de conferencia,
    reuniones no programadas y canceladas.
    """
    pat = re.compile(r"^([A-Za-z/]+) (\d{1,2})(?:-(?:([A-Za-z]+) )?(\d{1,2}))?(?: \((\w+)\))? "
                     r"Meeting - (\d{4})$")
    out = []
    for line in _plain(html).split("\n"):
        m = pat.match(line.strip())
        if not m or int(m.group(6)) != year or m.group(5):
            continue
        mon = _month_num(m.group(3) or m.group(1))
        if mon is None:
            continue
        out.append(pd.Timestamp(year, mon, int(m.group(4) or m.group(2))))
    return sorted(set(out))


def parse_fomc_current_page(html: str) -> list[pd.Timestamp]:
    """
    Página `fomccalendars.htm` (2021+): bajo cada "YYYY FOMC Meetings" una lista
    "January 27-28", "Jan/Feb 31-1", "March 17-18*" ... Se ignoran las fechas de
    publicación de minutas ("Released February 18, 2026").
    """
    text = re.sub(r"\s+", " ", _plain(html))
    heads = [(m.end(), int(m.group(1))) for m in re.finditer(r"(\d{4}) FOMC Meetings", text)]
    tok = re.compile(r"(?<!Released )\b([A-Z][a-z]{2,8}(?:/[A-Z][a-z]{2,8})?) (\d{1,2})(?:-(\d{1,2}))?\*?(?![\d,])(?! \()")
    out = []
    for i, (pos, year) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        seg = text[pos: (heads[i + 1][0] - 25) if i + 1 < len(heads) else end].split("* Meeting associated")[0]
        for m in tok.finditer(seg):
            mon = _month_num(m.group(1))
            if mon is not None and re.fullmatch(r"[A-Z][a-z]+(/[A-Z][a-z]+)?", m.group(1)):
                out.append(pd.Timestamp(year, mon, int(m.group(3) or m.group(2))))
    return sorted(set(out))


def parse_banxico_announcements(html: str) -> list[pd.Timestamp]:
    """Lista de anuncios (dd/mm/aa) de la página oficial de decisiones de política monetaria."""
    out = []
    for d, m, y in re.findall(r"\b(\d\d)/(\d\d)/(\d\d)\b", html):
        try:
            out.append(pd.Timestamp(2000 + int(y), int(m), int(d)))
        except ValueError:
            continue
    return sorted(set(out))


def parse_inegi_inpc_monthly(text: str, year: int) -> list[pd.Timestamp]:
    """
    Fechas de publicación del INPC "2.ª quincena y mensual" (el dato de inflación
    mensual) del PDF de calendario de INEGI, formato "dd/mes  Periodo".
    Devuelve [] si no hay una racha de exactamente 12 fechas junto a esa fila.
    """
    lines = text.split("\n")
    row = re.compile(r"(\d{1,2})/([A-Za-zéÉ]+)\s{2,}([A-Za-zé]+)(?:/\d{4})?\s*$")
    key = [i for i, l in enumerate(lines) if re.search(r"quincena y\s*(mensual)?", l) or
           ("quincena y" in l.lower() and "mensual" in l.lower())]
    for k in key:
        # racha de 12 filas de fecha que contenga la línea k (o esté a ≤ 8 líneas)
        idx = [i for i in range(max(0, k - 14), min(len(lines), k + 15)) if row.search(lines[i])]
        runs, cur = [], []
        for i in idx:
            if cur and i - cur[-1] > 2:
                runs.append(cur)
                cur = []
            cur.append(i)
        runs.append(cur)
        for r in runs:
            if len(r) >= 12 and r[0] - 4 <= k <= r[-1] + 4:
                days = []
                for i in r[-12:] if len(r) > 12 else r:
                    d, mon = row.search(lines[i]).group(1, 2)
                    if mon.lower() not in _MONTHS_ES:
                        days = []
                        break
                    days.append(pd.Timestamp(year, _MONTHS_ES[mon.lower()], int(d)))
                if len(days) == 12 and len(set(d.month for d in days)) == 12:
                    return days
    return []


# ─── Carga del calendario ────────────────────────────────────────────────────

def load_calendar(path: Path | str = EVENTS_CSV) -> pd.DataFrame:
    """CSV de eventos: columnas date, event, source (url), note. Ordenado por fecha."""
    cal = pd.read_csv(path, parse_dates=["date"])
    missing = {"date", "event", "source"} - set(cal.columns)
    if missing:
        raise ValueError(f"faltan columnas en el calendario: {missing}")
    return cal.sort_values("date").reset_index(drop=True)


def event_dates(cal: pd.DataFrame, event: str) -> pd.DatetimeIndex:
    """Fechas únicas ordenadas de un tipo de evento ('banxico', 'fomc', 'inpc')."""
    return pd.DatetimeIndex(sorted(cal.loc[cal["event"] == event, "date"].unique()))


# ─── Ventanas de evento (días hábiles de calendario) ─────────────────────────

def _d64(x) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(x).values.astype("datetime64[D]"))


def bdays_offset(index: pd.DatetimeIndex, event: pd.Timestamp) -> np.ndarray:
    """Días hábiles de t al evento (positivo = futuro). Sin feriados: se conocen de antemano."""
    return np.busday_count(_d64(index), np.datetime64(pd.Timestamp(event).date()))


def event_features(index: pd.DatetimeIndex, events: pd.DatetimeIndex, horizons=(5, 21),
                   *, tol: int = 0, known_from: pd.DatetimeIndex | None = None,
                   cap: int = CAP, prefix: str = "ev") -> pd.DataFrame:
    """
    Banderas y distancia a eventos conocidos.

      {prefix}_in{h}   1 si algún evento cae en (t−tol, t+h+tol] días hábiles.
      {prefix}_days    días hábiles al próximo evento (−tol, cap], recortado.

    `known_from[i]`: desde cuándo se sabe el evento i (default: siempre). Antes
    de esa fecha el evento no existe para t. Cobertura: NaN antes de que
    empiece el calendario (no sabemos si había evento) y después de su fin.
    """
    n = len(index)
    cols = {f"{prefix}_in{h}": np.zeros(n) for h in horizons}
    nxt = np.full(n, float(cap))
    if len(events) == 0:
        return pd.DataFrame(np.nan, index=index, columns=[*cols, f"{prefix}_days"])
    kf = _d64(known_from) if known_from is not None else None
    idx64 = _d64(index)
    for i, ev in enumerate(events):
        off = bdays_offset(index, ev)
        usable = np.ones(n, bool) if kf is None else idx64 >= kf[i]
        for h in horizons:
            cols[f"{prefix}_in{h}"] = np.where(usable & (off > -tol) & (off <= h + tol),
                                               1.0, cols[f"{prefix}_in{h}"])
        sel = usable & (off > -tol) & (off < nxt)
        nxt = np.where(sel, off, nxt)
    out = pd.DataFrame({**cols, f"{prefix}_days": np.clip(nxt, 0, cap)}, index=index)
    lo, hi = events.min() - pd.Timedelta(days=cap * 2), events.max()
    out.loc[(index < lo) | (index > hi)] = np.nan
    return out


# ─── Resultados trimestrales: fecha esperada sin mirar el futuro ─────────────

def reaction_score(df: pd.DataFrame) -> pd.Series:
    """Intensidad de reacción del día: |gap|/σ_{t-1} (σ de 22d) + log(volumen / mediana 60d previa)."""
    r = np.log(df["Close"]).diff()
    sigma = r.rolling(22).std().shift(1)
    gap = np.log(df["Open"] / df["Close"].shift(1)).abs() / sigma.where(sigma > 0)
    vmed = df["Volume"].where(df["Volume"] > 0).rolling(60, min_periods=20).median().shift(1)
    lv = np.log((df["Volume"].where(df["Volume"] > 0) / vmed))
    return (gap.clip(upper=10).fillna(0) + lv.clip(-3, 3).fillna(0)).where(gap.notna() | lv.notna())


def detect_report_peaks(df: pd.DataFrame, window: tuple[int, int] = REPORT_WINDOW,
                        min_rows: int = 15) -> pd.DataFrame:
    """
    Pico de reacción tras cada cierre de trimestre. Devuelve filas
    (quarter, peak, known_from) donde `known_from` = último día de la ventana:
    antes de esa fecha el pico no está completo.
    """
    score = reaction_score(df)
    rows = []
    for y in range(df.index.min().year, df.index.max().year + 1):
        for q, (m, d) in enumerate([(3, 31), (6, 30), (9, 30), (12, 31)], start=1):
            qe = pd.Timestamp(y, m, d)
            lo, hi = qe + pd.Timedelta(days=window[0]), qe + pd.Timedelta(days=window[1])
            if hi > df.index.max():
                continue
            s = score[(score.index >= lo) & (score.index <= hi)].dropna()
            if len(s) < min_rows:
                continue
            rows.append({"quarter": q, "year": y, "peak": s.idxmax(), "known_from": hi})
    return pd.DataFrame(rows, columns=["quarter", "year", "peak", "known_from"])


def _snap_weekday(ts: pd.Timestamp) -> pd.Timestamp:
    while ts.weekday() >= 5:
        ts += pd.Timedelta(days=1)
    return ts


def expected_report_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fechas esperadas = pico del mismo trimestre del año anterior + 364 días.
    Columnas: expected, known_from. Un evento esperado nunca depende del trimestre
    que predice (su `known_from` es ≈ un año antes de la fecha esperada).
    """
    pk = detect_report_peaks(df)
    if pk.empty:
        return pd.DataFrame(columns=["expected", "known_from"])
    exp = pk["peak"].map(lambda p: _snap_weekday(p + pd.Timedelta(days=SHIFT_DAYS)))
    return pd.DataFrame({"expected": exp.values, "known_from": pk["known_from"].values})


def earnings_features(df: pd.DataFrame) -> pd.DataFrame:
    """earn_in5, earn_in21, earn_days (tol ±2 días hábiles). NaN hasta tener 1 año de historia."""
    ex = expected_report_dates(df)
    if ex.empty:
        return pd.DataFrame(np.nan, index=df.index, columns=["earn_in5", "earn_in21", "earn_days"])
    out = event_features(df.index, pd.DatetimeIndex(ex["expected"]), (5, 21), tol=TOL_EARN,
                         known_from=pd.DatetimeIndex(ex["known_from"]), prefix="earn")
    # sin al menos una fecha esperada usable en t no hay información: NaN
    first_ok = pd.DatetimeIndex(ex["known_from"]).min()
    out.loc[out.index < first_ok] = np.nan
    return out


# ─── Dividendos ──────────────────────────────────────────────────────────────

def load_dividends(symbol: str, directory: Path = DIVIDENDS_DIR) -> pd.Series | None:
    """Dividendos en caché (sin red). None si no hay archivo."""
    path = Path(directory) / f"{symbol.replace('/', '_').replace('^', '')}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)["dividend"]


def exdiv_declared_features(index: pd.DatetimeIndex, dividends: pd.Series | None) -> pd.DataFrame:
    """Ex-dividendo REAL próximo (supone fecha ya declarada). NaN antes del primer dividendo registrado."""
    cols = ["exdiv_in5", "exdiv_in21", "exdiv_days"]
    if dividends is None:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    ev = pd.DatetimeIndex(dividends.index)
    if len(ev) == 0:
        return pd.DataFrame({"exdiv_in5": 0.0, "exdiv_in21": 0.0, "exdiv_days": float(CAP)}, index=index)
    out = event_features(index, ev, (5, 21), prefix="exdiv")
    out.loc[index < ev.min()] = np.nan
    after = index > ev.max()                      # sin dividendos posteriores registrados
    out.loc[after, ["exdiv_in5", "exdiv_in21"]] = 0.0
    out.loc[after, "exdiv_days"] = float(CAP)
    return out[cols]


def exdiv_expected_features(index: pd.DatetimeIndex, dividends: pd.Series | None) -> pd.DataFrame:
    """Ex-dividendo ESPERADO = ex-fechas del año anterior + 364 días (conocidas desde la fecha original)."""
    cols = ["exdivx_in5", "exdivx_in21", "exdivx_days"]
    if dividends is None:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    ev = pd.DatetimeIndex(dividends.index)
    if len(ev) == 0:
        return pd.DataFrame({"exdivx_in5": 0.0, "exdivx_in21": 0.0, "exdivx_days": float(CAP)}, index=index)
    exp = pd.DatetimeIndex([_snap_weekday(d + pd.Timedelta(days=SHIFT_DAYS)) for d in ev])
    out = event_features(index, exp, (5, 21), tol=TOL_DIV, known_from=ev, prefix="exdivx")
    out.loc[index < ev.min() + pd.Timedelta(days=SHIFT_DAYS - 30)] = np.nan
    after = index > exp.max()
    out.loc[after, ["exdivx_in5", "exdivx_in21"]] = 0.0
    out.loc[after, "exdivx_days"] = float(CAP)
    return out[cols]


# ─── Macro con rezago ────────────────────────────────────────────────────────

def lag_to_calendar(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """
    Serie de EE.UU. con rezago de 1 sesión PROPIA y ffill al calendario BMV: el
    valor en t es el cierre de la sesión de EE.UU. anterior a t (≤ t−1).
    """
    s = series.dropna().sort_index()
    return s.shift(1).reindex(s.index.union(index)).ffill().reindex(index)


def vix_features(vix: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """log(VIX) y cambio log a 5 días, ambos con rezago."""
    lv = np.log(vix.where(vix > 0))
    full = pd.DataFrame({"vix_log": lv, "vix_d5": lv.diff(5)})
    return pd.DataFrame({c: lag_to_calendar(full[c], index) for c in full})


def dxy_features(dxy: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cambio log del dólar a 5 y 22 días, con rezago."""
    ld = np.log(dxy.where(dxy > 0))
    full = pd.DataFrame({"dxy_d5": ld.diff(5), "dxy_d22": ld.diff(22)})
    return pd.DataFrame({c: lag_to_calendar(full[c], index) for c in full})


def load_macro(path: Path = MACRO_PARQUET) -> pd.DataFrame:
    """Caché parquet con columnas VIX y DXY (la construye build_event_calendar.py --macro)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} no existe; corre scripts/build_event_calendar.py --macro")
    return pd.read_parquet(path)


# ─── Registro en el arnés de M7a ─────────────────────────────────────────────

_CAL_CACHE: dict[str, pd.DatetimeIndex] = {}


def _cal_events(name: str) -> pd.DatetimeIndex:
    if name not in _CAL_CACHE:
        _CAL_CACHE[name] = event_dates(load_calendar(), name)
    return _CAL_CACHE[name]


def _cal_regressor(name: str):
    def fn(df: pd.DataFrame) -> pd.DataFrame:
        return event_features(df.index, _cal_events(name), (5, 21), prefix=name)
    return fn


def _ticker_dividends(df: pd.DataFrame) -> pd.Series | None:
    sym = df.attrs.get("symbol")
    return load_dividends(sym) if sym else None


def _macro_fn(kind: str):
    def fn(raws: dict[str, pd.DataFrame]) -> pd.DataFrame:
        idx = pd.DatetimeIndex(sorted(set().union(*[df.index for df in raws.values()])))
        macro = load_macro()
        return vix_features(macro["VIX"], idx) if kind == "vix" else dxy_features(macro["DXY"], idx)
    return fn


def prepare(raws: dict[str, pd.DataFrame]) -> None:
    """Anota ticker y símbolo de Yahoo en `df.attrs` (necesario para ex-dividendos)."""
    from src.data.store import BMV_TICKERS
    for name, df in raws.items():
        df.attrs["ticker"] = name
        df.attrs["symbol"] = BMV_TICKERS.get(name)


def _register_all() -> None:
    specs = [
        vr.Regressor("earnings_cal", ("earn_in5", "earn_in21", "earn_days"), "ticker", earnings_features),
        vr.Regressor("exdiv_declared", ("exdiv_in5", "exdiv_in21", "exdiv_days"), "ticker",
                     lambda df: exdiv_declared_features(df.index, _ticker_dividends(df))),
        vr.Regressor("exdiv_expected", ("exdivx_in5", "exdivx_in21", "exdivx_days"), "ticker",
                     lambda df: exdiv_expected_features(df.index, _ticker_dividends(df))),
        vr.Regressor("banxico", ("banxico_in5", "banxico_in21", "banxico_days"), "ticker",
                     _cal_regressor("banxico")),
        vr.Regressor("fomc", ("fomc_in5", "fomc_in21", "fomc_days"), "ticker", _cal_regressor("fomc")),
        vr.Regressor("inpc", ("inpc_in5", "inpc_in21", "inpc_days"), "ticker", _cal_regressor("inpc")),
        vr.Regressor("vix", ("vix_log", "vix_d5"), "universe", _macro_fn("vix")),
        vr.Regressor("dxy", ("dxy_d5", "dxy_d22"), "universe", _macro_fn("dxy")),
    ]
    for s in specs:
        if s.name not in vr.REGRESSORS:
            vr.register(s)


_register_all()


# ─── Skill dentro / fuera de ventanas de evento ──────────────────────────────

def skill_in_out(panel: pd.DataFrame, target: str, h: int, reg_cols: list[str],
                 mask_col: str, *, n_splits: int = 6) -> dict:
    """
    Skill (1 − MSE/MSE ingenuo, agrupando los folds de prueba) de HAR y HAR+reg_cols
    en filas con `mask_col == 1` (dentro de ventana de evento) vs `== 0` (fuera).
    Mismos folds purgados que `evaluate_regressor`.
    """
    from sklearn.linear_model import LinearRegression

    from src.ml.purged_cv import purged_walk_forward_date_splits
    from src.ml.targets import HAR_COLS
    from src.ml.vol_models import HARModel, naive_forecast

    data = panel.dropna(subset=[target, mask_col] + list(reg_cols)).reset_index(drop=True)
    y = data[target].to_numpy()
    cols_x = HAR_COLS + list(reg_cols)
    splits = purged_walk_forward_date_splits(
        data["date"], n_splits=n_splits, horizon=h, embargo=max(vr.MIN_EMBARGO, h))
    eb, ex, en, mk = [], [], [], []
    for tr, te in splits:
        Xtr, Xte = data.iloc[tr], data.iloc[te]
        pb = HARModel().fit(Xtr, y[tr]).predict(Xte)
        px = LinearRegression().fit(Xtr[cols_x].to_numpy(), y[tr]).predict(Xte[cols_x].to_numpy())
        pn = naive_forecast(Xte, h)
        eb.append((y[te] - pb) ** 2)
        ex.append((y[te] - px) ** 2)
        en.append((y[te] - pn) ** 2)
        mk.append(Xte[mask_col].to_numpy() > 0.5)
    eb, ex, en, mk = (np.concatenate(a) for a in (eb, ex, en, mk))
    out = {"horizon": h, "mask": mask_col, "regressors": list(reg_cols)}
    for label, sel in (("in", mk), ("out", ~mk)):
        if sel.sum() == 0:
            out[label] = None
            continue
        out[label] = {"n": int(sel.sum()), "skill_har": float(1 - eb[sel].mean() / en[sel].mean()),
                      "skill_reg": float(1 - ex[sel].mean() / en[sel].mean()),
                      "delta": float((eb[sel].mean() - ex[sel].mean()) / en[sel].mean())}
    return out


def post_study(panel: pd.DataFrame, trials: list[tuple[str, list[str]]], n_splits: int) -> dict:
    """Gancho de `run_vol_regressor_study.py --extra-module`: skill dentro/fuera de ventanas de evento."""
    cols = dict(trials)
    pairs = [("earnings_cal", "earn_in5", 5), ("earnings_cal", "earn_in21", 21),
             ("banxico", "banxico_in5", 5), ("banxico", "banxico_in21", 21),
             ("fomc", "fomc_in5", 5), ("fomc", "fomc_in21", 21),
             ("exdiv_declared", "exdiv_in5", 5), ("exdiv_declared", "exdiv_in21", 21)]
    res = []
    for name, mask, h in pairs:
        if name not in cols or mask not in panel.columns:
            continue
        r = skill_in_out(panel, f"log_rv_fwd_{h}", h, cols[name], mask, n_splits=n_splits)
        r["name"] = name
        res.append(r)
    return {"in_out_windows": res}
