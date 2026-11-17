"""Producto de datos Precios & FX: completitud y calidad (Equipo Mercado).

Lee lo que ya está en data/ y genera gráficas + tablas en
productos/precios_fx/salida/. No descarga nada.

    python productos/precios_fx/calidad_datos.py

Fuentes:
  data/market/*.parquet    OHLCV diario (yfinance): BMV, USA, ETFs, cripto, FIBRAs, macro
  data/market/MXN=X        tipo de cambio USD/MXN
  data/intraday/*.parquet  precio cada 5 minutos (DataBursátil), 36 emisoras BMV
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.universes import UNIVERSES  # noqa: E402

MARKET = ROOT / "data" / "market"
INTRADAY = ROOT / "data" / "intraday"
OUT = Path(__file__).resolve().parent / "salida"

GRUPOS = ["bmv", "fibras", "etfs", "usa", "cripto", "macro"]
COLOR = {
    "bmv": "#2a9d8f", "fibras": "#8ab17d", "etfs": "#457b9d",
    "usa": "#1d3557", "cripto": "#e9c46a", "macro": "#e76f51",
}
NOMBRE = {
    "bmv": "Acciones BMV", "fibras": "FIBRAs", "etfs": "ETFs",
    "usa": "Acciones USA", "cripto": "Cripto (proxies)", "macro": "Macro / FX",
}
BARRAS_ESPERADAS = 79  # 6.5 h de sesión / 5 min + 1
SALTO = 0.25  # cambio diario de cierre > 25 % se marca como sospechoso

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11,
})


# ── carga ────────────────────────────────────────────────────────────────────

def grupo_de_simbolo() -> dict[str, str]:
    """Símbolo yfinance → universo. FIBRAs gana sobre BMV/ETFs si se repite."""
    out: dict[str, str] = {}
    for cfg in UNIVERSES.values():
        for sym in cfg["tickers"].values():
            out.setdefault(sym, cfg["slug"])
    for sym in UNIVERSES["🏢 FIBRAs (bienes raíces MX)"]["tickers"].values():
        out[sym] = "fibras"
    return out


def esperados_sin_archivo() -> list[tuple[str, str]]:
    faltan = []
    for cfg in UNIVERSES.values():
        for sym in cfg["tickers"].values():
            if not (MARKET / f"{sym.replace('^', '')}.parquet").exists():
                faltan.append((sym, cfg["slug"]))
    return sorted(set(faltan))


def cargar_diario() -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(MARKET.glob("*.parquet")):
        df = pd.read_parquet(path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        frames[path.stem] = df.sort_index()
    return frames


def cargar_intradia() -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(INTRADAY.glob("*.parquet")):
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["dia"] = df["datetime"].dt.normalize()
        frames[path.stem] = df.sort_values("datetime").reset_index(drop=True)
    return frames


# ── calendarios de referencia ────────────────────────────────────────────────

def calendario_bmv(diario: dict[str, pd.DataFrame], grupos: dict[str, str]) -> pd.DatetimeIndex:
    """Días en que operó al menos la mitad de las emisoras .MX vivas ese día."""
    mx = {s: df for s, df in diario.items() if s.endswith(".MX")}
    presentes = pd.concat({s: pd.Series(1, index=df.index) for s, df in mx.items()}, axis=1, sort=True)
    vivas = pd.DataFrame(
        {s: (presentes.index >= df.index.min()) & (presentes.index <= df.index.max())
         for s, df in mx.items()},
        index=presentes.index,
    )
    ratio = presentes.notna().sum(axis=1) / vivas.sum(axis=1).clip(lower=1)
    return presentes.index[ratio >= 0.5]


def referencia(simbolo: str, cal_bmv: pd.DatetimeIndex, cal_usa: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if simbolo == "MXN=X":
        return pd.bdate_range(cal_usa.min(), cal_usa.max())  # FX opera lunes a viernes
    if simbolo.endswith(".MX"):
        return cal_bmv
    return cal_usa


# ── diagnóstico diario ───────────────────────────────────────────────────────

def diagnostico_diario(diario, grupos, cal_bmv, cal_usa) -> pd.DataFrame:
    filas = []
    for sym, df in diario.items():
        g = grupos.get(sym, "macro")
        if df.empty:
            filas.append({"simbolo": sym, "grupo": g, "filas": 0})
            continue
        ref = referencia(sym, cal_bmv, cal_usa)
        ref = ref[(ref >= df.index.min()) & (ref <= df.index.max())]
        ohlc = df[["Open", "High", "Low", "Close"]]
        incoherente = (df["High"] < ohlc.max(axis=1) - 1e-9) | (df["Low"] > ohlc.min(axis=1) + 1e-9)
        ret = df["Close"].pct_change().abs()
        filas.append({
            "simbolo": sym,
            "grupo": g,
            "filas": len(df),
            "primer_dato": df.index.min().date(),
            "ultimo_dato": df.index.max().date(),
            "dias_esperados": len(ref),
            "dias_faltantes": int(len(ref.difference(df.index))),
            "completitud_pct": round(100 * len(ref.intersection(df.index)) / max(len(ref), 1), 2),
            "nulos": int(df.isna().any(axis=1).sum()),
            "duplicados": int(df.index.duplicated().sum()),
            "precio_<=0": int((ohlc <= 0).any(axis=1).sum()),
            "ohlc_incoherente": int(incoherente.sum()),
            "volumen_0": int((df["Volume"] == 0).sum()),
            "saltos_>25%": int((ret > SALTO).sum()),
            "cierre_repetido_pct": round(100 * float((df["Close"].diff() == 0).mean()), 2),
        })
    tabla = pd.DataFrame(filas)
    tabla["grupo"] = pd.Categorical(tabla["grupo"], GRUPOS, ordered=True)
    return tabla.sort_values(["grupo", "simbolo"]).reset_index(drop=True)


def completitud_anual(diario, cal_bmv, cal_usa) -> pd.DataFrame:
    """% de días presentes por año, solo dentro de la vida del instrumento."""
    out = {}
    for sym, df in diario.items():
        if df.empty:
            continue
        ref = referencia(sym, cal_bmv, cal_usa)
        ref = ref[(ref >= df.index.min()) & (ref <= df.index.max())]
        esperados = pd.Series(1, index=ref).groupby(ref.year).sum()
        presentes = pd.Series(1, index=ref.intersection(df.index))
        presentes = presentes.groupby(presentes.index.year).sum()
        out[sym] = (100 * presentes.reindex(esperados.index).fillna(0) / esperados)
    return pd.DataFrame(out).T.sort_index(axis=1)


# ── diagnóstico intradía ─────────────────────────────────────────────────────

def diagnostico_intradia(intradia, diario, cal_bmv) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.data.bmv_catalog import BMV_YFINANCE

    filas, barras = [], {}
    for t, df in intradia.items():
        por_dia = df.groupby("dia")
        n = por_dia.size()
        barras[t] = n
        unico = por_dia["price"].nunique()
        festivo = ~n.index.isin(cal_bmv)
        fila = {
            "ticker": t,
            "barras": len(df),
            "columnas": ",".join(c for c in df.columns if c not in ("dia",)),
            "primer_dato": df["datetime"].min(),
            "ultimo_dato": df["datetime"].max(),
            "dias": int(n.size),
            "dias_completos_79": int((n == BARRAS_ESPERADAS).sum()),
            "dias_incompletos": int((n < BARRAS_ESPERADAS).sum()),
            "dias_en_festivo_o_sin_diario": int(festivo.sum()),
            "dias_precio_plano": int((unico == 1).sum()),
            "barras_sin_cambio_pct": round(100 * float((df["price"].diff() == 0).mean()), 1),
            "hora_min": df["datetime"].dt.strftime("%H:%M").min(),
            "hora_max": df["datetime"].dt.strftime("%H:%M").max(),
            "nulos": int(df["price"].isna().sum()),
            "duplicados": int(df["datetime"].duplicated().sum()),
        }
        sym = BMV_YFINANCE.get(t)
        d = diario.get(sym) if sym else None
        if d is not None and not d.empty:
            agg = por_dia["price"].agg(["first", "last", "max", "min"])
            j = agg.join(d[["Open", "High", "Low", "Close"]], how="inner")
            err = lambda a, b: (j[a] - j[b]).abs() / j[b]
            fila.update({
                "dias_cruzados_con_diario": len(j),
                "ultimo_5m_igual_cierre_pct": round(100 * float((err("last", "Close") < 0.001).mean()), 1),
                "err_mediano_cierre_pct": round(100 * float(err("last", "Close").median()), 3),
                "err_mediano_max_vs_high_pct": round(100 * float(err("max", "High").median()), 3),
                "err_mediano_min_vs_low_pct": round(100 * float(err("min", "Low").median()), 3),
                "err_mediano_primero_vs_open_pct": round(100 * float(err("first", "Open").median()), 3),
            })
        filas.append(fila)
    return pd.DataFrame(filas), pd.DataFrame(barras).sort_index()


# ── gráficas ─────────────────────────────────────────────────────────────────

def _guardar(fig, nombre: str) -> None:
    fig.savefig(OUT / nombre, bbox_inches="tight")
    plt.close(fig)
    print("  ✓", nombre)


def g_ejemplo_ohlcv(diario, sym="WALMEX.MX", dias=60) -> None:
    df = diario[sym].tail(dias)
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    x = np.arange(len(df))
    sube = df["Close"] >= df["Open"]
    col = np.where(sube, "#2a9d8f", "#e63946")
    ax.vlines(x, df["Low"], df["High"], color=col, lw=1)
    ax.bar(x, (df["Close"] - df["Open"]).abs(), bottom=df[["Open", "Close"]].min(axis=1),
           color=col, width=0.6)
    axv.bar(x, df["Volume"] / 1e6, color=col, width=0.6)
    r = df.iloc[-1]
    ax.text(0.99, 0.97,
            f"Último día ({df.index[-1]:%d-%b-%Y})\n"
            f"Open  {r['Open']:.2f}\nHigh  {r['High']:.2f}\nLow   {r['Low']:.2f}\n"
            f"Close {r['Close']:.2f}\nVolume {r['Volume'] / 1e6:.1f} M",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    ax.set_xlim(-1, len(df))
    ax.set_title(f"Así se lee el OHLCV diario — {sym}, últimas {dias} sesiones")
    ax.set_ylabel("Precio (MXN)")
    axv.set_ylabel("Volumen\n(millones)")
    ticks = x[:: max(1, len(x) // 8)]
    axv.set_xticks(ticks, [df.index[k].strftime("%d-%b-%y") for k in ticks])
    ax.text(0.01, 0.03, "Cada vela es un día: la mecha va del Low al High;\n"
                        "el cuerpo va del Open al Close (verde si cerró arriba, rojo si cerró abajo).",
            transform=ax.transAxes, va="bottom", fontsize=8, color="#333")
    _guardar(fig, "01_ejemplo_ohlcv.png")


def g_cobertura(diario, tabla, faltan, hoy) -> None:
    t = tabla[tabla["filas"] > 0].copy()
    t = t.sort_values(["grupo", "primer_dato"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(11, 15))
    y = 0
    yticks, ylabels = [], []
    for g in GRUPOS:
        sub = t[t["grupo"] == g]
        for _, r in sub.iterrows():
            ini, fin = pd.Timestamp(r["primer_dato"]), pd.Timestamp(r["ultimo_dato"])
            ax.barh(y, (fin - ini).days, left=ini, color=COLOR[g], height=0.7)
            yticks.append(y)
            ylabels.append(r["simbolo"])
            y += 1
        for sym, slug in faltan:
            if slug == g:
                ax.text(pd.Timestamp("2000-03-01"), y, f"{sym}: sin archivo", va="center",
                        fontsize=7.5, color="#e63946")
                yticks.append(y)
                ylabels.append(sym)
                y += 1
        y += 1
    ax.set_yticks(yticks, ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(hoy, color="black", ls="--", lw=1)
    ultimo = pd.Timestamp(t["ultimo_dato"].max())
    ax.axvspan(ultimo, hoy, color="#e63946", alpha=0.12)
    ax.text(hoy, -1.2, f"hoy {hoy:%d-%b-%Y}  ", fontsize=8, ha="right")
    ax.text(ultimo, -2.6, f"último dato {ultimo:%d-%b-%Y} → {(hoy - ultimo).days} días sin actualizar  ",
            fontsize=8, ha="right", color="#e63946")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[g]) for g in GRUPOS]
    ax.legend(handles, [NOMBRE[g] for g in GRUPOS], loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=6, fontsize=8, frameon=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Cobertura del OHLCV diario: del dato más antiguo al más reciente\n"
                 "(las barras que arrancan en ene-2000 empiezan ahí porque la descarga empieza en 2000)",
                 pad=22)
    _guardar(fig, "02_cobertura_diaria.png")


def g_completitud_anual(anual, tabla) -> None:
    orden = [s for s in tabla["simbolo"] if s in anual.index]
    a = anual.loc[orden]
    fig, ax = plt.subplots(figsize=(13, 15))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(a.values, aspect="auto", cmap=cmap, vmin=80, vmax=100)
    ax.set_yticks(range(len(a)), a.index, fontsize=7)
    ax.set_xticks(range(a.shape[1]), a.columns, rotation=90, fontsize=8)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            v = a.values[i, j]
            if not np.isnan(v) and v < 99.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, fraction=0.025, label="% de días hábiles con dato")
    ax.set_title("Completitud por año (gris = el instrumento todavía no existía; número = año con huecos)")
    _guardar(fig, "03_completitud_anual.png")


def g_problemas(tabla) -> None:
    cols = ["dias_faltantes", "volumen_0", "ohlc_incoherente", "precio_<=0", "saltos_>25%"]
    t = tabla[tabla["filas"] > 0].set_index("simbolo")
    t = t[(t[cols].sum(axis=1) > 0) | (t["cierre_repetido_pct"] > 10)]
    t = t.loc[(t["volumen_0"] + t["dias_faltantes"] + t["ohlc_incoherente"]).sort_values(ascending=False).index]
    fig, axes = plt.subplots(1, len(cols) + 1, figsize=(15, 0.22 * len(t) + 2), sharey=True,
                             layout="constrained")
    for ax, c in zip(axes, cols + ["cierre_repetido_pct"]):
        colores = [COLOR[g] for g in t["grupo"]]
        ax.barh(range(len(t)), t[c], color=colores)
        ax.set_title(c.replace("_", " "), fontsize=9)
        for i, v in enumerate(t[c]):
            if v:
                ax.text(v, i, f" {v:g}", va="center", fontsize=6)
    axes[0].set_yticks(range(len(t)), t.index, fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle("Problemas por instrumento (diario). Solo aparecen los que tienen alguno.",
                 fontweight="bold")
    _guardar(fig, "04_problemas_diario.png")


def g_tipo_cambio(diario, tabla) -> None:
    fx = diario["MXN=X"]
    fila = tabla.set_index("simbolo").loc["MXN=X"]
    ref = pd.bdate_range(fx.index.min(), fx.index.max())
    falt = ref.difference(fx.index)
    incoh = fx[(fx["High"] < fx[["Open", "Close"]].max(axis=1) - 1e-9)
               | (fx["Low"] > fx[["Open", "Close"]].min(axis=1) + 1e-9)]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(fx.index, fx["Close"], color=COLOR["macro"], lw=0.8)
    ax.scatter(incoh.index, incoh["Close"], s=12, color="black", zorder=3,
               label=f"High/Low incoherente ({len(incoh)})")
    ax.set_ylabel("Pesos por dólar")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"Tipo de cambio USD/MXN (MXN=X): {fx.index.min():%d-%b-%Y} a {fx.index.max():%d-%b-%Y}")
    huecos = pd.Series(1, index=falt).resample("YE").sum().reindex(
        pd.date_range(ref.min(), ref.max(), freq="YE"), fill_value=0)
    ax2.bar(huecos.index, huecos.values, width=200, color="#e63946")
    ax2.set_ylabel("Días hábiles\nsin dato")
    ax.text(0.99, 0.03, f"Volumen = 0 en {int(fila['volumen_0'])} de {int(fila['filas'])} filas "
                        "(normal: el FX no se negocia en una bolsa central)", transform=ax.transAxes,
            ha="right", fontsize=8, color="#555")
    _guardar(fig, "05_tipo_cambio.png")


def g_intradia_calendario(barras, cal_bmv) -> None:
    b = barras.T.sort_index()
    dias = b.columns
    fig, ax = plt.subplots(figsize=(15, 9))
    cmap = ListedColormap(["#e63946", "#f4a261", "#e9c46a", "#2a9d8f"])
    bounds = np.array([0, 40, 70, 78.5, 80])
    idx = np.digitize(b.fillna(0).values, bounds) - 1
    idx = np.where(b.isna().values, np.nan, idx)
    ax.imshow(idx, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    festivos = [i for i, d in enumerate(dias) if d not in cal_bmv]
    for i in festivos:
        ax.axvline(i, color="black", lw=0.6, alpha=0.6)
    ax.set_yticks(range(len(b)), b.index, fontsize=7)
    meses = [i for i, d in enumerate(dias) if i == 0 or d.month != dias[i - 1].month]
    ax.set_xticks(meses, [dias[i].strftime("%b-%y") for i in meses], rotation=90, fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.colors]
    handles.append(plt.Line2D([0], [0], color="black", lw=1))
    ax.legend(handles, ["< 40 barras", "40–70", "71–78", "79 (día completo)",
                        "día festivo / sin dato diario"],
              ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=8)
    ax.set_title("Intradía 5 min: barras por día y emisora (se esperan 79)")
    _guardar(fig, "06_intradia_barras_por_dia.png")


def g_intradia_un_dia(intradia, diario, ticker="WALMEX", sym="WALMEX.MX") -> None:
    df = intradia[ticker]
    d = diario[sym]
    n = df.groupby("dia").size()
    distintos = df.groupby("dia")["price"].nunique()
    # el último día completo con precios reales: un día con un solo precio es relleno de la fuente
    dia = [x for x in n[n == BARRAS_ESPERADAS].index if x in d.index and distintos[x] > 1][-1]
    x = df[df["dia"] == dia]
    r = d.loc[dia]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(x["datetime"], x["price"], where="post", color=COLOR["bmv"], lw=1.2)
    ax.scatter(x["datetime"], x["price"], s=8, color=COLOR["bmv"], zorder=3)
    for etq, val, c in (("High", r["High"], "#2a9d8f"), ("Low", r["Low"], "#e63946"),
                        ("Open", r["Open"], "gray"), ("Close", r["Close"], "black")):
        ax.axhline(val, ls="--", lw=0.8, color=c)
        ax.text(x["datetime"].iloc[-1], val, f"  {etq} diario {val:.2f}", va="center", fontsize=8, color=c)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(f"Un día intradía: {ticker} {dia:%d-%b-%Y} — {len(x)} barras de 5 min")
    ax.set_ylabel("Precio (MXN)")
    ax.text(0.01, 0.03, "Cada punto es el último precio negociado en esa marca de 5 min.\n"
                        "El archivo no trae Open/High/Low ni volumen por barra: solo datetime y price.",
            transform=ax.transAxes, fontsize=8, color="#333")
    _guardar(fig, "07_intradia_un_dia.png")


def g_intradia_vs_diario(tint) -> None:
    t = tint.dropna(subset=["ultimo_5m_igual_cierre_pct"]).sort_values("ultimo_5m_igual_cierre_pct")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    a1.barh(t["ticker"], t["ultimo_5m_igual_cierre_pct"], color=COLOR["bmv"])
    a1.set_title("% de días en que la última barra = Close diario (±0.1 %)")
    a1.set_xlim(0, 100)
    y = np.arange(len(t))
    a2.barh(y - 0.2, t["err_mediano_max_vs_high_pct"], height=0.4, label="máx 5 min vs High")
    a2.barh(y + 0.2, t["err_mediano_min_vs_low_pct"], height=0.4, label="mín 5 min vs Low")
    a2.set_title("Error mediano (%) al reconstruir High/Low con barras de 5 min")
    a2.legend(frameon=False)
    a1.tick_params(axis="y", labelsize=7)
    _guardar(fig, "08_intradia_vs_diario.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hoy = pd.Timestamp.today().normalize()
    grupos = grupo_de_simbolo()
    diario = cargar_diario()
    intradia = cargar_intradia()
    cal_usa = diario["SPY"].index
    cal_bmv = calendario_bmv(diario, grupos)

    tabla = diagnostico_diario(diario, grupos, cal_bmv, cal_usa)
    anual = completitud_anual(diario, cal_bmv, cal_usa)
    tint, barras = diagnostico_intradia(intradia, diario, cal_bmv)
    faltan = esperados_sin_archivo()

    tabla.to_csv(OUT / "calidad_diario.csv", index=False)
    anual.round(2).to_csv(OUT / "completitud_anual.csv")
    tint.to_csv(OUT / "calidad_intradia.csv", index=False)
    pd.DataFrame(faltan, columns=["simbolo", "grupo"]).to_csv(OUT / "sin_archivo.csv", index=False)

    print(f"Diario: {len(diario)} archivos, {int(tabla['filas'].sum()):,} filas")
    print(f"Intradía: {len(intradia)} archivos, {int(tint['barras'].sum()):,} barras")
    print(f"Esperados sin archivo: {', '.join(s for s, _ in faltan) or 'ninguno'}")

    g_ejemplo_ohlcv(diario)
    g_cobertura(diario, tabla, faltan, hoy)
    g_completitud_anual(anual, tabla)
    g_problemas(tabla)
    g_tipo_cambio(diario, tabla)
    g_intradia_calendario(barras, cal_bmv)
    g_intradia_un_dia(intradia, diario)
    g_intradia_vs_diario(tint)
    print(f"Salida en {OUT}")


if __name__ == "__main__":
    main()
