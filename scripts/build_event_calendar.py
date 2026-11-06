"""
Construye el calendario de eventos desde fuentes OFICIALES (M7b).

  FOMC    federalreserve.gov (fomchistoricalYYYY.htm 2000-2020 y fomccalendars.htm 2021+)
  Banxico banxico.org.mx, lista oficial de anuncios de política monetaria
  INPC    INEGI, calendario de difusión anual (PDF); solo años cuyo PDF se parsea
          sin ambigüedad (12 fechas junto a la fila "2.ª quincena y mensual")

Una fecha que no se pueda verificar NO se incluye; los huecos se anotan en
data/calendars/unverified.txt. Con --macro baja VIX y DXY (yfinance) a parquet.

Uso:
    python scripts/build_event_calendar.py            # eventos
    python scripts/build_event_calendar.py --macro    # además VIX/DXY
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ml import vol_calendar as vc  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0"}
FOMC_HIST = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
FOMC_CUR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BANXICO = ("https://www.banxico.org.mx/publicaciones-y-prensa/"
           "anuncios-de-las-decisiones-de-politica-monetaria/anuncios-politica-monetaria-t.html")
INEGI = "https://www.inegi.org.mx/contenidos/saladeprensa/doc/cal_{y}.pdf"
INEGI_YEARS = range(2016, 2027)


def fetch(url: str) -> bytes:
    with urlopen(Request(url, headers=UA), timeout=60) as r:
        return r.read()


def build_events() -> tuple[pd.DataFrame, list[str]]:
    rows, unverified = [], []

    for y in range(2000, 2021):
        url = FOMC_HIST.format(y=y)
        try:
            dates = vc.parse_fomc_year_page(fetch(url).decode("utf-8", "ignore"), y)
        except Exception as exc:
            dates = []
            unverified.append(f"FOMC {y}: fallo al bajar {url} ({exc})")
        if not dates:
            unverified.append(f"FOMC {y}: sin reuniones parseadas")
        rows += [(d, "fomc", url, "reunión programada; día de la decisión") for d in dates]
    cur = vc.parse_fomc_current_page(fetch(FOMC_CUR).decode("utf-8", "ignore"))
    # 2027+ es tentativo ("cada fecha es tentativa hasta confirmarse en la reunión previa"): se excluye
    rows += [(d, "fomc", FOMC_CUR, "reunión programada; día de la decisión") for d in cur if d.year <= 2026]

    bx = vc.parse_banxico_announcements(fetch(BANXICO).decode("utf-8", "ignore"))
    rows += [(d, "banxico", BANXICO, "anuncio de política monetaria") for d in bx]
    unverified.append("Banxico: la lista oficial no distingue anuncios programados de extraordinarios; "
                      "antes de 2008 los anuncios eran irregulares (se incluyen tal cual aparecen).")

    for y in INEGI_YEARS:
        url = INEGI.format(y=y)
        try:
            with tempfile.TemporaryDirectory() as td:
                pdf = Path(td) / "c.pdf"
                pdf.write_bytes(fetch(url))
                txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True,
                                     text=True, check=True).stdout
            dates = vc.parse_inegi_inpc_monthly(txt, y)
        except Exception as exc:
            dates = []
            unverified.append(f"INPC {y}: fallo {url} ({exc})")
        if len(dates) != 12:
            unverified.append(f"INPC {y}: formato del PDF no parseable sin ambigüedad; sin fechas")
        rows += [(d, "inpc", url, "INPC 2.ª quincena y mensual; 06:00") for d in dates]
    unverified.append("INPC antes de 2016: sin calendario PDF accesible en la ruta oficial; sin fechas.")

    cal = pd.DataFrame(rows, columns=["date", "event", "source", "note"])
    cal = cal.drop_duplicates(["date", "event"]).sort_values(["date", "event"]).reset_index(drop=True)
    return cal, unverified


def build_macro(path: Path) -> None:
    import yfinance as yf

    out = {}
    for name, sym in (("VIX", "^VIX"), ("DXY", "DX-Y.NYB")):
        d = yf.download(sym, start="2000-01-01", progress=False, auto_adjust=False)
        c = d["Close"]
        out[name] = (c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c)
    macro = pd.DataFrame(out)
    macro.index = pd.DatetimeIndex(macro.index).tz_localize(None)
    macro.to_parquet(path)
    print(f"macro: {len(macro)} filas {macro.index.min().date()} → {macro.index.max().date()} → {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--macro", action="store_true", help="bajar VIX/DXY a parquet")
    a = p.parse_args()
    vc.CAL_DIR.mkdir(parents=True, exist_ok=True)
    cal, unverified = build_events()
    cal.to_csv(vc.EVENTS_CSV, index=False)
    (vc.CAL_DIR / "unverified.txt").write_text("\n".join(unverified) + "\n")
    print(cal.groupby("event")["date"].agg(["count", "min", "max"]))
    print(f"→ {vc.EVENTS_CSV}")
    for u in unverified:
        print("  [no verificado]", u)
    if a.macro:
        build_macro(vc.MACRO_PARQUET)


if __name__ == "__main__":
    main()
