"""Recuperacion de registros historicos invalidos."""

from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import time

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.store import BMV_TICKERS, MARKET_DIR

FECHA = pd.Timestamp("2026-10-23")

COLUMNAS = ["Open", "High", "Low", "Close", "Volume"]

SIMBOLOS = sorted(set(BMV_TICKERS.values()) | {
    "NAFTRACISHRS.MX",
    "NAFTRAC.MX"
})

EJECUCION = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

RESPALDO = Path("data/market_backups") / EJECUCION
REPORTE = Path("data/quality") / EJECUCION


def registro_valido(fila):

    precios = fila[["Open", "High", "Low", "Close"]]

    return bool(
        np.isfinite(precios).all()
        and (precios > 0).all()
        and np.isfinite(fila["Volume"])
        and fila["Volume"] > 0
        and fila["High"] >= max(
            fila["Open"], fila["Close"], fila["Low"]
        )
        and fila["Low"] <= min(
            fila["Open"], fila["Close"]
        )
    )


def descargar(simbolo):

    df = yf.download(
        simbolo,
        start="2026-10-22",
        end="2026-10-27",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or FECHA not in df.index:
        return None

    fila = df.loc[FECHA, COLUMNAS]

    if not registro_valido(fila):
        return None

    return fila


def main():

    RESPALDO.mkdir(parents=True, exist_ok=True)
    REPORTE.mkdir(parents=True, exist_ok=True)

    resultados = []

    for simbolo in SIMBOLOS:

        archivo = MARKET_DIR / (
            simbolo.replace("/", "_").replace("^", "")
            + ".parquet"
        )

        resultado = {
            "symbol": simbolo,
            "date": str(FECHA.date()),
            "source": "Yahoo Finance"
        }

        try:

            if not archivo.exists():
                resultado["status"] = "sin_archivo"

            else:

                df = pd.read_parquet(archivo).sort_index()

                if df.empty:
                    resultado["status"] = "archivo_vacio"

                elif not df.index.is_unique:
                    resultado["status"] = "fechas_duplicadas"

                elif (
                    FECHA in df.index
                    and registro_valido(df.loc[FECHA])
                ):
                    resultado["status"] = "ya_valido"

                else:

                    fila = descargar(simbolo)

                    if fila is None:
                        resultado["status"] = "no_recuperable"

                    else:

                        shutil.copy2(
                            archivo,
                            RESPALDO / archivo.name
                        )

                        if FECHA in df.index:
                            resultado["previous"] = {
                                c: float(df.loc[FECHA, c])
                                if pd.notna(df.loc[FECHA, c])
                                and np.isfinite(df.loc[FECHA, c])
                                else None
                                for c in COLUMNAS
                            }

                        df.loc[FECHA, COLUMNAS] = fila
                        df = df.sort_index()

                        assert df.index.is_unique
                        assert registro_valido(df.loc[FECHA])

                        temporal = archivo.with_suffix(".tmp")

                        df.to_parquet(
                            temporal,
                            compression="snappy"
                        )

                        temporal.replace(archivo)

                        resultado["status"] = "reparado"

                        resultado["recovered"] = {
                            c: float(fila[c])
                            for c in COLUMNAS
                        }

                    time.sleep(1)

        except Exception as e:

            resultado["status"] = "error"
            resultado["error"] = str(e)

        resultados.append(resultado)

        print(
            f"{simbolo:25} "
            f"{resultado['status']}"
        )

    from collections import Counter

    resumen = dict(Counter(
        r["status"] for r in resultados
    ))

    reporte = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance",
        "target_date": str(FECHA.date()),
        "backup_directory": str(RESPALDO),
        "summary": resumen,
        "records": resultados
    }

    salida = REPORTE / "repair_report.json"

    salida.write_text(
        json.dumps(reporte, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    print("\nRESUMEN")

    for estado, cantidad in resumen.items():
        print(estado, ":", cantidad)

    print("\nRespaldos:", RESPALDO)
    print("Reporte:", salida)
    print("\nPROCESO COMPLETADO")


if __name__ == "__main__":
    main()
