#primero queremos descargar los datos y dividirlos en train/test/val dados en el proyecto 60/20/20
import os
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["VZ", "T", "BCE", "DIS", "AMX", "NFLX"]
START, END = "2015-10-01", "2026-10-01"   # ~10 años, fijo => reproducible
SEED = 42
ART = "artifacts"


EVENTOS = {
    "T":    ("2022-04-11", "Spin-off de WarnerMedia (AT&T)"),
    "DIS":  ("2019-03-20", "Cierre de compra de Fox"),
    "NFLX": ("2015-07-15", "Split 7:1 (previo a la ventana)"),
}


def descargar():
    os.makedirs(ART, exist_ok=True)
    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    px = raw["Close"].copy()            # auto_adjust=True => Close ya es ajustado
    px = px[TICKERS]                    # orden fijo de columnas
    return px


def chequear_huecos(px):
    """Reporta fechas faltantes por activo y primera fecha con dato (series cortas)."""
    rep = []
    for t in px.columns:
        s = px[t]
        rep.append({
            "ticker": t,
            "primer_dato": s.first_valid_index().date(),
            "ultimo_dato": s.last_valid_index().date(),
            "n_obs": int(s.notna().sum()),
            "n_faltantes": int(s.isna().sum()),
        })
    return pd.DataFrame(rep)


def chequear_saltos(px, k=6.0):
    """Marca retornos diarios con |z| > k como posibles saltos estructurales."""
    ret = np.log(px).diff()
    z = (ret - ret.mean()) / ret.std()
    saltos = []
    for t in px.columns:
        for fecha in z.index[z[t].abs() > k]:
            nota = ""
            if t in EVENTOS and abs((pd.Timestamp(EVENTOS[t][0]) - fecha).days) <= 5:
                nota = EVENTOS[t][1]
            saltos.append({"ticker": t, "fecha": fecha.date(),
                           "ret_%": round(100 * ret.loc[fecha, t], 2),
                           "z": round(z.loc[fecha, t], 1), "evento": nota})
    return pd.DataFrame(saltos).sort_values("fecha")


def split_60_20_20(px):
    """Split CRONOLÓGICO sin lookahead. Alinea a días comunes a todos los activos."""
    px = px.dropna(how="any")           # solo fechas con los 6 activos
    n = len(px)
    i1, i2 = int(n * 0.60), int(n * 0.80)
    return px.iloc[:i1], px.iloc[i1:i2], px.iloc[i2:]


def main():
    np.random.seed(SEED)
    px = descargar()
    px.to_csv(f"{ART}/prices_full.csv")

    huecos = chequear_huecos(px)
    saltos = chequear_saltos(px)
    huecos.to_csv(f"{ART}/data_gaps.csv", index=False)
    saltos.to_csv(f"{ART}/data_jumps.csv", index=False)

    train, test, val = split_60_20_20(px)
    train.to_csv(f"{ART}/prices_train.csv")
    test.to_csv(f"{ART}/prices_test.csv")
    val.to_csv(f"{ART}/prices_val.csv")

    print("=== Huecos / cobertura ===");  print(huecos.to_string(index=False))
    print("\n=== Saltos detectados ===");  print(saltos.to_string(index=False))
    print(f"\nSplit -> train={len(train)}  test={len(test)}  val={len(val)}")
    print(f"Fechas: {train.index[0].date()} .. {val.index[-1].date()}")


if __name__ == "__main__":
    main()