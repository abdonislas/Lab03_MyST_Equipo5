# performance/metricas.py
# Metricas de desempeno a partir de las salidas del backtest (no vuelve a simular):
# Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, Sharpe movil de 252 dias,
# atribucion de costos y costo del rebalanceo contra no rebalancear.
import numpy as np
import pandas as pd

ART = "artifacts"
DIAS_ANIO = 252
VENTANA_SHARPE = 252
MUESTRAS = ["train", "test", "val"]

# Metricas individuales

def retornos(capital):
    return pd.Series(capital, dtype=float).pct_change().dropna()


def sharpe(r):
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(DIAS_ANIO)) if sd > 0 else np.nan


def sortino(r):
    abajo = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return float(r.mean() / abajo * np.sqrt(DIAS_ANIO)) if abajo > 0 else np.nan


def drawdown(capital):
    capital = pd.Series(capital, dtype=float)
    return capital / capital.cummax() - 1


def max_drawdown(capital):
    return float(drawdown(capital).min())


def retorno_anualizado(capital):
    capital = pd.Series(capital, dtype=float)
    n = len(capital) - 1
    return float((capital.iloc[-1] / capital.iloc[0]) ** (DIAS_ANIO / n) - 1) if n > 0 else np.nan


def calmar(capital):
    mdd = max_drawdown(capital)
    return float(retorno_anualizado(capital) / abs(mdd)) if mdd < 0 else np.nan


def win_rate(pnl):
    pnl = np.asarray(pnl, dtype=float)
    return float((pnl > 0).mean()) if len(pnl) else np.nan


def profit_factor(pnl):
    pnl = np.asarray(pnl, dtype=float)
    ganado, perdido = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    if perdido > 0:
        return float(ganado / perdido)
    return np.inf if ganado > 0 else np.nan


def sharpe_movil(r, ventana=VENTANA_SHARPE):
    return r.rolling(ventana).mean() / r.rolling(ventana).std() * np.sqrt(DIAS_ANIO)


# Curvas

def curvas(diario):
    """Capital por muestra (cada una arranca en el capital inicial) y encadenado
    (test arranca con el capital final de train y val con el de test)."""
    partes = []
    for (metodo, politica), g in diario.groupby(["metodo", "politica"], sort=False):
        factor = 1.0
        for m in MUESTRAS:
            s = g[g["muestra"] == m].sort_values("fecha")
            if s.empty:
                continue
            c = s[["fecha", "metodo", "politica", "muestra", "d", "capital"]].copy()
            c["capital_encadenado"] = c["capital"] * factor
            c["drawdown"] = drawdown(c["capital"]).to_numpy()
            factor *= c["capital"].iloc[-1] / c["capital"].iloc[0]
            partes.append(c)
    out = pd.concat(partes, ignore_index=True)
    res = []
    for _, g in out.groupby(["metodo", "politica"], sort=False):
        g = g.sort_values("fecha").copy()
        g["drawdown_encadenado"] = drawdown(g["capital_encadenado"]).to_numpy()
        r = g["capital_encadenado"].pct_change()
        g["retorno"] = r
        g["sharpe_movil_252"] = sharpe_movil(r)
        res.append(g)
    return pd.concat(res, ignore_index=True)


# Tablas

def fila_metricas(capital, dia, ops, fills):
    r = retornos(capital)
    pnl = ops["pnl_neto"].to_numpy(dtype=float)
    reb = fills[fills["tipo"] == "rebalanceo"]
    return {
        "sharpe": sharpe(r), "sortino": sortino(r), "calmar": calmar(capital),
        "max_drawdown": max_drawdown(capital), "win_rate": win_rate(pnl),
        "profit_factor": profit_factor(pnl),
        "retorno_total": float(capital.iloc[-1] / capital.iloc[0] - 1),
        "retorno_anual": retorno_anualizado(capital),
        "volatilidad_anual": float(r.std() * np.sqrt(DIAS_ANIO)),
        "n_operaciones": int(len(ops)),
        "pnl_bruto": float(dia["pnl_bruto"].sum()), "comisiones": float(dia["comision"].sum()),
        "prestamo": float(dia["prestamo"].sum()),
        "pnl_neto": float(dia["pnl_bruto"].sum() - dia["comision"].sum() - dia["prestamo"].sum()),
        "n_rebalanceos": int(dia["rebalanceo"].astype(bool).sum()),
        "comisiones_rebalanceo": float(reb["comision"].sum()),
        "dias": int(len(dia)), "dias_con_posicion": int((dia["d"] != 0).sum()),
    }


def tabla_metricas(diario, ops, fills, cur):
    filas = []
    claves = ["metodo", "politica"]
    for (metodo, politica), g in diario.groupby(claves, sort=False):
        o = ops[(ops["metodo"] == metodo) & (ops["politica"] == politica)]
        f = fills[(fills["metodo"] == metodo) & (fills["politica"] == politica)]
        c = cur[(cur["metodo"] == metodo) & (cur["politica"] == politica)]
        for m in MUESTRAS:
            gm = g[g["muestra"] == m].sort_values("fecha")
            filas.append({"metodo": metodo, "politica": politica, "muestra": m,
                          **fila_metricas(gm["capital"].reset_index(drop=True), gm,
                                          o[o["muestra"] == m], f[f["muestra"] == m])})
        c = c.sort_values("fecha")
        filas.append({"metodo": metodo, "politica": politica, "muestra": "completa",
                      **fila_metricas(c["capital_encadenado"].reset_index(drop=True), g, o, f)})
    t = pd.DataFrame(filas)
    # en "completa" el P&L neto se mide sobre el capital encadenado
    return t


def tabla_rebalanceo(met):
    """Costo de cada politica contra no rebalancear, por metodo y muestra."""
    t = met[met["muestra"] != "completa"][["metodo", "muestra", "politica", "n_rebalanceos",
                                           "comisiones_rebalanceo", "comisiones", "prestamo",
                                           "pnl_neto", "retorno_total", "sharpe", "max_drawdown"]].copy()
    base = t[t["politica"] == "sin_rebalanceo"].set_index(["metodo", "muestra"])
    idx = pd.MultiIndex.from_frame(t[["metodo", "muestra"]])
    t["dif_pnl_neto_vs_sin"] = t["pnl_neto"].to_numpy() - base.loc[idx, "pnl_neto"].to_numpy()
    t["dif_comisiones_vs_sin"] = t["comisiones"].to_numpy() - base.loc[idx, "comisiones"].to_numpy()
    t["dif_sharpe_vs_sin"] = t["sharpe"].to_numpy() - base.loc[idx, "sharpe"].to_numpy()
    return t


def validar_contra_preliminares(met, art=ART):
    """Compara la politica sin rebalanceo contra artifacts/metricas_preliminares.csv
    (evaluador simple de strategy/evaluador.py). Deben coincidir."""
    try:
        pre = pd.read_csv(f"{art}/metricas_preliminares.csv")
    except FileNotFoundError:
        return None
    sin = met[(met["politica"] == "sin_rebalanceo") & (met["muestra"] != "completa")]
    a = sin.set_index(["metodo", "muestra"])
    b = pre.set_index(["metodo", "muestra"])
    pares = {"sharpe": "sharpe", "retorno_total": "retorno_total", "max_drawdown": "max_drawdown",
             "n_operaciones": "n_ops", "win_rate": "win_rate", "comisiones": "comisiones",
             "prestamo": "prestamo"}
    return pd.DataFrame({k: (a.loc[b.index, k] - b[v]).abs() for k, v in pares.items()})


def main(art=ART):
    diario = pd.read_csv(f"{art}/bt_diario.csv", parse_dates=["fecha"])
    ops = pd.read_csv(f"{art}/bt_operaciones.csv", parse_dates=["fecha_entrada", "fecha_salida"])
    fills = pd.read_csv(f"{art}/bt_fills.csv", parse_dates=["fecha"])

    cur = curvas(diario)
    met = tabla_metricas(diario, ops, fills, cur)
    reb = tabla_rebalanceo(met)
    cur.to_csv(f"{art}/bt_curvas.csv", index=False, float_format="%.10g")
    met.to_csv(f"{art}/bt_metricas.csv", index=False)
    reb.to_csv(f"{art}/bt_rebalanceo.csv", index=False)

    cols = ["sharpe", "sortino", "calmar", "max_drawdown", "win_rate", "profit_factor",
            "retorno_total", "n_operaciones", "comisiones", "prestamo"]
    oficial = met[met["politica"] == "diario"].set_index(["metodo", "muestra"])[cols]
    print("=== Metricas, politica oficial (rebalanceo diario) ===")
    print(oficial.round(4).to_string())
    print("\n=== Costo del rebalanceo (contra no rebalancear) ===")
    print(reb[["metodo", "muestra", "politica", "n_rebalanceos", "comisiones_rebalanceo",
               "dif_pnl_neto_vs_sin", "dif_sharpe_vs_sin"]].round(2).to_string(index=False))
    dif = validar_contra_preliminares(met, art)
    if dif is not None:
        print(f"\nValidacion contra metricas_preliminares.csv (sin rebalanceo): "
              f"diferencia maxima = {float(np.nanmax(dif.to_numpy())):.2e}")
    print(f"\nGuardado: {art}/bt_metricas.csv, bt_curvas.csv, bt_rebalanceo.csv")


if __name__ == "__main__":
    main()
