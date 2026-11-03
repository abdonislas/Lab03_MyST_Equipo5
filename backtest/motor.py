# backtest/motor.py
# Backtest oficial multi-pata: la senal del cierre de t se ejecuta al cierre de t+1,
# se invierte el 80% del capital segun el vector de pesos, comision de 0.125% por pata
# y prestamo de 0.25% anual sobre los cortos. Cada muestra arranca plana con 1,000,000.
# Politicas de rebalanceo: diario, bandas de 1%, 2% y 5%, y sin rebalanceo (acciones fijas).
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

ART = "artifacts"
MUESTRAS = ["train", "test", "val"]
METODOS = ["kalman", "estatico"]
# Bandas escogidas con la desviacion observada sin rebalanceo: mediana de 0.8% (estatico)
# y 1.5% (Kalman), percentil 95 de 2.7% y 4.5%. Una banda de 10% casi nunca se activa.
POLITICAS = {"diario": 0.0, "banda_1": 0.01, "banda_2": 0.02, "banda_5": 0.05,
             "sin_rebalanceo": float("inf")}
POLITICA_OFICIAL = "diario"
DIRECCION = {1: "largo spread", -1: "corto spread"}


@dataclass(frozen=True)
class Config:
    capital: float = 1_000_000.0
    exposicion: float = 0.80
    comision: float = 0.00125          # 0.125% por pata, por operacion
    tasa_prestamo: float = 0.0025      # 0.25% anual sobre el nocional en corto
    base_dias: int = 365


def patas_del_grupo(chosen):
    """Lista de activos con la pata de normalizacion primero (peso +1)."""
    patas = chosen["grupo"].split("+")
    norm = chosen.get("pata_normalizacion", patas[0])
    return [norm] + [p for p in patas if p != norm]


def pesos_por_dia(df, patas):
    """Matriz (dias x patas) con el vector de pesos w_t de cada dia.
    Contrato: si las senales traen columnas w_<activo> se usan tal cual (sirve para
    cualquier n). Si no, y el grupo es un par, se usa w = (1, -beta)."""
    cols = [f"w_{p}" for p in patas]
    if all(c in df.columns for c in cols):
        return df[cols].to_numpy(dtype=float)
    if "beta" in df.columns and len(patas) == 2:
        b = df["beta"].to_numpy(dtype=float)
        return np.column_stack([np.ones_like(b), -b])
    raise ValueError("Las senales deben traer una columna w_<activo> por pata "
                     "(o 'beta' si el grupo es un par).")


def nocional_objetivo(V, direccion, w, exposicion):
    """Nocional con signo por pata: d * exposicion * V * w / sum|w|."""
    w = np.asarray(w, dtype=float)
    return direccion * exposicion * V * w / np.abs(w).sum()


def desviacion_mezcla(actual, objetivo):
    """Distancia L1 entre la mezcla actual y la objetivo, en proporciones del bruto."""
    a = actual / np.abs(actual).sum()
    o = objetivo / np.abs(objetivo).sum()
    return float(np.abs(a - o).sum())


def a_bool(serie):
    """Convierte una columna leida de CSV (bool o texto 'True'/'False') a array bool."""
    if serie.dtype == bool:
        return serie.to_numpy()
    return serie.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()


# Simulacion de una muestra

def simular(fechas, precios, pesos, d, banda, patas, cfg=Config(), motivos=None):
    """Simula una muestra dia por dia.
    fechas: DatetimeIndex; precios: (n x k); pesos: (n x k) vector w de cada dia;
    d: senal en {-1, 0, +1} calculada al cierre de cada dia (se ejecuta al dia siguiente);
    banda: politica de rebalanceo; motivos: motivo_cierre de las reglas (opcional).
    Regresa (diario, operaciones, fills) como DataFrames."""
    precios = np.asarray(precios, dtype=float)
    pesos = np.asarray(pesos, dtype=float)
    d = np.asarray(d).astype(int)
    n, k = precios.shape
    efectivo = float(cfg.capital)
    q = np.zeros(k)                      # acciones con signo por pata
    pos = 0                              # direccion de la posicion abierta
    op = None                            # operacion abierta (dict)
    n_ops = 0
    diario, operaciones, fills = [], [], []

    def operar(t, q_nuevo, tipo):
        """Lleva las acciones a q_nuevo al cierre de t. Regresa la comision pagada."""
        nonlocal efectivo, q
        dq = q_nuevo - q
        nocional = dq * precios[t]                     # + compra, - venta
        com_pata = np.abs(nocional) * cfg.comision
        efectivo -= float(nocional.sum()) + float(com_pata.sum())
        for i in range(k):
            if dq[i] != 0:
                fills.append({"fecha": fechas[t], "operacion": n_ops, "pata": patas[i],
                              "tipo": tipo, "cantidad": dq[i], "precio": precios[t, i],
                              "nocional": nocional[i], "comision": com_pata[i]})
        q = np.asarray(q_nuevo, dtype=float).copy()
        return float(com_pata.sum())

    for t in range(n):
        pnl_bruto = prestamo = com_dia = 0.0
        desv, rebalanceo = np.nan, False
        if t > 0:
            # 1) P&L de precios de las acciones que se traian (cierre t-1 -> cierre t)
            pnl_bruto = float(q @ (precios[t] - precios[t - 1]))
            # 2) Prestamo de las patas cortas por los dias naturales transcurridos
            dias = (fechas[t] - fechas[t - 1]).days
            corto = float(-(q[q < 0] * precios[t - 1][q < 0]).sum())
            prestamo = corto * cfg.tasa_prestamo * dias / cfg.base_dias
            efectivo -= prestamo
            if op is not None:
                op["pnl_bruto"] += pnl_bruto
                op["prestamo"] += prestamo
            # 3) Ejecucion al cierre de t de la senal de ayer (el ultimo dia se liquida)
            objetivo = 0 if t == n - 1 else int(d[t - 1])
            if objetivo != pos:
                if pos != 0:                                       # cerrar
                    forzado = (t == n - 1 and d[t - 1] != 0)
                    c = operar(t, np.zeros(k), "liquidacion" if forzado else "salida")
                    com_dia += c
                    op["comisiones"] += c
                    motivo = "fin_muestra" if forzado else (
                        str(motivos[t - 1]) if motivos is not None and str(motivos[t - 1]) not in ("", "nan") else "salida")
                    op.update({"fecha_salida": fechas[t], "dias": t - op.pop("_t0"),
                               "motivo_cierre": motivo})
                    op["pnl_neto"] = op["pnl_bruto"] - op["comisiones"] - op["prestamo"]
                    op["retorno"] = op["pnl_neto"] / op["capital_entrada"]
                    operaciones.append(op)
                    op, pos = None, 0
                if objetivo != 0:                                  # abrir
                    n_ops += 1
                    V = efectivo + float(q @ precios[t])
                    N = nocional_objetivo(V, objetivo, pesos[t - 1], cfg.exposicion)
                    op = {"operacion": n_ops, "direccion": DIRECCION[objetivo],
                          "fecha_entrada": fechas[t], "_t0": t, "capital_entrada": V,
                          "pnl_bruto": 0.0, "comisiones": 0.0, "prestamo": 0.0,
                          "n_rebalanceos": 0}
                    c = operar(t, N / precios[t], "entrada")
                    com_dia += c
                    op["comisiones"] += c
                    pos = objetivo
            elif pos != 0:                                         # rebalanceo
                V = efectivo + float(q @ precios[t])
                N = nocional_objetivo(V, pos, pesos[t - 1], cfg.exposicion)
                desv = desviacion_mezcla(q * precios[t], N)
                if desv > banda:
                    c = operar(t, N / precios[t], "rebalanceo")
                    com_dia += c
                    op["comisiones"] += c
                    op["n_rebalanceos"] += 1
                    rebalanceo = True
        valores = q * precios[t]
        fila = {"fecha": fechas[t], "d": pos, "efectivo": efectivo}
        for i, p in enumerate(patas):
            fila[f"q_{p}"] = q[i]
        for i, p in enumerate(patas):
            fila[f"valor_{p}"] = valores[i]
        fila.update({"capital": efectivo + float(valores.sum()), "pnl_bruto": pnl_bruto,
                     "comision": com_dia, "prestamo": prestamo, "rebalanceo": rebalanceo,
                     "desviacion": desv})
        diario.append(fila)

    col_ops = ["operacion", "direccion", "fecha_entrada", "fecha_salida", "dias",
               "motivo_cierre", "capital_entrada", "pnl_bruto", "comisiones", "prestamo",
               "pnl_neto", "retorno", "n_rebalanceos"]
    col_fills = ["fecha", "operacion", "pata", "tipo", "cantidad", "precio", "nocional", "comision"]
    return (pd.DataFrame(diario),
            pd.DataFrame(operaciones, columns=col_ops),
            pd.DataFrame(fills, columns=col_fills))


# Un metodo completo: tres muestras x politicas

def correr_metodo(senales, patas, theta, reglas_fn=None, politicas=POLITICAS, cfg=Config()):
    """Corre las tres muestras de un metodo con cada politica de rebalanceo.
    senales: DataFrame de artifacts/senales_<metodo>.csv (indice fecha).
    theta: dict con z_in, z_out, z_stop, h_max. reglas_fn: funcion de reglas
    (por defecto strategy.reglas.aplicar_reglas). Cada muestra arranca plana."""
    if reglas_fn is None:
        from strategy.reglas import aplicar_reglas as reglas_fn
    burn = a_bool(senales["en_burn_in"])
    muestra = senales["muestra"].astype(str).to_numpy()
    diarios, ops, fills = [], [], []
    for m in MUESTRAS:
        sub = senales.loc[(muestra == m) & ~burn]
        reglas = reglas_fn(sub["z"].to_numpy(dtype=float), **theta)
        d = reglas["d"].to_numpy()
        motivos = reglas["motivo_cierre"].to_numpy()
        precios = sub[patas].to_numpy(dtype=float)
        pesos = pesos_por_dia(sub, patas)
        for politica, banda in politicas.items():
            dia, op, fl = simular(sub.index, precios, pesos, d, banda, patas, cfg, motivos)
            for tabla in (dia, op, fl):
                tabla.insert(0, "muestra", m)
                tabla.insert(0, "politica", politica)
            diarios.append(dia)
            ops.append(op)
            fills.append(fl)
    return (pd.concat(diarios, ignore_index=True),
            pd.concat(ops, ignore_index=True),
            pd.concat(fills, ignore_index=True))


def cargar_senales(metodo, art=ART):
    return pd.read_csv(f"{art}/senales_{metodo}.csv", index_col=0, parse_dates=True)


def main(art=ART):
    chosen = json.load(open(f"{art}/chosen_group.json"))
    theta_opt = json.load(open(f"{art}/theta_optimo.json"))
    patas = patas_del_grupo(chosen)
    cfg = Config()
    claves = ["z_in", "z_out", "z_stop", "h_max"]

    diarios, ops, fills = [], [], []
    for metodo in METODOS:
        theta = {c: theta_opt[metodo][c] for c in claves}
        dia, op, fl = correr_metodo(cargar_senales(metodo, art), patas, theta, cfg=cfg)
        for tabla in (dia, op, fl):
            tabla.insert(0, "metodo", metodo)
        diarios.append(dia)
        ops.append(op)
        fills.append(fl)
    diario = pd.concat(diarios, ignore_index=True)
    operaciones = pd.concat(ops, ignore_index=True)
    fl = pd.concat(fills, ignore_index=True)

    diario.to_csv(f"{art}/bt_diario.csv", index=False, float_format="%.10g")
    operaciones.to_csv(f"{art}/bt_operaciones.csv", index=False, float_format="%.10g")
    fl.to_csv(f"{art}/bt_fills.csv", index=False, float_format="%.10g")
    with open(f"{art}/bt_config.json", "w") as f:
        json.dump({**asdict(cfg), "patas": patas, "politicas": {k: (None if np.isinf(v) else v)
                   for k, v in POLITICAS.items()}, "politica_oficial": POLITICA_OFICIAL,
                   "theta": {m: {c: theta_opt[m][c] for c in claves} for m in METODOS},
                   "nota": "politica con banda null = sin rebalanceo (banda infinita)"},
                  f, indent=2)

    print(f"Patas: {patas}  |  politicas: {list(POLITICAS)}  |  oficial: {POLITICA_OFICIAL}")
    resumen = (operaciones.groupby(["metodo", "politica", "muestra"])
               .agg(operaciones=("operacion", "count"), pnl_neto=("pnl_neto", "sum"),
                    rebalanceos=("n_rebalanceos", "sum"))
               .round(0))
    print(resumen.to_string())
    print(f"\nGuardado: {art}/bt_diario.csv ({len(diario)} filas), bt_operaciones.csv "
          f"({len(operaciones)}), bt_fills.csv ({len(fl)}), bt_config.json")


if __name__ == "__main__":
    main()
