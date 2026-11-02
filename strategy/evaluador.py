# strategy/evaluador.py
# Evaluador SIMPLE de P&L, solo para el grid search de theta (paso 6 de Powell).
# NO es el backtest oficial: la contabilidad completa (rebalanceo, comparaciones, etc.)
# la hace el integrante C en backtest/. Aqui solo necesitamos un Sharpe neto comparable
# entre configuraciones.
#
# Supuestos:
#   - La senal d_t se calcula con el cierre de t y se ejecuta al CIERRE de t+1
#     (no tenemos precios de apertura).
#   - Capital inicial 1,000,000; exposicion 80% del capital al entrar:
#       N_AMT  = 0.8 * V * 1    / (1 + beta)
#       N_SBAC = 0.8 * V * beta / (1 + beta)
#     con el beta del dia de la senal. Las acciones quedan FIJAS mientras la posicion
#     esta abierta (sin rebalanceo).
#   - d = +1 (largo spread): compra AMT, vende SBAC.  d = -1: al reves.
#   - Comision 0.125% del nocional de cada pata al entrar y al salir.
#   - Prestamo 0.25% anual sobre el nocional de la pata corta, por dia natural (base 365).
#   - Al final de la muestra se liquida la posicion abierta (cada muestra es independiente).
import numpy as np
import pandas as pd

CAPITAL = 1_000_000
EXPOSICION = 0.80
COMISION = 0.00125          # 0.125% por pata
TASA_PRESTAMO = 0.0025      # 0.25% anual


def evaluar(df, capital=CAPITAL, exposicion=EXPOSICION, comision=COMISION,
            tasa_prestamo=TASA_PRESTAMO):
    """Simula la estrategia dia por dia.
    Recibe df (indice fecha) con columnas AMT, SBAC (precios), beta y d (senal).
    Regresa diccionario con la curva de capital y las metricas."""
    fechas = df.index
    pa, ps = df["AMT"].values, df["SBAC"].values
    beta, d = df["beta"].values, df["d"].values
    n = len(df)

    V = float(capital)                 # capital (valor de la cuenta)
    n_a, n_s = 0.0, 0.0                # acciones con signo (+ largo, - corto)
    pos = 0                            # direccion actual: +1, -1 o 0
    pnl_op = 0.0                       # P&L acumulado de la operacion abierta
    ops = []                           # P&L neto de cada operacion cerrada
    comisiones, prestamo = 0.0, 0.0
    curva = [V]

    for t in range(1, n):
        # 1) Mark-to-market de las acciones abiertas (cierre t-1 -> cierre t)
        pnl = n_a * (pa[t] - pa[t - 1]) + n_s * (ps[t] - ps[t - 1])
        # 2) Costo de prestamo de la pata corta por los dias naturales transcurridos
        dias = (fechas[t] - fechas[t - 1]).days
        nocional_corto = (-n_a * pa[t - 1] if n_a < 0 else 0.0) + \
                         (-n_s * ps[t - 1] if n_s < 0 else 0.0)
        costo_prestamo = nocional_corto * tasa_prestamo * dias / 365
        V += pnl - costo_prestamo
        pnl_op += pnl - costo_prestamo
        prestamo += costo_prestamo

        # 3) Ejecucion al cierre de t de la senal de ayer (d_{t-1})
        objetivo = d[t - 1]
        if t == n - 1:
            objetivo = 0                                # ultimo dia: liquidar todo
        if objetivo != pos:
            if pos != 0:                                # cerrar la posicion actual
                c = (abs(n_a) * pa[t] + abs(n_s) * ps[t]) * comision
                V -= c
                comisiones += c
                ops.append(pnl_op - c)                  # P&L neto de la operacion
                n_a, n_s, pos = 0.0, 0.0, 0
            if objetivo != 0:                           # abrir la nueva posicion
                b = beta[t - 1]                         # beta del dia de la senal
                N_a = exposicion * V / (1 + b)          # nocional AMT
                N_s = exposicion * V * b / (1 + b)      # nocional SBAC
                n_a = objetivo * N_a / pa[t]            # +1: largo AMT
                n_s = -objetivo * N_s / ps[t]           # +1: corto SBAC
                c = (N_a + N_s) * comision
                V -= c
                comisiones += c
                pnl_op = -c                             # la operacion arranca pagando comision
                pos = objetivo
        curva.append(V)

    curva = pd.Series(curva, index=fechas, name="capital")
    return {"curva": curva, **metricas(curva, ops, comisiones, prestamo)}


def metricas(curva, ops, comisiones, prestamo):
    """Sharpe anualizado (raiz de 252), retorno total, max drawdown, operaciones,
    win rate, comisiones y prestamo totales."""
    r = curva.pct_change().dropna()                     # rendimientos diarios
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    drawdown = curva / curva.cummax() - 1               # caida desde el maximo previo
    ops = np.asarray(ops)
    return {
        "sharpe": float(sharpe),
        "retorno_total": float(curva.iloc[-1] / curva.iloc[0] - 1),
        "max_drawdown": float(drawdown.min()),
        "n_ops": int(len(ops)),
        "win_rate": float((ops > 0).mean()) if len(ops) else np.nan,
        "comisiones": float(comisiones),
        "prestamo": float(prestamo),
    }
