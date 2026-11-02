# kalman/filtro_pesos.py
# Filtro 1: pesos dinamicos de la relacion log(AMT) = alpha_t + beta_t * log(SBAC) + ruido.
#   Estado oculto:   z_t = (alpha_t, beta_t)'
#   Transicion:      z_t = A z_{t-1} + w_t,   A = I (paseo aleatorio),  w_t ~ N(0, Q)
#   Observacion:     y_t = C_t z_t + v_t,     C_t = [1, x_t],           v_t ~ N(0, R)
#   con y_t = log(AMT_t) y x_t = log(SBAC_t).
import numpy as np
import pandas as pd

BURN_IN = 126                                   # dias iniciales del train que no se operan
DELTAS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]         # candidatos para calibrar Q
# Solo se permite delta <= 1e-5. El hedge ratio entre dos empresas del mismo negocio
# (torres) es un parametro estructural que debe cambiar en meses, no en dias. Con
# delta >= 1e-4 la verosimilitud explica la reversion a la media del spread como
# movimientos de alfa y beta (R -> 0, beta persigue los precios), que es justo lo que
# queremos tradear. Las filas con delta mayor se calculan y guardan, pero no se eligen.
DELTA_MAX = 1e-5


def centrar_x(x, burn_in=BURN_IN):
    """Centra el regresor: x_c = x - media de x en el burn-in (solo burn-in -> causal).
    Por que: sin centrar, x = log(SBAC) ~ 5.2, asi que C_t = [1, ~5.2] y las columnas de
    alpha y beta son casi colineales. El filtro no distingue "subio el nivel" de "subio
    beta", y los cambios de nivel se cuelan en beta (corr(alpha, beta) ~ -0.98 en train).
    Centrado, alpha es el nivel cuando x esta en su media del burn-in y beta es solo la
    pendiente. El profe recomendo revisar el escalamiento de los log-precios en C_t
    cuando el filtro no se comporta bien.
    Regresa (x_c, x_media)."""
    x = np.asarray(x, dtype=float)
    x_media = float(np.mean(x[:burn_in]))            # media de log(SBAC) en el burn-in
    return x - x_media, x_media


def inicializar(y, x, burn_in=BURN_IN):
    """OLS de y sobre [1, x] en los primeros burn_in dias.
    Recibe arrays y, x. Regresa z0 = (alpha, beta) y R_ols = varianza de los residuales
    (punto de referencia para la rejilla de R en la calibracion)."""
    X = np.column_stack([np.ones(burn_in), x[:burn_in]])     # matriz de diseno [1, x]
    z0, *_ = np.linalg.lstsq(X, y[:burn_in], rcond=None)     # minimos cuadrados
    residuales = y[:burn_in] - X @ z0                         # errores del ajuste
    R_ols = float(np.var(residuales, ddof=1))                 # varianza de los residuales
    return z0, R_ols


def correr_filtro_pesos(y, x, delta, z0, R):
    """Corre el filtro de Kalman dia por dia sobre toda la serie (los estados se arrastran).
    Recibe arrays y, x, el delta de Q, el estado inicial z0 y la varianza R.
    Regresa un DataFrame (una fila por dia) con prior, posterior, P, K, innovacion e y S."""
    n = len(y)
    I = np.eye(2)                                  # identidad 2x2 (A = I)
    Q = delta / (1 - delta) * I                    # varianza del paseo aleatorio de los pesos
    z = np.asarray(z0, dtype=float).copy()         # z_{0|0}: arranca en el OLS del burn-in
    P = I.copy()                                   # P_{0|0} = I (incertidumbre inicial)
    filas = []
    for t in range(n):
        # ---- Prediccion (antes de ver y_t) ----
        z_prior = z                                # z_{t|t-1} = z_{t-1|t-1}
        P_prior = P + Q                            # P_{t|t-1} = P_{t-1|t-1} + Q
        # ---- Actualizacion (ya con y_t) ----
        C = np.array([1.0, x[t]])                  # C_t = [1, x_t]
        e = y[t] - C @ z_prior                     # innovacion: dato - prediccion
        S = C @ P_prior @ C + R                    # varianza de la innovacion (escalar)
        K = P_prior @ C / S                        # ganancia de Kalman (vector 2x1)
        z = z_prior + K * e                        # z_{t|t}: corrige con la innovacion
        P = (I - np.outer(K, C)) @ P_prior         # P_{t|t} = (I - K C) P_{t|t-1}
        filas.append({
            "alpha_prior": z_prior[0], "beta_prior": z_prior[1],
            "alpha_post": z[0], "beta_post": z[1],
            "P_alpha": P[0, 0], "P_beta": P[1, 1],
            "K_alpha": K[0], "K_beta": K[1],
            "e": e, "S": S,
        })
    return pd.DataFrame(filas)


def log_verosimilitud(e, S):
    """Log-verosimilitud gaussiana de las innovaciones: sum(-0.5*(log(2*pi*S) + e^2/S))."""
    e, S = np.asarray(e), np.asarray(S)
    return float(np.sum(-0.5 * (np.log(2 * np.pi * S) + e ** 2 / S)))


def calibrar_delta_R(y, x, mascara, z0, R_ols, deltas=DELTAS, n_R=10, delta_max=DELTA_MAX):
    """Busca en rejilla (delta, R) la pareja de mayor log-verosimilitud con delta <= delta_max.
    R se prueba en 10 valores logaritmicos entre R_ols/100 y R_ols: la R del OLS mide
    desviaciones de NIVEL del burn-in y sobreestima el ruido de un dia.
    'mascara' marca los dias que cuentan (solo train despues del burn-in).
    Regresa (tabla completa delta x R con columnas permitido/elegido, (delta, R) elegidos)."""
    R_grid = np.logspace(np.log10(R_ols / 100), np.log10(R_ols), n_R)   # rejilla de R
    filas = []
    for d in deltas:
        for R in R_grid:
            res = correr_filtro_pesos(y, x, d, z0, R)
            ll = log_verosimilitud(res["e"].values[mascara], res["S"].values[mascara])
            filas.append({"delta": d, "R": R, "log_verosimilitud": round(ll, 4)})
    tabla = pd.DataFrame(filas)
    tabla["permitido"] = tabla["delta"] <= delta_max              # region admisible
    mejor = tabla.loc[tabla["permitido"], "log_verosimilitud"].idxmax()   # max LL permitida
    tabla["max_LL_permitido"] = tabla.index == mejor   # evidencia; R final sale por momentos
    R_opt = float(tabla.loc[mejor, "R"])
    if R_opt in (R_grid.min(), R_grid.max()):                     # optimo en el borde
        print(f"!!! ADVERTENCIA: la R elegida ({R_opt:.3e}) esta en el borde de la rejilla")
    return tabla, (float(tabla.loc[mejor, "delta"]), R_opt)


# ---------------------------------------------------------------------------
# Calibracion de R por METODO DE MOMENTOS
# Las innovaciones del filtro tienen colas pesadas (curtosis en exceso ~5 con delta=1e-5).
# Con colas pesadas la verosimilitud gaussiana subestima R: prefiere ajustar muy bien los
# dias normales y "aguanta" los pocos errores grandes, asi el filtro reporta menos
# incertidumbre de la real (var(e/sqrt(S)) > 1). El metodo de momentos elige la R que hace
# var(e_t/sqrt(S_t)) = 1 en train, de modo que S_t refleje el tamano real de los errores.
# ---------------------------------------------------------------------------
def var_estandarizada(res, mascara):
    """Varianza de las innovaciones estandarizadas e/sqrt(S) en los dias de 'mascara'.
    Regresa NaN si hay menos de 2 dias."""
    u = (res["e"] / np.sqrt(res["S"])).values[mascara]
    return float(np.var(u, ddof=1)) if len(u) >= 2 else np.nan


def curtosis_exceso(u):
    """Curtosis en exceso (0 para una normal) de un array u."""
    u = np.asarray(u, dtype=float)
    m, v = u.mean(), u.var()
    return float(np.mean((u - m) ** 4) / v ** 2 - 3)


def calibrar_R_momentos(y, x, mascara, z0, R_ols, delta, tol=1e-3, max_iter=60):
    """Busca por biseccion (en escala log) la R con var(e/sqrt(S)) = 1 en 'mascara'.
    Rango: R_ols/1000 a R_ols. var(e/sqrt(S)) BAJA cuando R sube (S crece con R).
    Regresa la R encontrada."""
    def f(log_R):                                       # funcion cuyo cero buscamos
        res = correr_filtro_pesos(y, x, delta, z0, np.exp(log_R))
        return var_estandarizada(res, mascara) - 1.0

    lo, hi = np.log(R_ols / 1000), np.log(R_ols)        # extremos en escala log
    f_lo, f_hi = f(lo), f(hi)
    if f_lo < 0:                                        # ni con la R minima llega a 1
        print(f"!!! ADVERTENCIA delta={delta}: var < 1 aun con R = R_ols/1000 (borde)")
        return float(np.exp(lo))
    if f_hi > 0:                                        # ni con la R maxima baja a 1
        print(f"!!! ADVERTENCIA delta={delta}: var > 1 aun con R = R_ols (borde)")
        return float(np.exp(hi))
    for _ in range(max_iter):
        mid = (lo + hi) / 2                             # punto medio en log R
        f_mid = f(mid)
        if abs(f_mid) < tol:                            # var a menos de tol de 1
            break
        if f_mid > 0:                                   # var > 1 -> falta R -> subir
            lo = mid
        else:                                           # var < 1 -> sobra R -> bajar
            hi = mid
    return float(np.exp(mid))


def tabla_momentos(y, x, muestra, en_burn_in, z0, R_ols, deltas=(1e-6, 1e-5)):
    """Para cada delta: R por momentos (en train post burn-in), su log-verosimilitud en
    train, var(e/sqrt(S)) en train, test y val, y la curtosis en exceso en train."""
    muestra = np.asarray(muestra)
    m_train = (muestra == "train") & ~en_burn_in
    filas = []
    for d in deltas:
        R = calibrar_R_momentos(y, x, m_train, z0, R_ols, d)
        res = correr_filtro_pesos(y, x, d, z0, R)
        u_train = (res["e"] / np.sqrt(res["S"])).values[m_train]
        filas.append({
            "delta": d, "R_momentos": R,
            "LL": round(log_verosimilitud(res["e"].values[m_train], res["S"].values[m_train]), 4),
            "var_train": round(var_estandarizada(res, m_train), 4),
            "var_test": round(var_estandarizada(res, muestra == "test"), 4),
            "var_val": round(var_estandarizada(res, muestra == "val"), 4),
            "curtosis": round(curtosis_exceso(u_train), 4),
        })
    return pd.DataFrame(filas)
