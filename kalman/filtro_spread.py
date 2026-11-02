# kalman/filtro_spread.py
# Filtro 2: nivel (mu) y escala (sigma) del spread, para obtener un z-score causal.
#   Spread:       s_t = y_t - beta_{t|t-1} * x_t    (beta PRIOR del filtro 1; el nivel alpha
#                                                    lo absorbe este filtro como mu_t)
#   Transicion:   mu_t = mu_{t-1} + w_t,  w_t ~ N(0, Q2_t),  Q2_t = q * R2_{t-1}
#   Observacion:  s_t  = mu_t + v_t,      v_t ~ N(0, R2_t)   (R2_t se estima con EWMA)
#
# Por que Q2 proporcional a R2 (modelo de "nivel local"):
#   Si Q2 = q * R2, la ganancia de Kalman converge a un valor fijo K. En estado estable
#   P2_{t|t-1} = K/(1-K) * R2, y entonces K2 = P2_{t|t-1} / (P2_{t|t-1} + R2) = K.
#   Despejando, para obtener una ganancia objetivo K_obj hay que usar
#       q = K_obj^2 / (1 - K_obj).
#   Con ganancia fija, mu_{t|t} = mu_{t|t-1} + K*(s_t - mu_{t|t-1}) es un promedio
#   exponencial cuya memoria es ~1/K dias. Con K_obj = 1/L el filtro 2 "recuerda" lo
#   mismo que la ventana movil de L dias del modelo estatico (comparacion justa).
#   Lambda del EWMA de R2 = 1 - 1/L, por la misma razon: misma memoria para la escala.
import numpy as np
import pandas as pd


def parametros_memoria(L):
    """Regresa K_obj, q y lambda para una memoria de L dias."""
    K_obj = 1.0 / L                        # ganancia objetivo
    q = K_obj ** 2 / (1 - K_obj)           # Q2_t = q * R2_{t-1} da ganancia estable K_obj
    lam = 1.0 - 1.0 / L                    # EWMA de R2 con la misma memoria
    return K_obj, q, lam


def correr_filtro_spread(s, s_init, L):
    """Corre el filtro del spread dia por dia.
    Recibe el spread s (todas las fechas), s_init (spread del burn-in calculado con el beta
    OLS FIJO, solo para inicializar) y la memoria L en dias.
    En cada t: 1) predice mu y P2, 2) calcula z_t con informacion hasta t-1,
    3) actualiza mu, P2 y R2 con el dato s_t.
    Regresa (DataFrame por dia, diccionario de parametros)."""
    s = np.asarray(s, dtype=float)
    K_obj, q, lam = parametros_memoria(L)
    mu = float(np.mean(s_init))                     # mu_0: media del spread del burn-in
    R2 = float(np.var(s_init, ddof=1))              # R2_0: varianza del spread del burn-in
    # P2_0 = K_obj * R2_0: es la P2 posterior del estado estable (P2_{t|t} = K * R2),
    # asi el filtro arranca ya en equilibrio y no con una ganancia artificialmente alta.
    P2 = K_obj * R2
    params = {"mu_0": mu, "R2_0": R2, "P2_0": P2, "q": q, "K_obj": K_obj,
              "L": L, "lambda": lam}
    filas = []
    for t in range(len(s)):
        # ---- Prediccion ----
        mu_prior = mu                               # mu_{t|t-1} = mu_{t-1|t-1}
        Q2 = q * R2                                 # Q2_t = q * R2_{t-1}
        P2_prior = P2 + Q2                          # P2_{t|t-1} = P2_{t-1|t-1} + Q2_t
        S2 = P2_prior + R2                          # S2_t = P2_{t|t-1} + R2_{t-1}
        # ---- z-score causal: se calcula ANTES de usar s_t para actualizar ----
        e = s[t] - mu_prior                         # innovacion del spread
        z = e / np.sqrt(S2)                         # z_t = (s_t - mu_{t|t-1}) / sqrt(S2_t)
        # ---- Actualizacion con s_t ----
        K2 = P2_prior / S2                          # ganancia de Kalman (escalar)
        mu = mu_prior + K2 * e                      # mu_{t|t}
        P2 = (1 - K2) * P2_prior                    # P2_{t|t}
        # R2_t: EWMA de las innovaciones al cuadrado. e^2 mide S2 = P2_prior + R2, asi que
        # sobreestima R2 por P2_prior ~ K_obj/(1-K_obj)*R2 (~1.2% con L=83): despreciable.
        R2 = lam * R2 + (1 - lam) * e ** 2
        filas.append({"s": s[t], "mu_prior": mu_prior, "mu_post": mu,
                      "sigma": np.sqrt(S2), "z": z, "K2": K2})
    return pd.DataFrame(filas), params
