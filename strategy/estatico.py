# strategy/estatico.py
# Linea base SIN Kalman: spread con el vector fijo de Johansen y z-score movil.
#   s_t = log(AMT_t) - beta_fijo * log(SBAC_t),   beta_fijo = 0.8631
#   z_t = (s_t - media_{t-1}) / desv_{t-1}, con media y desv. moviles de L dias que
#   terminan en t-1 (shift(1)), asi z_t no usa el propio s_t para su media/escala.
import numpy as np
import pandas as pd


def zscore_estatico(y, x, beta_fijo, L):
    """Recibe Series y = log(AMT), x = log(SBAC), el beta fijo y la ventana L.
    Regresa DataFrame con beta, s, mu, sigma y z."""
    s = y - beta_fijo * x                            # spread con pesos fijos
    mu = s.rolling(L).mean().shift(1)                # media de los L dias hasta t-1
    sigma = s.rolling(L).std().shift(1)              # desviacion de los L dias hasta t-1
    z = (s - mu) / sigma                             # z-score causal
    return pd.DataFrame({"beta": beta_fijo, "s": s, "mu": mu, "sigma": sigma, "z": z},
                        index=y.index)
