# strategy/generar_senales.py
# Corre Filtro 1 (pesos) -> Filtro 2 (spread) -> reglas PFA, y la linea base estatica.
# Ejecutar desde la raiz:  python -m strategy.generar_senales
#
# ===================== SALIDAS (las lee el integrante C) =====================
# artifacts/senales_kalman.csv y artifacts/senales_estatico.csv  (una fila por dia)
#   fecha          dia de negociacion (indice)
#   muestra        train / test / val
#   en_burn_in     True en los primeros 126 dias del train (no se opera, d=0)
#   AMT, SBAC      precios de cierre ajustados (en nivel, no log)
#   beta           cobertura usada: beta_{t|t-1} del Kalman (prior) o 0.8631 fijo.
#                  Spread = log(AMT) - beta*log(SBAC). Por 1 unidad de AMT, -beta de SBAC
#                  (en terminos de log-precio / valor nocional).
#   s              spread del dia t. Kalman: s = log(AMT) - beta*(log(SBAC) - x_media_burnin)
#                  (x centrado, ver kalman_params.json); estatico: log(AMT) - 0.8631*log(SBAC).
#                  Centrar solo mueve el nivel del spread (lo absorbe mu); el hedge sigue siendo beta.
#   mu             nivel esperado del spread con info hasta t-1
#   sigma          escala del spread con info hasta t-1
#   z              z-score causal = (s - mu) / sigma
#   d              senal: +1 largo spread (largo AMT, corto SBAC), -1 corto spread, 0 flat.
#                  Se calcula con el cierre de t y SE EJECUTA EN t+1 (lo hace el backtest).
#   h              dias desde la entrada (el dia de cierre trae la duracion de la operacion)
#   motivo_cierre  "" | salida | stop_z | stop_tiempo  (solo el dia que se cierra)
# Ademas:
#   artifacts/kalman_filtro_pesos.csv   salida completa del Filtro 1
#   artifacts/kalman_filtro_spread.csv  salida completa del Filtro 2
#   artifacts/kalman_calibracion.csv    rejilla delta x R vs log-verosimilitud (train post
#                                       burn-in). Evidencia de lo probado; NO es la R usada.
#   artifacts/kalman_calibracion_momentos.csv  R por metodo de momentos para delta 1e-6 y
#                                       1e-5: R_momentos, LL, var en train/test/val, curtosis
#   artifacts/kalman_params.json        delta, R (momentos), Q, R_ols, z0, x_media_burnin,
#                                       burn_in, L, K_obj, q, lambda, mu_0, R2_0, P2_0, theta
# =============================================================================
import json
import os
import numpy as np
import pandas as pd

from kalman.filtro_pesos import (BURN_IN, DELTA_MAX, centrar_x, inicializar,
                                 correr_filtro_pesos, calibrar_delta_R, tabla_momentos)
from kalman.filtro_spread import correr_filtro_spread
from strategy.reglas import THETA_INICIAL, aplicar_reglas
from strategy.estatico import zscore_estatico

ART = "artifacts"
MUESTRAS = ["train", "test", "val"]
COLUMNAS = ["muestra", "en_burn_in", "AMT", "SBAC", "beta", "s", "mu", "sigma", "z",
            "d", "h", "motivo_cierre"]
# delta del Filtro 1. Por defecto 1e-6: beta mas estable y menos rebalanceo. La eleccion
# final entre 1e-6 y 1e-5 se hizo en el grid search por Sharpe en train
# (strategy/grid_search.py -> artifacts/theta_optimo.json): delta* = 1e-6.
DELTA_FINAL = 1e-6
DELTAS_MOMENTOS = (1e-6, 1e-5)


def cargar_precios(cols=("AMT", "SBAC")):
    """Concatena train, test y val (en orden) con una columna 'muestra'."""
    partes = []
    for m in MUESTRAS:
        px = pd.read_csv(f"{ART}/prices_{m}.csv", index_col=0, parse_dates=True)
        px = px[list(cols)].copy()
        px["muestra"] = m
        partes.append(px)
    return pd.concat(partes)


def correr_pipeline(px, beta_fijo, vida_media, theta=THETA_INICIAL, delta=DELTA_FINAL,
                    theta_estatico=None, L_estatico=None):
    """Corre todo sobre px (precios + columna muestra), en orden cronologico.
    R e inicializacion se fijan SOLO con filas de train; delta viene de afuera
    (DELTA_FINAL o el que pruebe el grid search). theta_estatico y L_estatico permiten
    usar otra politica/ventana en el estatico (None = mismo theta y L = 83).
    Regresa un diccionario con los DataFrames y los parametros."""
    y = np.log(px["AMT"]).values                     # observacion: log AMT
    x = np.log(px["SBAC"]).values                    # regresor: log SBAC
    en_burn_in = np.arange(len(px)) < BURN_IN        # primeros 126 dias
    es_train = (px["muestra"] == "train").values
    L = round(2 * vida_media)                        # memoria comun: filtro 2 y estatico

    # ---- Filtro 1 con x centrado (media del burn-in): C_t = [1, x_c_t] ----
    x_c, x_media = centrar_x(x, BURN_IN)
    z0, R_ols = inicializar(y, x_c, BURN_IN)
    # Rejilla de maxima verosimilitud: se guarda como evidencia, pero no fija R
    calib, (delta_ml, R_ml) = calibrar_delta_R(y, x_c, es_train & ~en_burn_in, z0, R_ols)
    # R por metodo de momentos (var(e/sqrt(S)) = 1 en train post burn-in)
    momentos = tabla_momentos(y, x_c, px["muestra"].values, en_burn_in, z0, R_ols,
                              DELTAS_MOMENTOS)
    R = float(momentos.loc[momentos["delta"] == delta, "R_momentos"].iloc[0])
    f1 = correr_filtro_pesos(y, x_c, delta, z0, R)
    f1.index = px.index

    # ---- Filtro 2: spread con beta PRIOR (sin mirar el dato de hoy) ----
    s = y - f1["beta_prior"].values * x_c
    # Para inicializar mu_0 y R2_0 se usa el beta OLS FIJO del burn-in: el beta_prior se
    # mueve mucho en esos dias (P0 = I) e inflaria la varianza inicial.
    s_init = y[:BURN_IN] - z0[1] * x_c[:BURN_IN]
    f2, params2 = correr_filtro_spread(s, s_init, L)
    f2.index = px.index

    # ---- Reglas sobre el z del Kalman (no se opera en burn-in) ----
    reglas_k = aplicar_reglas(f2["z"].values, operable=~en_burn_in, **theta)
    kalman = pd.DataFrame({
        "muestra": px["muestra"].values, "en_burn_in": en_burn_in,
        "AMT": px["AMT"].values, "SBAC": px["SBAC"].values,
        "beta": f1["beta_prior"].values, "s": f2["s"].values,
        "mu": f2["mu_prior"].values, "sigma": f2["sigma"].values, "z": f2["z"].values,
    }, index=px.index)
    kalman = pd.concat([kalman, reglas_k.set_index(px.index)], axis=1)

    # ---- Linea base estatica: mismo theta, misma ventana operable ----
    theta_e = theta if theta_estatico is None else theta_estatico
    L_e = L if L_estatico is None else L_estatico
    est = zscore_estatico(np.log(px["AMT"]), np.log(px["SBAC"]), beta_fijo, L_e)
    reglas_e = aplicar_reglas(est["z"].values, operable=~en_burn_in, **theta_e)
    estatico = pd.concat([px[["muestra", "AMT", "SBAC"]], est,
                          reglas_e.set_index(px.index)], axis=1)
    estatico.insert(1, "en_burn_in", en_burn_in)

    params = {"delta": delta, "R": R, "metodo_R": "momentos",
              "Q": (delta / (1 - delta)) * np.eye(2), "R_ols": R_ols,
              "deltas_momentos": list(DELTAS_MOMENTOS), "delta_max_ml": DELTA_MAX,
              "z0": z0, "x_media_burnin": x_media, "burn_in": BURN_IN, "theta": theta,
              "theta_estatico": theta_e, "L_estatico": L_e,
              "vida_media": vida_media, **params2}      # params2 trae L, K_obj, q, lambda...
    f1.insert(0, "muestra", px["muestra"].values)
    f2.insert(0, "muestra", px["muestra"].values)
    return {"kalman": kalman[COLUMNAS], "estatico": estatico[COLUMNAS],
            "filtro_pesos": f1, "filtro_spread": f2, "calibracion": calib,
            "calibracion_momentos": momentos, "params": params}


def main():
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    beta_fijo = -chosen["vector"][1]                 # vector [1, -0.8631] -> beta = 0.8631
    px = cargar_precios()
    # Si ya se corrio el grid search, usar el theta* congelado de cada metodo
    kw = {}
    if os.path.exists(f"{ART}/theta_optimo.json"):
        opt = json.load(open(f"{ART}/theta_optimo.json"))
        claves = ["z_in", "z_out", "z_stop", "h_max"]
        kw = {"theta": {c: opt["kalman"][c] for c in claves},
              "theta_estatico": {c: opt["estatico"][c] for c in claves},
              "L_estatico": opt["estatico"]["L"]}
        if opt["kalman"]["delta"] != DELTA_FINAL:
            print(f"!!! ADVERTENCIA: delta* del grid ({opt['kalman']['delta']}) "
                  f"!= DELTA_FINAL ({DELTA_FINAL})")
    res = correr_pipeline(px, beta_fijo, chosen["vida_media"], **kw)

    for nombre in ["kalman", "estatico"]:
        res[nombre].to_csv(f"{ART}/senales_{nombre}.csv", index_label="fecha")
    res["filtro_pesos"].to_csv(f"{ART}/kalman_filtro_pesos.csv", index_label="fecha")
    res["filtro_spread"].to_csv(f"{ART}/kalman_filtro_spread.csv", index_label="fecha")
    res["calibracion"].to_csv(f"{ART}/kalman_calibracion.csv", index=False)
    res["calibracion_momentos"].to_csv(f"{ART}/kalman_calibracion_momentos.csv", index=False)
    p = res["params"]
    p_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    with open(f"{ART}/kalman_params.json", "w") as f:
        json.dump(p_json, f, indent=2)

    calib = res["calibracion"]
    print(f"=== Max. verosimilitud (evidencia): 5 mejores con delta <= {DELTA_MAX} ===")
    print(calib[calib["permitido"]].nlargest(5, "log_verosimilitud").to_string(index=False))
    print("\n=== R por metodo de momentos ===")
    print(res["calibracion_momentos"].to_string(index=False))
    print(f"\ndelta={p['delta']}  R={p['R']:.3e} (momentos; R_ols={p['R_ols']:.3e})  "
          f"L={p['L']}  K_obj={p['K_obj']:.4f}  q={p['q']:.3e}  lambda={p['lambda']:.4f}  "
          f"R2_0={p['R2_0']:.3e}")
    for nombre in ["kalman", "estatico"]:
        df = res[nombre]
        print(f"\n[{nombre}] entradas={int(((df['d'] != 0) & (df['d'].shift(1, fill_value=0) == 0)).sum())}"
              f"  cierres={df['motivo_cierre'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
