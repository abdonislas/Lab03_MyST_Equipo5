# cointegration/diagnostico_quiebre.py
# Diagnostico pedido por el profe: en test, ¿se rompio la cointegracion o el modelo esta
# sobreajustado a train? Se prueba la COINTEGRACION por particion (no la estrategia) y se
# repite todo el procedimiento en una re-particion de train. Solo diagnostico: no cambia
# ningun resultado oficial; todo se escribe en artifacts/diagnostico/.
# Ejecutar desde la raiz:  python -m cointegration.diagnostico_quiebre
import itertools
import json
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from cointegration.screening import normalizar_vector, vida_media
from kalman.filtro_pesos import BURN_IN
from strategy.generar_senales import cargar_precios, correr_pipeline
from strategy.estatico import zscore_estatico
from strategy.grid_search import DELTAS, METRICAS, evaluar_theta, elegir, metricas_por_muestra

ART = "artifacts"
OUT = f"{ART}/diagnostico"
PVAL = 0.05
VENTANA, PASO = 252, 5          # ADF movil: 1 anio, paso semanal
MIN_TRAMO = 6                   # tramo "sostenido" = al menos 6 ventanas seguidas (~30 dias)
Z_IN, Z_OUT, Z_STOP = [1.5, 2.0, 2.5], [0.0, 0.25, 0.5], [3.0, 3.5, 4.0]


# =========================== Paso 1: cointegracion ===========================
def prueba_cointegracion(logpx, w_fijo):
    """Pruebas de cointegracion sobre log-precios (columnas AMT, SBAC) de una particion.
    w_fijo: vector de cointegracion fijo [w_AMT, w_SBAC] para el ADF del spread.
    Regresa un diccionario con EG, Johansen y ADF/vida media del spread."""
    a, s = logpx["AMT"], logpx["SBAC"]
    # a) Engle-Granger en ambos ordenes
    p_as = coint(a, s)[1]                               # AMT sobre SBAC
    p_sa = coint(s, a)[1]                               # SBAC sobre AMT
    # b) Johansen: traza (lr1) y max-eigenvalor (lr2) para r=0 y r<=1, criticos al 95%
    jo = coint_johansen(logpx[["AMT", "SBAC"]], det_order=0, k_ar_diff=1)
    rank_tr = int((jo.lr1 > jo.cvt[:, 1]).sum())
    rank_me = int((jo.lr2 > jo.cvm[:, 1]).sum())
    # c) Spread con el vector fijo: ADF y vida media
    spread = logpx[["AMT", "SBAC"]].values @ np.asarray(w_fijo, float)
    return {
        "fecha_inicio": logpx.index[0].date(), "fecha_fin": logpx.index[-1].date(),
        "n_obs": len(logpx),
        "EG_p_AMT_SBAC": round(p_as, 4), "EG_p_SBAC_AMT": round(p_sa, 4),
        "traza_r0": round(jo.lr1[0], 3), "cv95_traza_r0": round(jo.cvt[0, 1], 3),
        "traza_r1": round(jo.lr1[1], 3), "cv95_traza_r1": round(jo.cvt[1, 1], 3),
        "maxeig_r0": round(jo.lr2[0], 3), "cv95_maxeig_r0": round(jo.cvm[0, 1], 3),
        "maxeig_r1": round(jo.lr2[1], 3), "cv95_maxeig_r1": round(jo.cvm[1, 1], 3),
        "rank_traza": rank_tr, "rank_maxeig": rank_me,
        "ADF_spread_p": round(adfuller(spread, autolag="AIC")[1], 4),
        "vida_media": round(vida_media(spread), 1),
    }


def tabla_por_particion(logpx, muestra, w_fijo, etiquetas):
    """Corre prueba_cointegracion en cada particion. etiquetas: {nombre_interno: nombre_salida}."""
    filas = []
    for interno, salida in etiquetas.items():
        filas.append({"particion": salida, **prueba_cointegracion(logpx[muestra == interno], w_fijo)})
    return pd.DataFrame(filas)


# =========================== Paso 2: ADF movil ===========================
def adf_movil(logpx, muestra, w_fijo, ventana=VENTANA, paso=PASO):
    """ADF y vida media del spread (vector fijo) en ventanas moviles de 'ventana' dias."""
    spread = pd.Series(logpx[["AMT", "SBAC"]].values @ np.asarray(w_fijo, float), index=logpx.index)
    filas = []
    for fin in range(ventana, len(spread) + 1, paso):
        s = spread.iloc[fin - ventana:fin]
        filas.append({"fecha_fin": s.index[-1].date(), "muestra": muestra[fin - 1],
                      "ADF_p": round(adfuller(s.values, autolag="AIC")[1], 4),
                      "vida_media": round(vida_media(s.values), 1)})
    return pd.DataFrame(filas)


def tramos_sostenidos(tabla, min_ventanas=MIN_TRAMO):
    """Agrupa ventanas consecutivas del mismo lado de 0.05 y regresa los tramos largos."""
    lado = np.where(tabla["ADF_p"] < PVAL, "p<0.05 (cointegrado)", "p>=0.05 (sin evidencia)")
    grupo = (pd.Series(lado) != pd.Series(lado).shift()).cumsum()   # id de cada tramo
    filas = []
    for _, g in tabla.groupby(grupo.values):
        if len(g) >= min_ventanas:
            filas.append({"estado": lado[g.index[0]], "desde": g["fecha_fin"].iloc[0],
                          "hasta": g["fecha_fin"].iloc[-1], "ventanas": len(g),
                          "ADF_p_mediana": round(g["ADF_p"].median(), 4)})
    return pd.DataFrame(filas)


# =========================== Paso 3: re-particion de train ===========================
def repartir(px_train):
    """Parte el train original 60/20/20 en orden cronologico. Las etiquetas internas son
    train/test/val para reutilizar el pipeline del proyecto tal cual."""
    n = len(px_train)
    i1, i2 = int(n * 0.60), int(n * 0.80)
    px = px_train[["AMT", "SBAC"]].copy()
    px["muestra"] = ["train"] * i1 + ["test"] * (i2 - i1) + ["val"] * (n - i2)
    return px


def vector_johansen(logpx):
    """Vector de Johansen (1er eigenvector) normalizado por la pata de mayor |valor|.
    Regresa (vector, pata, beta equivalente en la forma log AMT - beta*log SBAC)."""
    jo = coint_johansen(logpx[["AMT", "SBAC"]], det_order=0, k_ar_diff=1)
    w, leg = normalizar_vector(jo.evec[:, 0])
    beta = -w[1] / w[0]                                  # spread proporcional a y - beta*x
    return w, ["AMT", "SBAC"][leg], float(beta)


def rejilla(vida):
    """Misma rejilla del proyecto: z fijos y h_max en [1, 2, 3] vidas medias."""
    multiplos = [round(k * vida) for k in (1, 2, 3)]
    claves = ["z_in", "z_out", "z_stop", "h_max"]
    thetas = [dict(zip(claves, v)) for v in itertools.product(Z_IN, Z_OUT, Z_STOP, multiplos)]
    return thetas, multiplos


def grid_y_metricas(px, beta_fijo, vida):
    """Mismo procedimiento que strategy/grid_search.py, calibrando solo con la etiqueta train.
    Regresa (theta optimo por metodo, metricas por sub-muestra, advertencias)."""
    muestra = px["muestra"].values
    en_burn_in = np.arange(len(px)) < BURN_IN
    m_train = (muestra == "train") & ~en_burn_in
    thetas, multiplos = rejilla(vida)

    # Kalman: filtros una vez por delta (R por momentos, L = 2 vidas medias)
    filas_k, series_k = [], {}
    for delta in DELTAS:
        res = correr_pipeline(px, beta_fijo, vida, delta=delta)
        k = res["kalman"]
        base = k[["AMT", "SBAC", "beta"]]
        series_k[delta] = (base, k["z"].values, res["params"])
        for th in thetas:
            filas_k.append({"delta": delta, "R": res["params"]["R"], **th,
                            **evaluar_theta(base, k["z"].values, th, ~en_burn_in, m_train)})
    grid_k = pd.DataFrame(filas_k)

    # Estatico: z una vez por ventana L
    filas_e, series_e = [], {}
    for L in multiplos:
        est = zscore_estatico(np.log(px["AMT"]), np.log(px["SBAC"]), beta_fijo, L)
        base = pd.concat([px[["AMT", "SBAC"]], est[["beta"]]], axis=1)
        series_e[L] = (base, est["z"].values)
        for th in thetas:
            filas_e.append({"L": L, **th, **evaluar_theta(base, est["z"].values, th, ~en_burn_in, m_train)})
    grid_e = pd.DataFrame(filas_e)

    mejor_k, min_k, aviso_k = elegir(grid_k)
    mejor_e, min_e, aviso_e = elegir(grid_e)
    claves = ["z_in", "z_out", "z_stop", "h_max"]
    th_k = {c: (int(mejor_k[c]) if c == "h_max" else float(mejor_k[c])) for c in claves}
    th_e = {c: (int(mejor_e[c]) if c == "h_max" else float(mejor_e[c])) for c in claves}
    d_opt, L_opt = float(mejor_k["delta"]), int(mejor_e["L"])
    p_k = series_k[d_opt][2]

    optimo = {
        "kalman": {**th_k, "delta": d_opt, "R": float(p_k["R"]), "L_filtro2": int(p_k["L"]),
                   "sharpe_sub_train": float(mejor_k["sharpe"]), "n_ops_sub_train": int(mejor_k["n_ops"]),
                   "min_ops": min_k, "evaluadas": len(grid_k),
                   "factibles_10": int((grid_k["n_ops"] >= 10).sum())},
        "estatico": {**th_e, "L": L_opt, "sharpe_sub_train": float(mejor_e["sharpe"]),
                     "n_ops_sub_train": int(mejor_e["n_ops"]), "min_ops": min_e,
                     "evaluadas": len(grid_e), "factibles_10": int((grid_e["n_ops"] >= 10).sum())},
    }
    avisos = [f"kalman: {aviso_k}"] * bool(aviso_k) + [f"estatico: {aviso_e}"] * bool(aviso_e)

    # Theta congelado: cada sub-muestra arranca plana con 1,000,000
    nombres = {"train": "sub_train", "test": "sub_test", "val": "sub_val"}
    base_k, z_k, _ = series_k[d_opt]
    base_e, z_e = series_e[L_opt]
    filas = [{"metodo": "kalman", **f} for f in metricas_por_muestra(base_k, z_k, th_k, muestra, en_burn_in)]
    filas += [{"metodo": "estatico", **f} for f in metricas_por_muestra(base_e, z_e, th_e, muestra, en_burn_in)]
    met = pd.DataFrame(filas)
    met["muestra"] = met["muestra"].map(nombres)
    return optimo, met, avisos


# =========================== main ===========================
def a_json(v):
    """Convierte tipos de numpy/fechas para json.dump."""
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return str(v)


def main():
    os.makedirs(OUT, exist_ok=True)
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    w_train = np.asarray(chosen["vector"], float)       # [1, -0.8631]
    px = cargar_precios()
    logpx = np.log(px[["AMT", "SBAC"]])
    muestra = px["muestra"].values
    avisos = []

    # ---- Paso 1 ----
    p1 = tabla_por_particion(logpx, muestra, w_train, {"train": "train", "test": "test", "val": "val"})
    p1.to_csv(f"{OUT}/cointegracion_por_particion.csv", index=False)
    oos = pd.read_csv(f"{ART}/oos_check.csv").set_index("muestra")
    for _, f in p1.iterrows():                           # deben coincidir con oos_check.csv
        dif = abs(f["ADF_spread_p"] - oos.loc[f["particion"], "ADF_spread_p"])
        if dif > 1e-3:
            avisos.append(f"paso 1: ADF de {f['particion']} difiere de oos_check.csv en {dif:.4f}")

    # ---- Paso 2 ----
    p2 = adf_movil(logpx, muestra, w_train)
    p2.to_csv(f"{OUT}/adf_movil.csv", index=False)
    tramos = tramos_sostenidos(p2)

    # ---- Paso 3 ----
    px_sub = repartir(pd.read_csv(f"{ART}/prices_train.csv", index_col=0, parse_dates=True))
    log_sub = np.log(px_sub[["AMT", "SBAC"]])
    m_sub = px_sub["muestra"].values
    w_sub, pata, beta_sub = vector_johansen(log_sub[m_sub == "train"])
    hl_sub = vida_media(log_sub[m_sub == "train"].values @ w_sub)
    if not np.isfinite(hl_sub):
        avisos.append("paso 3: vida media de sub_train no finita; se usa 41.3")
        hl_sub = 41.3
    if beta_sub <= 0:
        avisos.append(f"paso 3: beta de sub_train = {beta_sub:.4f} <= 0 (no es cobertura de par)")
    p3 = tabla_por_particion(log_sub, m_sub, w_sub,
                             {"train": "sub_train", "test": "sub_test", "val": "sub_val"})
    p3.insert(1, "vector_sub_train", str(np.round(w_sub, 4).tolist()))
    p3.insert(2, "pata", pata)
    p3.to_csv(f"{OUT}/resplit_cointegracion.csv", index=False)

    optimo, met, avisos3 = grid_y_metricas(px_sub, beta_sub, hl_sub)
    avisos += avisos3
    optimo.update({"vector_sub_train": np.round(w_sub, 4).tolist(), "pata_normalizacion": pata,
                   "beta_sub_train": round(beta_sub, 4), "vida_media_sub_train": round(float(hl_sub), 1),
                   "fechas": {n: [str(f["fecha_inicio"]), str(f["fecha_fin"]), int(f["n_obs"])]
                              for n, f in p3.set_index("particion").iterrows()},
                   "advertencias": avisos})
    with open(f"{OUT}/resplit_theta.json", "w") as fh:
        json.dump(optimo, fh, indent=2, default=a_json)
    met.to_csv(f"{OUT}/resplit_metricas.csv", index=False)

    # ---- Reporte ----
    pd.set_option("display.width", 200)
    print("=== Paso 1: cointegracion por particion (vector fijo de train) ===")
    print(p1.to_string(index=False))
    print("\n=== Paso 2: tramos sostenidos del ADF movil (>= 6 ventanas) ===")
    print(tramos.to_string(index=False))
    print(f"\n=== Paso 3: re-particion. Vector sub_train {np.round(w_sub, 4).tolist()} (pata {pata}), "
          f"beta={beta_sub:.4f}, vida media={hl_sub:.1f} ===")
    print(p3.to_string(index=False))
    print(json.dumps({k: optimo[k] for k in ["kalman", "estatico"]}, indent=2, default=a_json))
    print(met.round(4).to_string(index=False))
    for a in avisos:
        print(f"!!! ADVERTENCIA {a}")


if __name__ == "__main__":
    main()
