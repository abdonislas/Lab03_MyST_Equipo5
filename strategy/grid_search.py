# strategy/grid_search.py
# Paso 6 de Powell: buscar theta = (z_in, z_out, z_stop, h_max) en TRAIN y congelarlo.
#   Kalman:   ademas delta en {1e-6, 1e-5} (cada uno con su R de momentos; L del filtro 2 = 83)
#   Estatico: ademas L (ventana del z-score) en round([1, 2, 3] * vida_media)
# Objetivo: Sharpe neto en train (despues del burn-in) con al menos 10 operaciones.
# Ejecutar desde la raiz:  python -m strategy.grid_search
import itertools
import json
import numpy as np
import pandas as pd

from kalman.filtro_pesos import BURN_IN
from strategy.generar_senales import ART, cargar_precios, correr_pipeline
from strategy.reglas import aplicar_reglas
from strategy.estatico import zscore_estatico
from strategy.evaluador import evaluar

VIDA_MEDIA = 41.3
MULTIPLOS = [round(k * VIDA_MEDIA) for k in (1, 2, 3)]      # [41, 83, 124]
REJILLA_THETA = {
    "z_in": [1.5, 2.0, 2.5],
    "z_out": [0.0, 0.25, 0.5],
    "z_stop": [3.0, 3.5, 4.0],
    "h_max": MULTIPLOS,
}
DELTAS = [1e-6, 1e-5]
LS_ESTATICO = MULTIPLOS
MIN_OPS, MIN_OPS_RESPALDO = 10, 5
METRICAS = ["sharpe", "retorno_total", "max_drawdown", "n_ops", "win_rate",
            "comisiones", "prestamo"]


def todas_las_thetas():
    """Genera los 81 diccionarios theta de la rejilla."""
    claves = list(REJILLA_THETA)
    for valores in itertools.product(*REJILLA_THETA.values()):
        yield dict(zip(claves, valores))


def evaluar_theta(base, z, theta, operable, mascara):
    """Aplica las reglas con theta sobre z y evalua el P&L en las filas de 'mascara'.
    base: DataFrame con AMT, SBAC y beta. Regresa las metricas."""
    reglas = aplicar_reglas(z, operable=operable, **theta)
    df = base.assign(d=reglas["d"].values)[mascara]
    res = evaluar(df)
    return {k: res[k] for k in METRICAS}


def elegir(grid):
    """Mejor Sharpe con >= 10 operaciones; si no hay, con >= 5 (advertencia).
    Regresa (fila elegida, minimo de operaciones usado, advertencia)."""
    for minimo in (MIN_OPS, MIN_OPS_RESPALDO):
        factibles = grid[grid["n_ops"] >= minimo]
        if len(factibles):
            aviso = "" if minimo == MIN_OPS else \
                f"ninguna configuracion llega a {MIN_OPS} operaciones; se uso minimo {minimo}"
            return factibles.loc[factibles["sharpe"].idxmax()], minimo, aviso
    aviso = f"ninguna configuracion llega a {MIN_OPS_RESPALDO} operaciones; se eligio el mejor Sharpe"
    return grid.loc[grid["sharpe"].idxmax()], 0, aviso


def metricas_por_muestra(base, z, theta, muestra, en_burn_in):
    """Evalua theta* en train, test y val por separado. El z ya es continuo (usa la
    historia previa); cada muestra arranca PLANA (reglas sin posicion ni bloqueo heredado)
    y con 1,000,000. Train excluye el burn-in."""
    filas = []
    for m in ["train", "test", "val"]:
        mascara = (muestra == m) & ~en_burn_in
        z_m = np.asarray(z)[mascara]
        reglas = aplicar_reglas(z_m, **theta)                    # estado inicial plano
        df = base[mascara].assign(d=reglas["d"].values)
        res = evaluar(df)
        filas.append({"muestra": m, **{k: res[k] for k in METRICAS}})
    return filas


def main():
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    beta_fijo, vida_media = -chosen["vector"][1], chosen["vida_media"]
    px = cargar_precios()
    muestra = px["muestra"].values
    en_burn_in = np.arange(len(px)) < BURN_IN
    mascara_train = (muestra == "train") & ~en_burn_in

    # ---------------- Kalman: filtros una vez por delta, reglas por theta ----------------
    filas_k, series_k = [], {}
    for delta in DELTAS:
        res = correr_pipeline(px, beta_fijo, vida_media, delta=delta)
        k = res["kalman"]
        base = k[["AMT", "SBAC", "beta"]]                   # beta = beta_prior del filtro 1
        series_k[delta] = (base, k["z"].values, res["params"]["R"])
        for theta in todas_las_thetas():
            met = evaluar_theta(base, k["z"].values, theta, ~en_burn_in, mascara_train)
            filas_k.append({"delta": delta, "R": res["params"]["R"], **theta, **met})
    grid_k = pd.DataFrame(filas_k)
    grid_k["factible"] = grid_k["n_ops"] >= MIN_OPS

    # ---------------- Estatico: z una vez por L, reglas por theta ----------------
    filas_e, series_e = [], {}
    log_a, log_s = np.log(px["AMT"]), np.log(px["SBAC"])
    for L in LS_ESTATICO:
        est = zscore_estatico(log_a, log_s, beta_fijo, L)
        base = pd.concat([px[["AMT", "SBAC"]], est[["beta"]]], axis=1)
        series_e[L] = (base, est["z"].values)
        for theta in todas_las_thetas():
            met = evaluar_theta(base, est["z"].values, theta, ~en_burn_in, mascara_train)
            filas_e.append({"L": L, **theta, **met})
    grid_e = pd.DataFrame(filas_e)
    grid_e["factible"] = grid_e["n_ops"] >= MIN_OPS

    grid_k.to_csv(f"{ART}/grid_kalman.csv", index=False)
    grid_e.to_csv(f"{ART}/grid_estatico.csv", index=False)

    # ---------------- Elegir y congelar theta* ----------------
    mejor_k, min_k, aviso_k = elegir(grid_k)
    mejor_e, min_e, aviso_e = elegir(grid_e)
    theta_k = {c: (int(mejor_k[c]) if c == "h_max" else float(mejor_k[c])) for c in REJILLA_THETA}
    theta_e = {c: (int(mejor_e[c]) if c == "h_max" else float(mejor_e[c])) for c in REJILLA_THETA}
    delta_opt, L_opt = float(mejor_k["delta"]), int(mejor_e["L"])
    advertencias = []
    if aviso_k:
        advertencias.append(f"kalman: {aviso_k}")
    if aviso_e:
        advertencias.append(f"estatico: {aviso_e}")
    optimo = {
        "kalman": {**theta_k, "delta": delta_opt, "R": float(mejor_k["R"]), "L_filtro2": 83,
                   "sharpe_train": float(mejor_k["sharpe"]), "n_ops_train": int(mejor_k["n_ops"]),
                   "min_ops": min_k},
        "estatico": {**theta_e, "L": L_opt, "sharpe_train": float(mejor_e["sharpe"]),
                     "n_ops_train": int(mejor_e["n_ops"]), "min_ops": min_e},
        "advertencias": advertencias,
    }
    with open(f"{ART}/theta_optimo.json", "w") as f:
        json.dump(optimo, f, indent=2)

    # ---------------- Metricas por muestra con theta* ----------------
    base_k, z_k, _ = series_k[delta_opt]
    base_e, z_e = series_e[L_opt]
    filas = [{"metodo": "kalman", **f} for f in metricas_por_muestra(base_k, z_k, theta_k, muestra, en_burn_in)]
    filas += [{"metodo": "estatico", **f} for f in metricas_por_muestra(base_e, z_e, theta_e, muestra, en_burn_in)]
    met = pd.DataFrame(filas)
    met.to_csv(f"{ART}/metricas_preliminares.csv", index=False)

    # ---------------- Reporte ----------------
    for nombre, grid in [("KALMAN", grid_k), ("ESTATICO", grid_e)]:
        print(f"\n=== {nombre}: {len(grid)} evaluadas, {int(grid['factible'].sum())} factibles "
              f"(>= {MIN_OPS} ops) ===")
        print(grid[grid["factible"]].nlargest(5, "sharpe").round(4).to_string(index=False))
    print("\n=== theta* ===");  print(json.dumps(optimo, indent=2))
    print("\n=== Metricas por muestra ===");  print(met.round(4).to_string(index=False))
    for a in advertencias:
        print(f"!!! ADVERTENCIA {a}")


if __name__ == "__main__":
    main()
