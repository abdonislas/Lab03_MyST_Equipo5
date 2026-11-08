# strategy/grid_search.py / Se cambio a bayes search

# Paso 6 de Powell: buscar theta = (z_in, z_out, z_stop, h_max) en TRAIN y congelarlo.
# OPTIMIZACION BAYESIANA (Optuna / TPE) en lugar de rejilla exhaustiva.
#   Kalman:   delta en {1e-6, 1e-5} (categorico, cada uno con su R); theta continuo.
#   Estatico: L (ventana del z-score) entero en [vida_media, 3*vida_media]; theta continuo.
# Objetivo: Sharpe neto en train (despues del burn-in) con al menos 10 operaciones.
# Ejecutar desde la raiz:  python -m strategy.grid_search
import json
import numpy as np
import pandas as pd
import optuna

from kalman.filtro_pesos import BURN_IN
from strategy.generar_senales import ART, cargar_precios, correr_pipeline
from strategy.reglas import aplicar_reglas
from strategy.estatico import zscore_estatico
from strategy.evaluador import evaluar

SEMILLA = 42
VIDA_MEDIA = 41.3
DELTAS = [1e-6, 1e-5]
N_TRIALS_K, N_TRIALS_E = 150, 150        # evaluaciones por metodo (ajustable)
MIN_OPS, MIN_OPS_RESPALDO = 10, 5
METRICAS = ["sharpe", "retorno_total", "max_drawdown", "n_ops", "win_rate",
            "comisiones", "prestamo"]
# Rangos de busqueda continuos (equivalentes a la antigua rejilla)
RANGOS = {"z_in": (1.5, 2.5), "z_out": (0.0, 0.5), "z_stop": (3.0, 4.0),
          "h_max": (round(VIDA_MEDIA), round(3 * VIDA_MEDIA))}      # [41, 124]


def sugerir_theta(trial):
    return {
        "z_in":   trial.suggest_float("z_in",   *RANGOS["z_in"]),
        "z_out":  trial.suggest_float("z_out",  *RANGOS["z_out"]),
        "z_stop": trial.suggest_float("z_stop", *RANGOS["z_stop"]),
        "h_max":  trial.suggest_int("h_max",    *RANGOS["h_max"]),
    }


def evaluar_theta(base, z, theta, operable, mascara):
    """Aplica reglas con theta sobre z y evalua el P&L en las filas de 'mascara'."""
    reglas = aplicar_reglas(z, operable=operable, **theta)
    df = base.assign(d=reglas["d"].values)[mascara]
    res = evaluar(df)
    return {k: res[k] for k in METRICAS}


def objetivo_penalizado(met):
    """Sharpe real si cumple el minimo de ops; si no, valor bajo que premia acercarse."""
    if met["n_ops"] < MIN_OPS:
        return -10.0 + 0.01 * met["n_ops"]
    return met["sharpe"]


def elegir(grid):
    """Mejor Sharpe REAL con >= 10 ops; respaldo >= 5; si no, el mejor Sharpe."""
    for minimo in (MIN_OPS, MIN_OPS_RESPALDO):
        factibles = grid[grid["n_ops"] >= minimo]
        if len(factibles):
            aviso = "" if minimo == MIN_OPS else \
                f"ninguna configuracion llega a {MIN_OPS} operaciones; se uso minimo {minimo}"
            return factibles.loc[factibles["sharpe"].idxmax()], minimo, aviso
    aviso = f"ninguna configuracion llega a {MIN_OPS_RESPALDO} operaciones; se eligio el mejor Sharpe"
    return grid.loc[grid["sharpe"].idxmax()], 0, aviso


def trials_a_df(estudio, extra):
    """Convierte los trials completos en un DataFrame con theta + metricas (como la rejilla vieja)."""
    filas = []
    for t in estudio.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        fila = dict(t.params)
        fila.update({k: t.user_attrs.get(k) for k in METRICAS})
        fila.update({k: t.user_attrs.get(k) for k in extra})
        filas.append(fila)
    return pd.DataFrame(filas)


def metricas_por_muestra(base, z, theta, muestra, en_burn_in):
    """theta* evaluado en train/test/val por separado; cada muestra arranca PLANA."""
    filas = []
    for m in ["train", "test", "val"]:
        mascara = (muestra == m) & ~en_burn_in
        z_m = np.asarray(z)[mascara]
        reglas = aplicar_reglas(z_m, **theta)
        df = base[mascara].assign(d=reglas["d"].values)
        res = evaluar(df)
        filas.append({"muestra": m, **{k: res[k] for k in METRICAS}})
    return filas


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    beta_fijo, vida_media = -chosen["vector"][1], chosen["vida_media"]
    px = cargar_precios()
    muestra = px["muestra"].values
    en_burn_in = np.arange(len(px)) < BURN_IN
    mascara_train = (muestra == "train") & ~en_burn_in

    # ---------- Kalman: precalcula los filtros por delta (solo 2 corridas caras) ----------
    series_k = {}
    for delta in DELTAS:
        res = correr_pipeline(px, beta_fijo, vida_media, delta=delta)
        k = res["kalman"]
        series_k[delta] = (k[["AMT", "SBAC", "beta"]], k["z"].values, res["params"]["R"])

    def objetivo_kalman(trial):
        delta = trial.suggest_categorical("delta", DELTAS)
        theta = sugerir_theta(trial)
        base, z, R = series_k[delta]
        met = evaluar_theta(base, z, theta, ~en_burn_in, mascara_train)
        for kk, vv in met.items():
            trial.set_user_attr(kk, vv)
        trial.set_user_attr("R", R)
        return objetivo_penalizado(met)

    est_k = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    est_k.optimize(objetivo_kalman, n_trials=N_TRIALS_K)
    grid_k = trials_a_df(est_k, extra=["R"])
    grid_k["factible"] = grid_k["n_ops"] >= MIN_OPS

    # ---------- Estatico: z-score por L con cache (L entero, theta continuo) ----------
    log_a, log_s = np.log(px["AMT"]), np.log(px["SBAC"])
    cache_e = {}

    def serie_estatica(L):
        if L not in cache_e:
            est = zscore_estatico(log_a, log_s, beta_fijo, L)
            base = pd.concat([px[["AMT", "SBAC"]], est[["beta"]]], axis=1)
            cache_e[L] = (base, est["z"].values)
        return cache_e[L]

    def objetivo_estatico(trial):
        L = trial.suggest_int("L", *RANGOS["h_max"])           # misma banda de vida media
        theta = sugerir_theta(trial)
        base, z = serie_estatica(L)
        met = evaluar_theta(base, z, theta, ~en_burn_in, mascara_train)
        for kk, vv in met.items():
            trial.set_user_attr(kk, vv)
        return objetivo_penalizado(met)

    est_e = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    est_e.optimize(objetivo_estatico, n_trials=N_TRIALS_E)
    grid_e = trials_a_df(est_e, extra=[])
    grid_e["factible"] = grid_e["n_ops"] >= MIN_OPS

    grid_k.to_csv(f"{ART}/grid_kalman.csv", index=False)
    grid_e.to_csv(f"{ART}/grid_estatico.csv", index=False)

    # ---------- Elegir y congelar theta* (misma logica de seleccion) ----------
    mejor_k, min_k, aviso_k = elegir(grid_k)
    mejor_e, min_e, aviso_e = elegir(grid_e)
    theta_k = {c: (int(mejor_k[c]) if c == "h_max" else float(mejor_k[c])) for c in RANGOS}
    theta_e = {c: (int(mejor_e[c]) if c == "h_max" else float(mejor_e[c])) for c in RANGOS}
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

    # ---------- Metricas por muestra con theta* ----------
    base_k, z_k, _ = series_k[delta_opt]
    base_e, z_e = serie_estatica(L_opt)
    filas = [{"metodo": "kalman", **f} for f in metricas_por_muestra(base_k, z_k, theta_k, muestra, en_burn_in)]
    filas += [{"metodo": "estatico", **f} for f in metricas_por_muestra(base_e, z_e, theta_e, muestra, en_burn_in)]
    met = pd.DataFrame(filas)
    met.to_csv(f"{ART}/metricas_preliminares.csv", index=False)

    # ---------- Reporte ----------
    for nombre, grid in [("KALMAN", grid_k), ("ESTATICO", grid_e)]:
        print(f"\n=== {nombre}: {len(grid)} trials, {int(grid['factible'].sum())} factibles "
              f"(>= {MIN_OPS} ops) ===")
        print(grid[grid["factible"]].nlargest(5, "sharpe").round(4).to_string(index=False))
    print("\n=== theta* ===");  print(json.dumps(optimo, indent=2))
    print("\n=== Metricas por muestra ===");  print(met.round(4).to_string(index=False))
    for a in advertencias:
        print(f"!!! ADVERTENCIA {a}")


if __name__ == "__main__":
    main()