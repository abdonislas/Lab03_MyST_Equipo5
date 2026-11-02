# cointegration/screening.py
# Prefiltro informativo (correlacion movil de rendimientos + Engle-Granger, 15 pares)
# y embudo bloqueante en los 57 grupos: Johansen -> ADF spread -> vida media.
import itertools
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

ART = "artifacts"
CORR_WIN, CORR_THR = 126, 0.70
PVAL = 0.05
HL_MIN, HL_MAX = 5.0, 60.0      # vida media razonable: entre 5 y 60 dias habiles
ETAPAS = ["johansen", "ADF_spread", "vida_media", "PASA"]   # orden del embudo
GRUPO_MANUAL = "AMT+SBAC"       # si no es None, tiene prioridad sobre la eleccion automatica
CRITERIO_MANUAL = "manual: unico grupo que pasa sin bandera_roja y con mejor justificacion economica"


def normalizar_vector(vec):
    """Normaliza el vector de cointegracion por la pata de mayor |valor| (queda en +1).
    Regresa (vector normalizado, indice de la pata)."""
    vec = np.asarray(vec, dtype=float)
    leg = int(np.argmax(np.abs(vec)))
    if not np.isfinite(vec[leg]) or vec[leg] == 0:
        raise ValueError("Pata de normalizacion invalida")
    return vec / vec[leg], leg


def vida_media(spread):
    """Vida media (dias) de un AR(1) ajustado por OLS: d s_t = c + b s_{t-1}."""
    s = pd.Series(spread).dropna()
    ds = s.diff().dropna()
    s_lag = s.shift(1).dropna().loc[ds.index]
    beta = sm.OLS(ds.values, sm.add_constant(s_lag.values)).fit().params[1]
    return -np.log(2) / beta if beta < 0 else np.inf


def corr_movil_mediana(ret, a, b):
    """Mediana de la correlacion movil (ventana 126) entre rendimientos log de a y b."""
    return float(ret[a].rolling(CORR_WIN).corr(ret[b]).median())


def engle_granger_pares(logpx, corr_pares):
    """EG en los 15 pares en AMBOS sentidos + correlacion movil mediana (informativo)."""
    filas = []
    for a, b in itertools.combinations(logpx.columns, 2):
        p_ab = coint(logpx[a], logpx[b])[1]
        p_ba = coint(logpx[b], logpx[a])[1]
        rho = corr_pares[(a, b)]
        filas.append({"par": f"{a}-{b}", "corr_movil_mediana": round(rho, 4),
                      "pasa_corr": rho > CORR_THR,
                      "EG_p_AB": round(p_ab, 4), "EG_p_BA": round(p_ba, 4),
                      "pasa_EG": (min(p_ab, p_ba) < PVAL)})
    return pd.DataFrame(filas)


def evaluar_grupo(logpx, cols, corr_pares, no_i1):
    """Calcula todas las metricas del grupo y la primera etapa del embudo donde cae."""
    data = logpx[list(cols)].dropna()
    n = len(cols)

    jo = coint_johansen(data, det_order=0, k_ar_diff=1)
    rank_tr = int((jo.lr1 > jo.cvt[:, 1]).sum())   # relaciones al 95% (traza)
    rank_me = int((jo.lr2 > jo.cvm[:, 1]).sum())   # idem con max-eigenvalue
    activo_no_i1 = [c for c in cols if c in no_i1]
    bandera = rank_tr == n or rank_me == n or len(activo_no_i1) > 0

    w, leg = normalizar_vector(jo.evec[:, 0])      # 1er vector = mas reversivo
    spread = data.values @ w                       # spread en log-precios
    adf_p = adfuller(spread, autolag="AIC")[1]
    hl = vida_media(spread)

    if not (0 < rank_tr < n and 0 < rank_me < n):
        etapa = "johansen"
    elif adf_p >= PVAL:
        etapa = "ADF_spread"
    elif not (HL_MIN <= hl <= HL_MAX):
        etapa = "vida_media"
    else:
        etapa = "PASA"

    return {"grupo": "+".join(cols), "n": n,
            "corr_min_par": round(min(corr_pares[p] for p in itertools.combinations(cols, 2)), 4),
            "rank_traza": rank_tr, "rank_maxeig": rank_me, "bandera_roja": bandera,
            "activo_no_I1": activo_no_i1,
            "ADF_spread_p": round(adf_p, 4), "vida_media": round(hl, 1),
            "vector": np.round(w, 4).tolist(), "pata_normalizacion": cols[leg],
            "etapa_caida": etapa}


def elegir_grupo(res):
    """El que llega mas lejos en el embudo y, entre esos, menor ADF_spread_p."""
    r = res.assign(_nivel=res["etapa_caida"].map(ETAPAS.index))
    r = r.sort_values(["_nivel", "ADF_spread_p"], ascending=[False, True])
    return r.drop(columns="_nivel").iloc[0]


def main():
    px = pd.read_csv(f"{ART}/prices_train.csv", index_col=0, parse_dates=True)
    logpx = np.log(px)
    cols = list(logpx.columns)
    ret = logpx.diff()

    orden = pd.read_csv(f"{ART}/integration_order.csv")
    no_i1 = set(orden.loc[orden["orden"] != "I(1)", "ticker"])

    # --- Prefiltro informativo: correlacion movil + EG en los 15 pares ---
    corr_pares = {(a, b): corr_movil_mediana(ret, a, b)
                  for a, b in itertools.combinations(cols, 2)}
    eg = engle_granger_pares(logpx, corr_pares)
    eg.to_csv(f"{ART}/engle_granger_pairs.csv", index=False)

    # --- Johansen + embudo en los 57 grupos (tamanos 2 a 6), siempre ---
    grupos = []
    for k in range(2, 7):
        grupos += list(itertools.combinations(cols, k))
    res = pd.DataFrame([evaluar_grupo(logpx, g, corr_pares, no_i1) for g in grupos])
    res = res.sort_values(["etapa_caida", "ADF_spread_p"])
    res.to_csv(f"{ART}/screening_all_groups.csv", index=False)

    # --- Falsos positivos esperados (N_tests * alpha) ---
    n_eg = len(eg) * 2        # 15 pares x 2 sentidos
    n_jo = len(grupos)        # 57 grupos
    fp = pd.DataFrame([
        {"prueba": "Engle-Granger", "n_tests": n_eg, "alpha": PVAL, "FP_esperados": n_eg * PVAL},
        {"prueba": "Johansen",      "n_tests": n_jo, "alpha": PVAL, "FP_esperados": n_jo * PVAL},
    ])
    fp.to_csv(f"{ART}/false_positives.csv", index=False)

    # --- Grupo elegido (si nadie pasa, el mas cercano, sin esconderlo) ---
    auto = elegir_grupo(res)
    if GRUPO_MANUAL is not None:
        fila = res[res["grupo"] == GRUPO_MANUAL]
        if fila.empty:
            raise ValueError(f"GRUPO_MANUAL={GRUPO_MANUAL} no esta en los 57 grupos")
        elegido, criterio = fila.iloc[0], CRITERIO_MANUAL
    else:
        elegido, criterio = auto, "automatico: etapa mas avanzada y menor ADF_spread_p"
    pasa = elegido["etapa_caida"] == "PASA"
    chosen = {k: elegido[k] for k in ["grupo", "vector", "pata_normalizacion", "etapa_caida",
                                      "rank_traza", "rank_maxeig", "ADF_spread_p",
                                      "vida_media", "bandera_roja", "activo_no_I1"]}
    chosen["pasa_embudo"] = bool(pasa)
    chosen["criterio"] = criterio
    with open(f"{ART}/chosen_group.json", "w") as f:
        json.dump(chosen, f, indent=2, default=lambda x: x.item())

    print("=== Prefiltro informativo (15 pares) ===");  print(eg.to_string(index=False))
    print("\n=== Falsos positivos esperados ===");  print(fp.to_string(index=False))
    print(f"\n=== Grupos por etapa de caida (total {len(grupos)}) ===")
    print(res["etapa_caida"].value_counts().to_string())
    pasan = res[res["etapa_caida"] == "PASA"].sort_values("ADF_spread_p")
    print(f"\n=== Grupos que pasan el embudo completo ({len(pasan)}) ===")
    print(pasan[["grupo", "n", "rank_traza", "rank_maxeig", "ADF_spread_p",
                 "vida_media", "bandera_roja"]].to_string(index=False))
    if auto["etapa_caida"] != "PASA":
        print("\n!!! ADVERTENCIA: ningun grupo paso el embudo completo; "
              "se guarda el mas cercano con pasa_embudo=false.")
    print(f"\n--- Sugerido automatico: {auto['grupo']} (ADF_p={auto['ADF_spread_p']}, "
          f"bandera_roja={auto['bandera_roja']})")
    print(f">>> Criterio: {criterio}")
    print(f">>> GRUPO ELEGIDO: {elegido['grupo']}  vector={elegido['vector']} "
          f"(pata={elegido['pata_normalizacion']})  ADF_p={elegido['ADF_spread_p']}  "
          f"vida_media={elegido['vida_media']}d  pasa_embudo={pasa}")


if __name__ == "__main__":
    main()
