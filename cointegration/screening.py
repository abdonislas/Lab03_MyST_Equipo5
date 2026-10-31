# cointegration/screening.py
#realizamos la: correlacion -> EG -> Johansen -> ADF spread -> vida media.
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

ART = "artifacts"
CORR_WIN, CORR_THR = 126, 0.70
PVAL = 0.05
HL_MIN, HL_MAX = 1.0, 252.0     # vida media razonable: entre 1 dia y 1 anio


def normalizar_vector(vec, leg=0):
    """Normaliza el vector de cointegracion por una pata. Convencion: pata 'leg' = +1."""
    vec = np.asarray(vec, dtype=float)
    if not np.isfinite(vec[leg]) or vec[leg] == 0:
        raise ValueError("Pata de normalizacion invalida")
    w = vec / vec[leg]
    if w[leg] < 0:        
        w = -w
    return w


def vida_media(spread):
    s = pd.Series(spread).dropna()
    ds = s.diff().dropna()
    s_lag = s.shift(1).dropna().loc[ds.index]
    beta = sm.OLS(ds.values, sm.add_constant(s_lag.values)).fit().params[1]
    return -np.log(2) / beta if beta < 0 else np.inf


def corr_movil_min(logpx, cols):
    """Minima correlacion movil 126d promedio entre todos los pares del grupo."""
    mins = []
    for a, b in itertools.combinations(cols, 2):
        rc = logpx[a].rolling(CORR_WIN).corr(logpx[b]).mean()
        mins.append(rc)
    return float(np.nanmin(mins))


def engle_granger_pares(logpx):
    """EG en los 15 pares en AMBOS sentidos (OLS + ADF sobre residuales via coint)."""
    filas = []
    for a, b in itertools.combinations(logpx.columns, 2):
        p_ab = coint(logpx[a], logpx[b])[1]
        p_ba = coint(logpx[b], logpx[a])[1]
        filas.append({"par": f"{a}-{b}", "EG_p_AB": round(p_ab, 4),
                      "EG_p_BA": round(p_ba, 4),
                      "pasa_EG": (min(p_ab, p_ba) < PVAL)})
    return pd.DataFrame(filas)


def evaluar_grupo(logpx, cols):
    """Corre el embudo sobre un grupo y devuelve la etapa donde cae (o 'PASA')."""
    data = logpx[list(cols)].dropna()

    if corr_movil_min(logpx, cols) < CORR_THR:
        return {"grupo": "+".join(cols), "n": len(cols), "etapa_caida": "correlacion",
                "coint_rank": np.nan, "ADF_spread_p": np.nan, "vida_media": np.nan}

    jo = coint_johansen(data, det_order=0, k_ar_diff=1)
    rank = int((jo.lr1 > jo.cvt[:, 1]).sum())      # relaciones al 95% (traza)
    rank_me = int((jo.lr2 > jo.cvm[:, 1]).sum())   # idem con max-eigenvalue
    if rank == 0 or rank_me == 0:
        return {"grupo": "+".join(cols), "n": len(cols), "etapa_caida": "johansen",
                "coint_rank": rank, "ADF_spread_p": np.nan, "vida_media": np.nan}

    w = normalizar_vector(jo.evec[:, 0])           # 1er vector = mas reversivo
    spread = data.values @ w
    adf_p = adfuller(spread, autolag="AIC")[1]
    if adf_p >= PVAL:
        return {"grupo": "+".join(cols), "n": len(cols), "etapa_caida": "ADF_spread",
                "coint_rank": rank, "ADF_spread_p": round(adf_p, 4), "vida_media": np.nan}

    hl = vida_media(spread)
    if not (HL_MIN <= hl <= HL_MAX):
        return {"grupo": "+".join(cols), "n": len(cols), "etapa_caida": "vida_media",
                "coint_rank": rank, "ADF_spread_p": round(adf_p, 4),
                "vida_media": round(hl, 1)}

    return {"grupo": "+".join(cols), "n": len(cols), "etapa_caida": "PASA",
            "coint_rank": rank, "ADF_spread_p": round(adf_p, 4),
            "vida_media": round(hl, 1), "vector": np.round(w, 4).tolist()}


def main():
    px = pd.read_csv(f"{ART}/prices_train.csv", index_col=0, parse_dates=True)
    logpx = np.log(px)
    cols = list(logpx.columns)

    # --- Engle-Granger en los 15 pares, ambos sentidos ---
    eg = engle_granger_pares(logpx)
    eg.to_csv(f"{ART}/engle_granger_pairs.csv", index=False)

    # --- Johansen + embudo en los 57 grupos (tamanos 2 a 6) ---
    grupos = []
    for k in range(2, 7):
        grupos += list(itertools.combinations(cols, k))
    res = pd.DataFrame([evaluar_grupo(logpx, g) for g in grupos])
    res = res.sort_values(["etapa_caida", "ADF_spread_p"], na_position="last")
    res.to_csv(f"{ART}/screening_all_groups.csv", index=False)

    # --- Falsos positivos esperados (N_tests * alpha) ---
    n_eg = len(eg) * 2        # 15 pares x 2 sentidos
    n_jo = len(grupos)        # 57 grupos
    fp = pd.DataFrame([
        {"prueba": "Engle-Granger", "n_tests": n_eg, "alpha": PVAL, "FP_esperados": n_eg * PVAL},
        {"prueba": "Johansen",      "n_tests": n_jo, "alpha": PVAL, "FP_esperados": n_jo * PVAL},
    ])
    fp.to_csv(f"{ART}/false_positives.csv", index=False)

    # --- Grupo elegido: el que PASA con menor p-valor ADF (mas reversivo) ---
    pasan = res[res["etapa_caida"] == "PASA"]
    elegido = pasan.sort_values("ADF_spread_p").iloc[0] if len(pasan) else None

    print("=== Falsos positivos esperados ===");  print(fp.to_string(index=False))
    print(f"\n=== Grupos por etapa de caida (total {len(grupos)}) ===")
    print(res["etapa_caida"].value_counts().to_string())
    if elegido is not None:
        print(f"\n>>> GRUPO ELEGIDO: {elegido['grupo']}  vector={elegido['vector']}  "
              f"ADF_p={elegido['ADF_spread_p']}  vida_media={elegido['vida_media']}d")
        pd.Series({"grupo": elegido["grupo"], "vector": elegido["vector"]}).to_json(
            f"{ART}/chosen_group.json")
    else:
        print("\n>>> Ningun grupo paso el embudo completo. Revisa umbrales.")


if __name__ == "__main__":
    main()