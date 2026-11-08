# cointegration/robustez_oos.py
# Robustez fuera de muestra: re-parte el TRAIN en sub-train/sub-test/sub-val,
# estima el vector de Johansen SOLO en sub-train, lo congela, y mide si el
# spread sigue siendo estacionario en sub-test y sub-val, o si se rompe.
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from cointegration.screening import normalizar_vector, vida_media  # reusa tus funciones

ART = "artifacts"
PVAL = 0.05


def sub_split(df, p_train=0.60, p_test=0.20):
    """Re-parte un DataFrame cronologicamente en 3 (sin lookahead)."""
    n = len(df)
    i1, i2 = int(n * p_train), int(n * (p_train + p_test))
    return {"sub_train": df.iloc[:i1],
            "sub_test":  df.iloc[i1:i2],
            "sub_val":   df.iloc[i2:]}


def evaluar_spread(spread):
    s = pd.Series(spread).dropna()
    adf_p = adfuller(s.values, autolag="AIC")[1]
    hl = vida_media(s.values)
    return {"n_obs": len(s), "ADF_spread_p": round(adf_p, 4),
            "estacionario": bool(adf_p < PVAL),
            "vida_media": round(hl, 1) if np.isfinite(hl) else np.inf}


def main():
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    cols = chosen["grupo"].split("+")

    px = pd.read_csv(f"{ART}/prices_train.csv", index_col=0, parse_dates=True)
    logpx = np.log(px[cols]).dropna()
    partes = sub_split(logpx)

    # 1) Vector estimado SOLO en sub-train y congelado
    jo = coint_johansen(partes["sub_train"], det_order=0, k_ar_diff=1)
    w, leg = normalizar_vector(jo.evec[:, 0])

    # 2) Mismo vector aplicado a los tres pedazos
    filas = [{"sub_muestra": nombre, **evaluar_spread(parte.values @ w)}
             for nombre, parte in partes.items()]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(f"{ART}/robustez_oos.csv", index=False)

    # 3) Veredicto: se rompe si el spread deja de ser estacionario fuera de sub-train
    est = tabla.set_index("sub_muestra")["estacionario"].to_dict()
    rompe_test = not est.get("sub_test", False)
    rompe_val = not est.get("sub_val", False)

    print(f"Grupo: {chosen['grupo']}  |  vector (sub-train, pata={cols[leg]}): "
          f"{np.round(w, 4).tolist()}")
    print(tabla.to_string(index=False))
    if rompe_test:
        print("\n>>> SE ROMPE en sub-test: el spread deja de ser estacionario fuera de muestra.")
        print("    La relacion de equilibrio NO es estable -> posible artefacto/sobreajuste.")
    elif rompe_val:
        print("\n>>> Aguanta sub-test pero SE ROMPE en sub-val. Estabilidad parcial.")
    else:
        print("\n>>> AGUANTA: estacionario en los tres pedazos -> relacion estructural, no artefacto.")


if __name__ == "__main__":
    main()