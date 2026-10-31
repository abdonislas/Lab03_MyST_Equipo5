#Prueba Johansen primer eigenvalue
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

ART = "artifacts"
WIN, STEP = 252, 5     # ventana de 1 anio, paso semanal


def normalizar_vector(vec, leg=0):
    vec = np.asarray(vec, float)
    w = vec / vec[leg]
    return -w if w[leg] < 0 else w


def main():
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    cols = chosen["grupo"].split("+")
    px = pd.read_csv(f"{ART}/prices_train.csv", index_col=0, parse_dates=True)
    data = np.log(px[cols]).dropna()

    filas = []
    for fin in range(WIN, len(data) + 1, STEP):
        ventana = data.iloc[fin - WIN:fin]
        try:
            jo = coint_johansen(ventana, det_order=0, k_ar_diff=1)
            w = normalizar_vector(jo.evec[:, 0])
        except Exception:
            continue
        fila = {"fecha": ventana.index[-1].date()}
        fila.update({c: round(wi, 4) for c, wi in zip(cols, w)})
        filas.append(fila)

    out = pd.DataFrame(filas).set_index("fecha")
    out.to_csv(f"{ART}/rolling_eigenvector.csv")
    print(out.tail().to_string())
    print(f"\nGuardado: {ART}/rolling_eigenvector.csv  ({len(out)} ventanas)")


if __name__ == "__main__":
    main()