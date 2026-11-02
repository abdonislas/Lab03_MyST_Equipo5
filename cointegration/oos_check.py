# Validacion fuera de muestra (solo diagnostico): spread con el vector FIJO de train
# evaluado en test y val. No cambia la eleccion del grupo.
# Ejecutar desde la raiz: python -m cointegration.oos_check
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from cointegration.screening import vida_media, PVAL, HL_MIN, HL_MAX

ART = "artifacts"
MUESTRAS = ["train", "test", "val"]     # train solo como referencia


def evaluar_muestra(nombre, cols, w):
    """ADF y vida media del spread log-precios @ w en prices_{nombre}.csv."""
    px = pd.read_csv(f"{ART}/prices_{nombre}.csv", index_col=0, parse_dates=True)
    spread = np.log(px[cols]).dropna() @ w
    adf_p = adfuller(spread, autolag="AIC")[1]
    hl = vida_media(spread)
    return {"muestra": nombre, "fecha_inicio": spread.index[0].date(),
            "fecha_fin": spread.index[-1].date(), "n_obs": len(spread),
            "ADF_spread_p": round(adf_p, 4), "vida_media": round(hl, 1),
            "pasa_ADF": adf_p < PVAL, "pasa_vida_media": HL_MIN <= hl <= HL_MAX}


def main():
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    cols, w = chosen["grupo"].split("+"), np.asarray(chosen["vector"], float)
    out = pd.DataFrame([evaluar_muestra(m, cols, w) for m in MUESTRAS])
    out.to_csv(f"{ART}/oos_check.csv", index=False)
    print(f"Grupo {chosen['grupo']}  vector={chosen['vector']}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
