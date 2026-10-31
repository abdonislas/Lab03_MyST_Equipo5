
#Hacemos la cointegración por activo esto para empezar a filtrar
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

ART = "artifacts"


def tabla_integracion(logpx):
    filas = []
    for t in logpx.columns:
        s = logpx[t].dropna()
        adf_lvl = adfuller(s, autolag="AIC")[1]
        adf_dif = adfuller(s.diff().dropna(), autolag="AIC")[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # KPSS avisa cuando p sale del rango
            kpss_lvl = kpss(s, regression="c", nlags="auto")[1]
        # Clasificacion: I(1) si ADF-nivel NO rechaza, ADF-diff SI rechaza y KPSS rechaza
        es_i1 = (adf_lvl > 0.05) and (adf_dif < 0.05) and (kpss_lvl < 0.05)
        filas.append({
            "ticker": t,
            "ADF_nivel_p": round(adf_lvl, 4),
            "ADF_diff_p": round(adf_dif, 4),
            "KPSS_nivel_p": round(kpss_lvl, 4),
            "orden": "I(1)" if es_i1 else "revisar",
        })
    return pd.DataFrame(filas)


def main():
    px = pd.read_csv(f"{ART}/prices_train.csv", index_col=0, parse_dates=True)
    logpx = np.log(px)
    tab = tabla_integracion(logpx)
    tab.to_csv(f"{ART}/integration_order.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()