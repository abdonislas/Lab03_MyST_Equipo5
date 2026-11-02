# tests/test_causalidad.py
# Prueba de lookahead: si cortamos los datos en una fecha T, todo lo calculado hasta T
# debe ser IDENTICO a la corrida con todos los datos. Si cambia, algo usa el futuro.
import json
import numpy as np
from strategy.generar_senales import cargar_precios, correr_pipeline, ART


def test_sin_lookahead():
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    beta_fijo, vida_media = -chosen["vector"][1], chosen["vida_media"]
    px = cargar_precios()

    # T = fecha a la mitad de la muestra de test
    fechas_test = px.index[px["muestra"] == "test"]
    T = fechas_test[len(fechas_test) // 2]

    completo = correr_pipeline(px, beta_fijo, vida_media)
    cortado = correr_pipeline(px.loc[:T], beta_fijo, vida_media)
    print(f"\nCorte T = {T.date()}  (filas comparadas: {len(cortado['kalman'])})")

    for metodo, cols in [("kalman", ["z", "beta", "d"]), ("estatico", ["z", "d"])]:
        a = completo[metodo].loc[:T]
        b = cortado[metodo]
        for c in cols:
            # assert_array_equal trata NaN == NaN como iguales
            np.testing.assert_array_equal(a[c].values, b[c].values,
                                          err_msg=f"lookahead en {metodo}.{c}")
