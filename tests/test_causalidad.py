# tests/test_causalidad.py
# Prueba de lookahead: si cortamos los datos en una fecha T, todo lo calculado hasta T
# debe ser IDENTICO a la corrida con todos los datos. Si cambia, algo usa el futuro.
# El enunciado pide al menos tres valores de t: se prueban dos cortes en test y uno en val.
# No se corta dentro de train a proposito: delta, R y la inicializacion de los filtros se calibran
# con train completo y luego se congelan (es estimacion dentro de muestra, como el vector de
# Johansen). La causalidad que importa es la de la operacion fuera de muestra.
import json
import numpy as np
import pytest
from strategy.generar_senales import cargar_precios, correr_pipeline, ART


@pytest.fixture(scope="module")
def datos():
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    beta_fijo, vida_media = -chosen["vector"][1], chosen["vida_media"]
    px = cargar_precios()
    completo = correr_pipeline(px, beta_fijo, vida_media)
    return px, beta_fijo, vida_media, completo


@pytest.mark.parametrize("muestra,fraccion", [("test", 1 / 3), ("test", 2 / 3), ("val", 1 / 2)])
def test_sin_lookahead(datos, muestra, fraccion):
    px, beta_fijo, vida_media, completo = datos
    # T = fecha a una fraccion de la muestra indicada
    fechas = px.index[px["muestra"] == muestra]
    T = fechas[int(len(fechas) * fraccion)]

    cortado = correr_pipeline(px.loc[:T], beta_fijo, vida_media)
    print(f"\nCorte T = {T.date()} ({muestra}); filas comparadas: {len(cortado['kalman'])}")

    for metodo, cols in [("kalman", ["z", "beta", "d"]), ("estatico", ["z", "d"])]:
        a = completo[metodo].loc[:T]
        b = cortado[metodo]
        for c in cols:
            # assert_array_equal trata NaN == NaN como iguales
            np.testing.assert_array_equal(a[c].values, b[c].values,
                                          err_msg=f"lookahead en {metodo}.{c} con corte en {muestra}")
