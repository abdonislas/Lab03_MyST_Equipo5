# tests/test_calibracion_filtro.py
# Si los filtros estan bien calibrados, sus innovaciones estandarizadas deben parecerse
# a una N(0, 1) en el train (despues del burn-in): media ~ 0 y varianza ~ 1.
import json
import numpy as np
import pytest
from strategy.generar_senales import cargar_precios, correr_pipeline, ART

MEDIA_MAX = 0.1
VAR_MIN, VAR_MAX = 0.7, 1.3


@pytest.fixture(scope="module")
def resultado():
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    return correr_pipeline(cargar_precios(), -chosen["vector"][1], chosen["vida_media"])


def _train_post_burnin(res):
    """Mascara: filas de train que ya salieron del burn-in."""
    k = res["kalman"]
    return ((k["muestra"] == "train") & ~k["en_burn_in"]).values


def _revisar(nombre, serie):
    media, var = float(np.mean(serie)), float(np.var(serie, ddof=1))
    print(f"\n{nombre}: media={media:.4f}  varianza={var:.4f}  (n={len(serie)})")
    assert abs(media) < MEDIA_MAX, f"{nombre}: |media|={abs(media):.4f} >= {MEDIA_MAX}"
    assert VAR_MIN <= var <= VAR_MAX, f"{nombre}: varianza={var:.4f} fuera de [{VAR_MIN}, {VAR_MAX}]"


def _revisar_autocorrelacion(nombre, serie):
    """Igual que _revisar, pero el criterio de la media usa el error estandar EFECTIVO.
    El criterio anterior (|media| < 0.1) suponia observaciones independientes. El z del
    filtro 2 es muy persistente (rho1 ~ 0.95): con ~1500 dias hay muy pocos episodios
    independientes y el error estandar real de la media es mucho mayor que 1/sqrt(n).
      n_efectivo  = n * (1 - rho1) / (1 + rho1)
      EE_efectivo = sqrt(var / n_efectivo)
      criterio:     |media| < 2 * EE_efectivo"""
    serie = np.asarray(serie, dtype=float)
    n = len(serie)
    media, var = float(np.mean(serie)), float(np.var(serie, ddof=1))
    rho1 = float(np.corrcoef(serie[:-1], serie[1:])[0, 1])   # autocorrelacion de orden 1
    n_ef = n * (1 - rho1) / (1 + rho1)                         # observaciones "independientes"
    ee_ef = np.sqrt(var / n_ef)                                # error estandar de la media
    print(f"\n{nombre}: n={n}  rho1={rho1:.4f}  n_efectivo={n_ef:.1f}  "
          f"EE_efectivo={ee_ef:.4f}  media={media:.4f}  (limite {2 * ee_ef:.4f})  "
          f"varianza={var:.4f}")
    assert abs(media) < 2 * ee_ef, f"{nombre}: |media|={abs(media):.4f} >= 2*EE={2 * ee_ef:.4f}"
    assert VAR_MIN <= var <= VAR_MAX, f"{nombre}: varianza={var:.4f} fuera de [{VAR_MIN}, {VAR_MAX}]"


VAR_OOS_MIN, VAR_OOS_MAX = 0.5, 2.0     # banda fuera de muestra (test)


def test_innovaciones_filtro_pesos(resultado):
    # R se calibra por metodo de momentos para que var(e/sqrt(S)) = 1 en train post burn-in,
    # asi que el criterio [0.7, 1.3] en train se cumple POR CONSTRUCCION: es una prueba de
    # consistencia de la calibracion. La prueba informativa es la de test (fuera de muestra),
    # con banda mas amplia [0.5, 2.0] porque son solo 553 dias y otro regimen.
    f1 = resultado["filtro_pesos"]
    u = (f1["e"] / np.sqrt(f1["S"])).values
    _revisar("Filtro 1  e_t/sqrt(S_t) [train]", u[_train_post_burnin(resultado)])

    var_test = float(np.var(u[(f1["muestra"] == "test").values], ddof=1))
    var_val = float(np.var(u[(f1["muestra"] == "val").values], ddof=1))
    print(f"Filtro 1  var(e/sqrt(S)) fuera de muestra: test={var_test:.4f}  val={var_val:.4f}")
    assert VAR_OOS_MIN <= var_test <= VAR_OOS_MAX, \
        f"var test={var_test:.4f} fuera de [{VAR_OOS_MIN}, {VAR_OOS_MAX}]"


def test_zscore_filtro_spread(resultado):
    z = resultado["filtro_spread"]["z"].values[_train_post_burnin(resultado)]
    _revisar_autocorrelacion("Filtro 2  z_t", z)
