# tests/test_normalizacion.py
# Prueba 2 del enunciado: el vector de pesos mantiene escala y signo estables entre
# reestimaciones consecutivas.
#  - Johansen (train): normalizado sobre la pata de mayor valor absoluto (AMT = 1) y con
#    SBAC de signo contrario.
#  - Filtro 1 (Kalman): el vector que se opera es w_t = (1, -beta_t), con AMT fija en 1.
#    Despues del burn-in, beta_t no cambia de signo y entre un dia y el siguiente se mueve poco.
import json
import numpy as np
import pandas as pd
from strategy.generar_senales import ART

SALTO_MAX = 0.05     # cambio maximo permitido de beta entre dias consecutivos


def test_vector_johansen_normalizado():
    chosen = json.load(open(f"{ART}/chosen_group.json"))
    w = np.asarray(chosen["vector"], dtype=float)
    assert chosen["pata_normalizacion"] == "AMT"
    assert w[0] == 1.0, "la pata de normalizacion debe valer 1"
    assert np.abs(w).max() == 1.0, "la pata de normalizacion debe ser la de mayor valor absoluto"
    assert w[1] < 0, "SBAC debe tener signo contrario a AMT"


def test_pesos_kalman_estables():
    s = pd.read_csv(f"{ART}/senales_kalman.csv", index_col=0, parse_dates=True)
    operable = ~s["en_burn_in"].astype(str).str.lower().isin(["true", "1"])
    beta = s.loc[operable, "beta"].to_numpy(dtype=float)
    saltos = np.abs(np.diff(beta))
    print(f"\nbeta despues del burn-in: min={beta.min():.4f}  max={beta.max():.4f}  "
          f"salto diario maximo={saltos.max():.4f}  p99={np.quantile(saltos, 0.99):.4f}")
    assert np.all(beta > 0), "beta cambio de signo: el vector (1, -beta) se invertiria"
    assert saltos.max() < SALTO_MAX, f"beta salto {saltos.max():.4f} en un dia (limite {SALTO_MAX})"
