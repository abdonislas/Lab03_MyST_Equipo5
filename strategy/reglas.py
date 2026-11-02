# strategy/reglas.py
# Politica PFA: convierte el z-score en una posicion d_t en {-1, 0, +1}.
#   d_t = +1 -> largo el spread (compra AMT, vende beta*SBAC): el spread esta "barato"
#   d_t = -1 -> corto el spread: el spread esta "caro"
# TIMING: la senal d_t se calcula con el precio de CIERRE del dia t y se ejecuta en t+1.
# Ese desfase lo aplica el backtest (integrante C); aqui NO se desplaza la senal.
import numpy as np
import pandas as pd

VIDA_MEDIA = 41.3
THETA_INICIAL = {"z_in": 2.0, "z_out": 0.5, "z_stop": 3.5, "h_max": round(2 * VIDA_MEDIA)}


def aplicar_reglas(z, z_in, z_out, z_stop, h_max, operable=None):
    """Recorre z_t dia por dia y decide la posicion.
    Recibe la serie z (array o Series), los umbrales theta y 'operable' (bool por dia;
    False = no se opera, p. ej. burn-in). Regresa DataFrame con d, h y motivo_cierre.
      - Flat: entra si z_in < |z_t| <= z_stop (+1 si z<0, -1 si z>0), salvo que este bloqueado.
      - En posicion: cierra por 'salida' (|z|<z_out), 'stop_z' (|z|>z_stop)
        o 'stop_tiempo' (h >= h_max); si no, mantiene.
      - Tras un stop_z o stop_tiempo queda bloqueado hasta que |z_t| < z_in.
      - h_t = dias desde la entrada (0 el dia que entra; el dia del cierre trae los dias
        que duro la operacion)."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    if operable is None:
        operable = np.ones(n, dtype=bool)
    d_prev, h, bloqueado = 0, 0, False
    d_out, h_out, motivo_out = np.zeros(n, dtype=int), np.zeros(n, dtype=int), [""] * n

    for t in range(n):
        zt = z[t]
        d, motivo = 0, ""
        if not operable[t] or np.isnan(zt):
            # Dia no operable (burn-in) o sin z: se queda flat
            d, h = 0, 0
        elif d_prev != 0:
            # ---- Estamos en posicion ----
            h += 1                                   # un dia mas en la operacion
            if abs(zt) < z_out:
                motivo = "salida"                    # el spread regreso a la media
            elif abs(zt) > z_stop:
                motivo = "stop_z"                    # el spread se fue demasiado lejos
                bloqueado = True
            elif h >= h_max:
                motivo = "stop_tiempo"               # tardo demasiado en revertir
                bloqueado = True
            d = 0 if motivo else d_prev              # cierra o mantiene
        else:
            # ---- Estamos flat ----
            if bloqueado and abs(zt) < z_in:
                bloqueado = False                    # el z regreso a la zona normal
            if not bloqueado and z_in < abs(zt) <= z_stop:
                d = 1 if zt < 0 else -1              # spread barato -> largo; caro -> corto
                h = 0                                # dia de entrada
        d_out[t] = d
        h_out[t] = h if (d != 0 or motivo) else 0
        motivo_out[t] = motivo
        d_prev = d

    return pd.DataFrame({"d": d_out, "h": h_out, "motivo_cierre": motivo_out})
