# main.py
# Corre todo el pipeline:  python main.py
# Usa los precios guardados

import argparse
import importlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

SEMILLA = 42
RAIZ = Path(__file__).resolve().parent
PRECIOS = ["prices_full.csv", "prices_train.csv", "prices_test.csv", "prices_val.csv"]

#  (nombre, modulo, descripcion)
PASOS = [
    ("datos", "data.download_data", "Descarga, revision de huecos y saltos, split 60/20/20"),
    ("integracion", "cointegration.integration_order", "Orden de integracion (ADF y KPSS)"),
    ("screening", "cointegration.screening", "Correlacion, Engle-Granger, Johansen y embudo"),
    ("fuera_de_muestra", "cointegration.oos_check", "Vector de train evaluado en test y val"),
    ("johansen_movil", "cointegration.rolling_johansen", "Johansen movil (primer eigenvector)"),
    ("vecm", "strategy.VECM", "VECM con el vector de Johansen"),
    ("grid_search", "strategy.grid_search", "Grid search de theta en train (unos 15 segundos)"),
    ("senales", "strategy.generar_senales", "Filtros de Kalman, reglas y senales"),
    ("backtest", "backtest.motor", "Backtest multi-pata y politicas de rebalanceo"),
    ("metricas", "performance.metricas", "Metricas de desempeno"),
]
NOMBRES = [p[0] for p in PASOS]


def fijar_semilla(semilla=SEMILLA):
    random.seed(semilla)
    np.random.seed(semilla)
    os.environ.setdefault("PYTHONHASHSEED", str(semilla))


def hay_snapshot():
    return all((RAIZ / "artifacts" / f).exists() for f in PRECIOS)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lab 03 MyST, Equipo 5: pipeline completo")
    parser.add_argument("--descargar", action="store_true",
                        help="volver a descargar precios con yfinance en lugar de usar el snapshot")
    parser.add_argument("--desde", choices=NOMBRES, default=NOMBRES[0],
                        help="correr desde este paso en adelante")
    args = parser.parse_args(argv)

    os.chdir(RAIZ)                          # todos los modulos usan rutas relativas a la raiz
    sys.path.insert(0, str(RAIZ))
    (RAIZ / "artifacts").mkdir(exist_ok=True)
    fijar_semilla()

    inicio = NOMBRES.index(args.desde)
    t_total = time.time()
    for i, (nombre, modulo, descripcion) in enumerate(PASOS[inicio:], start=inicio + 1):
        print(f"\n{'=' * 78}\n[{i}/{len(PASOS)}] {nombre}: {descripcion}\n{'=' * 78}")
        if nombre == "datos" and not args.descargar and hay_snapshot():
            print("Se usan los precios guardados en artifacts/ (python main.py --descargar "
                  "para bajarlos de nuevo).")
            continue
        t0 = time.time()
        fijar_semilla()
        importlib.import_module(modulo).main()
        print(f"--- {nombre} listo en {time.time() - t0:.1f} s")

    print(f"\nPipeline completo en {time.time() - t_total:.1f} s.")
    print("Dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()