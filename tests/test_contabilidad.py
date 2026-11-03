# tests/test_contabilidad.py
# Prueba 3: valor = efectivo + patas, costos = comisiones + prestamo, y el P&L cuadra.
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.motor import Config, MUESTRAS, a_bool, correr_metodo, patas_del_grupo, simular

ART = Path(__file__).resolve().parents[1] / "artifacts"
CFG = Config()
TOL = 1e-6                     # tolerancia relativa al capital inicial


def prestamo_recalculado(diario, patas, precios, fechas):
    """Devengo de prestamo con las acciones del cierre anterior (sin usar la columna del motor)."""
    q = diario[[f"q_{p}" for p in patas]].to_numpy()
    total = 0.0
    for t in range(1, len(diario)):
        corto = np.clip(-q[t - 1], 0, None) @ precios[t - 1]
        total += corto * CFG.tasa_prestamo * (fechas[t] - fechas[t - 1]).days / CFG.base_dias
    return total


def revisar_identidades(diario, ops, fills, patas, precios, fechas):
    escala = CFG.capital * TOL
    # a) valor = efectivo + patas, todos los dias
    patas_valor = diario[[f"valor_{p}" for p in patas]].sum(axis=1)
    np.testing.assert_allclose(diario["capital"], diario["efectivo"] + patas_valor, atol=escala)
    # b) costos registrados = comision por operacion y pata + prestamo
    com_fills = float((fills["nocional"].abs() * CFG.comision).sum())
    prest = prestamo_recalculado(diario, patas, precios, fechas)
    costo_registrado = float(diario["comision"].sum() + diario["prestamo"].sum())
    assert costo_registrado == pytest.approx(com_fills + prest, abs=escala)
    assert float(fills["comision"].sum()) == pytest.approx(com_fills, abs=escala)
    # c) cierre del P&L
    cambio = float(diario["capital"].iloc[-1] - diario["capital"].iloc[0])
    neto = float(diario["pnl_bruto"].sum() - diario["comision"].sum() - diario["prestamo"].sum())
    assert cambio == pytest.approx(neto, abs=escala)
    assert float(ops["pnl_neto"].sum()) == pytest.approx(cambio, abs=escala)
    # al final de la muestra todo queda liquidado
    assert np.allclose(diario[[f"q_{p}" for p in patas]].iloc[-1], 0.0)


# Caso sintetico de 3 patas

@pytest.fixture(scope="module")
def sintetico():
    rng = np.random.default_rng(42)
    n, patas = 300, ["A", "B", "C"]
    fechas = pd.bdate_range("2020-01-01", periods=n)
    precios = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 3)), axis=0))
    pesos = np.column_stack([np.ones(n), -0.6 + 0.05 * np.sin(np.arange(n) / 20),
                             -0.3 + 0.02 * np.cos(np.arange(n) / 15)])
    d = np.zeros(n, dtype=int)
    d[10:60], d[100:180], d[250:] = 1, -1, 1     # la ultima queda abierta: se liquida
    return patas, fechas, precios, pesos, d


@pytest.mark.parametrize("banda", [0.0, 0.01, float("inf")])
def test_identidades_sintetico_3_patas(sintetico, banda):
    patas, fechas, precios, pesos, d = sintetico
    diario, ops, fills = simular(fechas, precios, pesos, d, banda, patas, CFG)
    assert len(ops) == 3
    assert ops["motivo_cierre"].iloc[-1] == "fin_muestra"
    revisar_identidades(diario, ops, fills, patas, precios, fechas)


def test_tamano_y_signos_al_entrar(sintetico):
    """Al entrar: bruto = 80% del capital y nocional por pata proporcional a |w| con signo d*w."""
    patas, fechas, precios, pesos, d = sintetico
    diario, ops, fills = simular(fechas, precios, pesos, d, float("inf"), patas, CFG)
    for _, op in ops.iterrows():
        f = fills[(fills["operacion"] == op["operacion"]) & (fills["tipo"] == "entrada")]
        t = fechas.get_loc(op["fecha_entrada"])
        w = pesos[t - 1]
        signo = 1 if op["direccion"] == "largo spread" else -1
        nocional = f.set_index("pata").loc[patas, "nocional"].to_numpy()
        assert np.abs(nocional).sum() == pytest.approx(CFG.exposicion * op["capital_entrada"])
        np.testing.assert_allclose(nocional / np.abs(nocional).sum(), signo * w / np.abs(w).sum())


# Senales reales (artifacts)

def _cargar(metodo):
    chosen = json.load(open(ART / "chosen_group.json"))
    opt = json.load(open(ART / "theta_optimo.json"))
    theta = {c: opt[metodo][c] for c in ["z_in", "z_out", "z_stop", "h_max"]}
    senales = pd.read_csv(ART / f"senales_{metodo}.csv", index_col=0, parse_dates=True)
    return patas_del_grupo(chosen), theta, senales


@pytest.mark.parametrize("metodo", ["kalman", "estatico"])
def test_identidades_senales_reales(metodo):
    patas, theta, senales = _cargar(metodo)
    diario, ops, fills = correr_metodo(senales, patas, theta,
                                       politicas={"diario": 0.0, "sin_rebalanceo": float("inf")})
    burn = a_bool(senales["en_burn_in"])
    for politica in ["diario", "sin_rebalanceo"]:
        for m in MUESTRAS:
            sub = senales[(senales["muestra"] == m).to_numpy() & ~burn]
            sel = lambda df: df[(df["politica"] == politica) & (df["muestra"] == m)]
            revisar_identidades(sel(diario).reset_index(drop=True), sel(ops), sel(fills),
                                patas, sub[patas].to_numpy(float), sub.index)


@pytest.mark.parametrize("metodo", ["kalman", "estatico"])
def test_sin_rebalanceo_reproduce_evaluador_simple(metodo):
    """Con acciones fijas, la curva de capital debe ser la del evaluador de strategy/."""
    from strategy.evaluador import evaluar
    from strategy.reglas import aplicar_reglas
    patas, theta, senales = _cargar(metodo)
    if patas != ["AMT", "SBAC"]:
        pytest.skip("el evaluador simple de strategy/ solo maneja el par AMT/SBAC")
    diario, _, _ = correr_metodo(senales, patas, theta,
                                 politicas={"sin_rebalanceo": float("inf")})
    burn = a_bool(senales["en_burn_in"])
    for m in MUESTRAS:
        sub = senales[(senales["muestra"] == m).to_numpy() & ~burn]
        d = aplicar_reglas(sub["z"].to_numpy(float), **theta)["d"].to_numpy()
        ref = evaluar(sub[["AMT", "SBAC", "beta"]].assign(d=d))["curva"].to_numpy()
        mio = diario.loc[diario["muestra"] == m, "capital"].to_numpy()
        np.testing.assert_allclose(mio, ref, rtol=0, atol=CFG.capital * TOL)
