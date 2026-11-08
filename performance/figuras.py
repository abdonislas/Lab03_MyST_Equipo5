# performance/figuras.py
# Figuras del reporte en PNG (docs/figuras/). Solo lee artifacts/: no vuelve a correr nada.
#   fig1 a fig8   las ocho figuras del enunciado (mismo orden que Figuras_Lab03.ipynb)
#   extra_*       atribucion de costos y Sharpe movil
#   apoyo_*       calibracion del filtro 1, autocorrelacion de innovaciones y busqueda de theta
# Ejecutar desde la raiz:  python -m performance.figuras
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ART = Path("artifacts")
SALIDA = Path("docs/figuras")
METODOS = {"kalman": "Kalman", "estatico": "Estático"}
# Paleta categorica validada (orden fijo): azul = Kalman, naranja = estatico, aqua = apoyo
AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#888780"
COLOR = {"kalman": AZUL, "estatico": NARANJA}
TINTA, TINTA_2, SUPERFICIE, REJILLA = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
# Mismo codigo de sombras que el dashboard: gris burn-in, naranja test, verde val
SOMBRA = {"burn_in": "#88878030", "test": "#eb68341a", "val": "#1baf7a1a"}
RAMPA_AZUL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # claro a oscuro
ANCHO = 6.7        # pulgadas = ancho de texto del reporte (17 cm): la letra queda a escala real
PAD_TITULO = 12    # deja lugar para las etiquetas train/test/val sobre el panel de arriba

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2,
    "ytick.color": TINTA_2, "text.color": TINTA, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "font.size": 7.6, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.fontsize": 7, "legend.frameon": False,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0, "axes.linewidth": 0.6,
    "mathtext.fontset": "dejavusans",
})


# ---------------------------------------------------------------- utilidades
def a_bool(serie):
    if serie.dtype == bool:
        return serie.to_numpy()
    return serie.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()


def leer_senales(metodo):
    sen = pd.read_csv(ART / f"senales_{metodo}.csv", index_col=0, parse_dates=True)
    sen["en_burn_in"] = a_bool(sen["en_burn_in"])
    return sen


def leer_csv(nombre, **kw):
    return pd.read_csv(ART / nombre, **kw)


def tramos(sen):
    """Fechas (inicio, fin) del burn-in y de cada muestra."""
    b = sen.index[sen["en_burn_in"]]
    r = {"burn_in": (b[0], b[-1])}
    for m, g in sen.groupby("muestra", sort=False):
        r[m] = (g.index[0], g.index[-1])
    return r


def sombrear(ax, r, etiquetas=False, burn=True):
    for m in (["burn_in"] if burn else []) + ["test", "val"]:
        ax.axvspan(*r[m], color=SOMBRA[m], lw=0, zorder=0)
    if etiquetas:
        for m in ["train", "test", "val"]:
            a, b = r[m]
            ax.text(a + (b - a) / 2, 1.01, m, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7, color=TINTA_2)


def eje_anios(ax, cada=None):
    lo, hi = mdates.num2date(ax.get_xlim()[0]), mdates.num2date(ax.get_xlim()[1])
    cada = cada or (2 if (hi - lo).days > 5 * 365 else 1)
    ax.xaxis.set_major_locator(mdates.YearLocator(cada))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def leyenda_abajo(fig, axes, ncol):
    """Una sola leyenda debajo de la figura con las entradas de todos los paneles (sin repetir)."""
    vistos = {}
    for ax in np.atleast_1d(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            vistos.setdefault(l, h)
    filas = int(np.ceil(len(vistos) / ncol))
    fig.legend(list(vistos.values()), list(vistos.keys()), loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.4)
    return 0.17 * filas / fig.get_figheight()          # fraccion de alto que ocupa la leyenda


def guardar(fig, nombre, abajo=0.0):
    fig.tight_layout(rect=(0, abajo, 1, 1))
    fig.savefig(SALIDA / nombre, dpi=220)
    plt.close(fig)
    return nombre


# ---------------------------------------------------------------- figuras del enunciado
def fig_precios(r):
    px = pd.read_csv(ART / "prices_full.csv", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(2, 1, figsize=(ANCHO, 3.6), sharex=True)
    for i, (ax, activo) in enumerate(zip(axes, ["AMT", "SBAC"])):
        ax.plot(px.index, px[activo], color=TINTA_2, lw=0.9)
        ax.set_ylabel("USD")
        ax.set_title(f"{activo}: cierre ajustado", pad=PAD_TITULO if i == 0 else 4)
        sombrear(ax, r, etiquetas=(i == 0))
    eje_anios(axes[-1])
    return guardar(fig, "fig1_precios.png")


def fig_spread(sen, ops, r):
    fig, axes = plt.subplots(2, 1, figsize=(ANCHO, 3.9), sharex=True)
    for i, (ax, m) in enumerate(zip(axes, METODOS)):
        s = sen[m][~sen[m]["en_burn_in"]]
        ax.plot(s.index, s["s"], color=COLOR[m], lw=0.8, label="spread $s_t$")
        ax.plot(s.index, s["mu"], color=GRIS, lw=0.9, ls="--", label=r"nivel $\mu_{t|t-1}$")
        o = ops[ops["metodo"] == m]
        for direccion, marca, color, nombre in [("largo spread", "^", AQUA, "entra largo spread"),
                                                ("corto spread", "v", TINTA, "entra corto spread")]:
            f = pd.to_datetime(o.loc[o["direccion"] == direccion, "fecha_entrada"])
            ax.scatter(f, s["s"].reindex(f), marker=marca, s=22, color=color, zorder=3, label=nombre,
                       edgecolors=SUPERFICIE, linewidths=0.4)
        f = pd.to_datetime(o["fecha_salida"])
        ax.scatter(f, s["s"].reindex(f), marker="x", s=16, color=GRIS, zorder=3, linewidths=0.9,
                   label="salida")
        ax.set_title(f"{METODOS[m]} (spread de {'log AMT y log SBAC centrado' if m == 'kalman' else 'log-precios con el vector fijo'})",
                     pad=PAD_TITULO if i == 0 else 4)
        ax.set_ylabel("log-precio")
        sombrear(ax, r, etiquetas=(i == 0), burn=False)
    eje_anios(axes[-1])
    return guardar(fig, "fig2_spread_senales.png", leyenda_abajo(fig, axes, 5))


def fig_pesos(sen, f1, beta_fijo, Q, r):
    k = sen["kalman"]
    post = ~k["en_burn_in"]
    w = -k["beta"][post]
    sd = np.sqrt(f1["P_beta"].shift(1) + Q)[post]          # desviacion previa de beta en t
    fig, ax = plt.subplots(figsize=(ANCHO, 2.8))
    ax.fill_between(w.index, w - 2 * sd, w + 2 * sd, color=AZUL, alpha=0.16, lw=0,
                    label=r"banda $\pm 2\sqrt{P_{\beta,t|t-1}}$")
    ax.plot(w.index, w, color=AZUL, lw=1.0, label=r"Kalman: $-\beta_{t|t-1}$")
    ax.axhline(-beta_fijo, color=NARANJA, lw=1.0, ls="--",
               label=f"estático (Johansen en train): {-beta_fijo:.4f}")
    ax.set_ylabel("peso de SBAC (AMT = 1)")
    sombrear(ax, r, etiquetas=True, burn=False)
    eje_anios(ax)
    return guardar(fig, "fig3_pesos.png", leyenda_abajo(fig, ax, 3))


def fig_johansen(eig, beta_fijo):
    w = eig["SBAC"]
    lim = 3.0
    fuera = w[w.abs() > lim]
    fig, ax = plt.subplots(figsize=(ANCHO, 2.8))
    ax.plot(eig.index, w.clip(-lim, lim), color=AZUL, lw=0.9, label="peso de SBAC en cada ventana (AMT = 1)")
    ax.scatter(fuera.index, np.sign(fuera) * lim, marker="D", s=10, color=TINTA, zorder=3,
               label=f"fuera de escala: {len(fuera)} ventanas")
    ax.axhline(-beta_fijo, color=NARANJA, lw=1.0, ls="--", label=f"vector de todo el train: {-beta_fijo:.4f}")
    ax.axhline(0, color=GRIS, lw=0.7)
    ax.set_ylim(-lim - 0.3, lim + 0.3)
    ax.set_ylabel("peso de SBAC")
    eje_anios(ax, 1)
    return guardar(fig, "fig4_johansen_movil.png", leyenda_abajo(fig, ax, 3))


def fig_ganancia(sen, f1, f2, K_obj, r):
    burn = sen["kalman"]["en_burn_in"]
    a, b = f1[~burn], f2[~burn]
    fig, axes = plt.subplots(2, 1, figsize=(ANCHO, 4.1), sharex=True)
    axes[0].plot(a.index, a["K_alpha"], color=AZUL, lw=0.8, label=r"filtro 1: $K_{\alpha,t}$")
    axes[0].plot(a.index, a["K_beta"], color=AQUA, lw=0.8, label=r"filtro 1: $K_{\beta,t}$")
    axes[0].axhline(0, color=GRIS, lw=0.6)
    axes[0].set_title(r"Filtro 1: ganancia de $(\alpha_t, \beta_t)$", pad=PAD_TITULO)
    axes[1].plot(b.index, b["K2"], color=TINTA_2, lw=0.9, label=r"filtro 2: $K2_t$")
    axes[1].axhline(K_obj, color=NARANJA, lw=1.0, ls="--", label=f"objetivo del filtro 2: 1/L = {K_obj:.4f}")
    axes[1].set_title(r"Filtro 2: ganancia del nivel $\mu_t$", pad=4)
    for i, ax in enumerate(axes):
        sombrear(ax, r, etiquetas=(i == 0), burn=False)
    eje_anios(axes[-1])
    return guardar(fig, "fig5_ganancia.png", leyenda_abajo(fig, axes, 4))


def fig_zscore(sen, theta, r):
    fig, axes = plt.subplots(2, 1, figsize=(ANCHO, 3.9), sharex=True)
    for i, (ax, m) in enumerate(zip(axes, METODOS)):
        s = sen[m][~sen[m]["en_burn_in"]]
        t = theta[m]
        ax.plot(s.index, s["z"].clip(-5, 5), color=COLOR[m], lw=0.7, label="$z_t$ (Kalman o estático)")
        for nivel, estilo, nombre in [(t["z_in"], "--", r"$\pm z_{in}$ (entrada)"),
                                      (t["z_out"], ":", r"$\pm z_{out}$ (salida)"),
                                      (t["z_stop"], "-", r"$\pm z_{stop}$ (stop)")]:
            for signo in (1, -1):
                ax.axhline(signo * nivel, color=TINTA_2, lw=0.8, ls=estilo, label=nombre if signo == 1 else None)
        ax.set_ylim(-5, 5)
        ax.set_title(f"{METODOS[m]}: z_in = {t['z_in']:.2f}, z_out = {t['z_out']:.2f}, "
                     f"z_stop = {t['z_stop']:.2f}, h_max = {t['h_max']}", pad=PAD_TITULO if i == 0 else 4)
        sombrear(ax, r, etiquetas=(i == 0), burn=False)
    eje_anios(axes[-1])
    return guardar(fig, "fig6_zscore.png", leyenda_abajo(fig, axes[:1], 4))


def fig_capital(cur):
    fig, axes = plt.subplots(1, 3, figsize=(ANCHO, 2.6), sharey=True,
                             gridspec_kw={"width_ratios": [3, 1, 1]})
    for ax, mu in zip(axes, ["train", "test", "val"]):
        for m in METODOS:
            g = cur[(cur["metodo"] == m) & (cur["muestra"] == mu)].sort_values("fecha")
            ax.plot(g["fecha"], g["capital"] / 1e6, color=COLOR[m], lw=1.0, label=METODOS[m])
        ax.axhline(1.0, color=GRIS, lw=0.6)
        ax.set_title(mu)
        eje_anios(ax, 1)
    axes[0].set_ylabel("capital (millones de USD)")
    axes[0].legend(loc="upper left")
    return guardar(fig, "fig7_capital.png")


def fig_retornos(ops):
    fig, axes = plt.subplots(1, 2, figsize=(ANCHO, 2.7), sharey=True)
    todos = 100 * ops["retorno"]
    bordes = np.arange(np.floor(todos.min() * 2) / 2, np.ceil(todos.max() * 2) / 2 + 0.5, 0.5)
    for ax, m in zip(axes, METODOS):
        r = 100 * ops.loc[ops["metodo"] == m, "retorno"]
        ax.hist(r, bins=bordes, color=COLOR[m], edgecolor=SUPERFICIE, linewidth=0.8,
                label="operaciones (Kalman o estático)")
        x = np.linspace(bordes[0], bordes[-1], 300)
        dens = np.exp(-0.5 * ((x - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi))
        ax.plot(x, dens * len(r) * 0.5, color=TINTA_2, lw=0.9, ls="--",
                label="normal con la misma media y desviación")
        ax.axvline(0, color=GRIS, lw=0.7)
        ax.set_title(f"{METODOS[m]}: {len(r)} operaciones")
        ax.text(0.98, 0.96, f"media {r.mean():.2f}%\nmediana {r.median():.2f}%\nganadoras {100 * (r > 0).mean():.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=TINTA_2)
        ax.set_xlabel("retorno neto por operación (%)")
    axes[0].set_ylabel("número de operaciones")
    return guardar(fig, "fig8_retornos_operacion.png", leyenda_abajo(fig, axes[:1], 2))


# ---------------------------------------------------------------- extras
def fig_costos(met):
    fig, axes = plt.subplots(1, 2, figsize=(ANCHO, 2.6), sharey=True)
    for ax, m in zip(axes, METODOS):
        f = met[(met["metodo"] == m) & (met["muestra"] == "completa")].iloc[0]
        pasos = [("P&L bruto", f["pnl_bruto"]), ("comisiones", -f["comisiones"]), ("préstamo", -f["prestamo"])]
        nivel = 0.0
        for i, (nombre, v) in enumerate(pasos):
            base = nivel if v >= 0 else nivel + v
            ax.bar(i, abs(v) / 1e3, bottom=base / 1e3, color=COLOR[m] if i == 0 else GRIS, width=0.6)
            ax.text(i, (nivel + max(v, 0)) / 1e3, f"{v / 1e3:+,.1f}", ha="center", va="bottom", fontsize=7)
            nivel += v
        ax.bar(3, f["pnl_neto"] / 1e3, color=COLOR[m], width=0.6, hatch="////", edgecolor=SUPERFICIE)
        ax.text(3, f["pnl_neto"] / 1e3, f"{f['pnl_neto'] / 1e3:,.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(4), [p[0] for p in pasos] + ["P&L neto"])
        ax.set_title(f"{METODOS[m]}: train + test + val")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("miles de USD")
    return guardar(fig, "extra_costos.png")


def fig_sharpe_movil(cur, r):
    fig, ax = plt.subplots(figsize=(ANCHO, 2.6))
    for m in METODOS:
        g = cur[cur["metodo"] == m].sort_values("fecha")
        ax.plot(g["fecha"], g["sharpe_movil_252"], color=COLOR[m], lw=1.0, label=METODOS[m])
    ax.axhline(0, color=GRIS, lw=0.7)
    ax.set_ylabel("Sharpe anualizado, 252 días")
    sombrear(ax, r, etiquetas=True)
    eje_anios(ax)
    return guardar(fig, "extra_sharpe_movil.png", leyenda_abajo(fig, ax, 2))


# ---------------------------------------------------------------- apoyo
def fig_calibracion(cal, mom, delta_final):
    fig, ax = plt.subplots(figsize=(ANCHO, 3.2))
    for color, (d, g) in zip(RAMPA_AZUL, cal.groupby("delta")):
        permitido = bool(a_bool(g["permitido"])[0])
        ax.plot(g["R"], g["log_verosimilitud"], color=color, lw=1.1, ls="-" if permitido else "--",
                marker="o", ms=2.5, label=f"δ = {potencia(d)}" + ("" if permitido else " (no permitido)"))
    best = cal[a_bool(cal["max_LL_permitido"])].iloc[0]
    ax.scatter(best["R"], best["log_verosimilitud"], marker="*", s=90, color=TINTA, zorder=4,
               label=r"máxima verosimilitud con δ ≤ $10^{-5}$")
    for _, f in mom.iterrows():
        elegido = np.isclose(f["delta"], delta_final)
        ax.scatter(f["R_momentos"], f["LL"], marker="s", s=34 if elegido else 22, zorder=4,
                   color=NARANJA if elegido else SUPERFICIE, edgecolors=NARANJA, linewidths=1.0,
                   label=(f"R por momentos con el δ usado ({potencia(f['delta'])})" if elegido
                          else f"R por momentos con δ = {potencia(f['delta'])}"))
    ax.set_xscale("log")
    tope = cal.groupby("delta")["log_verosimilitud"].max()
    lo = tope.min() - 0.08 * (tope.max() - tope.min())
    ax.set_ylim(lo, tope.max() + 0.06 * (tope.max() - lo))
    ax.set_xlabel("R (escala logarítmica)")
    ax.set_ylabel("log-verosimilitud en train")
    return guardar(fig, "apoyo_calibracion_filtro1.png", leyenda_abajo(fig, ax, 3))


def potencia(x):
    return f"$10^{{{int(round(np.log10(x)))}}}$"


def acf(u, k):
    u = np.asarray(u, float) - np.mean(u)
    return np.array([np.sum(u[j:] * u[:-j]) / np.sum(u * u) for j in range(1, k + 1)])


def fig_innovaciones(sen, f1, f2):
    k = sen["kalman"]
    tr = ((k["muestra"] == "train") & ~k["en_burn_in"]).to_numpy()
    series = {"Filtro 1: $e_t/\\sqrt{S_t}$": (f1["e"] / np.sqrt(f1["S"])).to_numpy()[tr],
              "Filtro 2: $z_t = e2_t/\\sqrt{S2_t}$": f2["z"].to_numpy()[tr]}
    fig, axes = plt.subplots(1, 2, figsize=(ANCHO, 2.4), sharey=True)
    n_rez = 20
    for ax, (nombre, u) in zip(axes, series.items()):
        banda = 1.96 / np.sqrt(len(u))
        ax.bar(np.arange(1, n_rez + 1), acf(u, n_rez), width=0.6, color=AZUL, label="autocorrelación")
        ax.axhspan(-banda, banda, color="#2a78d633", lw=0, label="banda de ruido blanco al 95%")
        ax.axhline(0, color=GRIS, lw=0.6)
        ax.set_title(nombre)
        ax.set_xlabel("rezago (días)")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("autocorrelación en train")
    return guardar(fig, "apoyo_innovaciones_acf.png", leyenda_abajo(fig, axes[:1], 2))


def fig_optimizacion(theta):
    fig, axes = plt.subplots(1, 2, figsize=(ANCHO, 2.8), sharey=True)
    for ax, m in zip(axes, METODOS):
        g = leer_csv(f"grid_{m}.csv")
        fact = a_bool(g["factible"])
        i = np.arange(1, len(g) + 1)
        ax.scatter(i[fact], g["sharpe"][fact], s=9, color=COLOR[m], label="trial con 10 o más operaciones")
        ax.scatter(i[~fact], g["sharpe"][~fact], s=9, marker="x", color=GRIS, linewidths=0.8,
                   label="trial con menos de 10 (penalizado)")
        mejor = pd.Series(np.where(fact, g["sharpe"], -np.inf)).cummax().replace(-np.inf, np.nan)
        ax.step(i, mejor, where="post", color=TINTA, lw=0.9, label="mejor Sharpe hasta ese trial")
        j = int(np.nanargmax(np.where(fact, g["sharpe"], np.nan)))
        ax.scatter(j + 1, g["sharpe"].iloc[j], marker="*", s=80, color=TINTA, zorder=4, label="θ* elegido")
        ax.set_title(f"{METODOS[m]}: θ* en el trial {j + 1} de {len(g)}")
        ax.set_xlabel("trial de Optuna (TPE)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Sharpe neto en train")
    return guardar(fig, "apoyo_optimizacion_theta.png", leyenda_abajo(fig, axes, 4))


# ---------------------------------------------------------------- main
def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    chosen = json.load(open(ART / "chosen_group.json"))
    kp = json.load(open(ART / "kalman_params.json"))
    theta = json.load(open(ART / "theta_optimo.json"))
    oficial = json.load(open(ART / "bt_config.json")).get("politica_oficial", "diario")
    sen = {m: leer_senales(m) for m in METODOS}
    r = tramos(sen["kalman"])
    f1 = leer_csv("kalman_filtro_pesos.csv", index_col=0, parse_dates=True)
    f2 = leer_csv("kalman_filtro_spread.csv", index_col=0, parse_dates=True)
    eig = leer_csv("rolling_eigenvector.csv", index_col=0, parse_dates=True)
    ops = leer_csv("bt_operaciones.csv")
    ops = ops[ops["politica"] == oficial]
    cur = leer_csv("bt_curvas.csv", parse_dates=["fecha"])
    cur = cur[cur["politica"] == oficial]
    met = leer_csv("bt_metricas.csv")
    met = met[met["politica"] == oficial]
    beta_fijo = -chosen["vector"][1]
    hechas = [
        fig_precios(r), fig_spread(sen, ops, r), fig_pesos(sen, f1, beta_fijo, kp["Q"][1][1], r),
        fig_johansen(eig, beta_fijo), fig_ganancia(sen, f1, f2, kp["K_obj"], r), fig_zscore(sen, theta, r),
        fig_capital(cur), fig_retornos(ops), fig_costos(met), fig_sharpe_movil(cur, r),
        fig_calibracion(leer_csv("kalman_calibracion.csv"), leer_csv("kalman_calibracion_momentos.csv"),
                        kp["delta"]),
        fig_innovaciones(sen, f1, f2), fig_optimizacion(theta),
    ]
    print(f"{len(hechas)} figuras en {SALIDA}/: " + ", ".join(hechas))


if __name__ == "__main__":
    main()
