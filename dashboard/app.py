# dashboard/app.py
# Dashboard en Streamlit; solo lee artifacts
# Vista ejecutiva (portfolio manager) y vista tecnica (quant / riesgos).

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ART = Path(__file__).resolve().parents[1] / "artifacts"
MUESTRAS = ["train", "test", "val"]
SOMBRA = {"burn-in": "rgba(136,135,128,0.15)", "test": "rgba(216,90,48,0.10)",
          "val": "rgba(29,158,117,0.10)"}
NOMBRE_METODO = {"kalman": "Kalman (pesos dinámicos)", "estatico": "Estático (vector fijo de Johansen)"}
NOMBRE_POLITICA = {"diario": "Rebalanceo diario", "banda_1": "Banda de 1%", "banda_2": "Banda de 2%",
                   "banda_5": "Banda de 5%", "sin_rebalanceo": "Sin rebalanceo (acciones fijas)"}
NOMBRE_MUESTRA = {"completa": "Completa (train + test + val)", "train": "Train", "test": "Test",
                  "val": "Validación"}
AZUL, NARANJA, VERDE, ROJO, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#888780"
COLORES_PATAS = [AZUL, NARANJA, VERDE, "#eda100", "#e87ba4", "#008300", "#6250d6", ROJO]

st.set_page_config(page_title="Lab 03 MyST | Equipo 5", layout="wide")


# Carga de datos
@st.cache_data(show_spinner="Leyendo artifacts/ ...")
def cargar():
    leer = lambda f, **kw: pd.read_csv(ART / f, **kw)
    d = {
        "curvas": leer("bt_curvas.csv", parse_dates=["fecha"]),
        "metricas": leer("bt_metricas.csv"),
        "ops": leer("bt_operaciones.csv", parse_dates=["fecha_entrada", "fecha_salida"]),
        "rebalanceo": leer("bt_rebalanceo.csv"),
        "senales": {m: leer(f"senales_{m}.csv", index_col=0, parse_dates=True)
                    for m in ["kalman", "estatico"]},
        "f1": leer("kalman_filtro_pesos.csv", index_col=0, parse_dates=True),
        "f2": leer("kalman_filtro_spread.csv", index_col=0, parse_dates=True),
        "eig": leer("rolling_eigenvector.csv", index_col=0, parse_dates=True),
        "precios": leer("prices_full.csv", index_col=0, parse_dates=True),
    }
    for nombre in ["bt_config", "theta_optimo", "chosen_group"]:
        d[nombre] = json.load(open(ART / f"{nombre}.json"))
    return d


try:
    D = cargar()
except FileNotFoundError as e:
    st.error(f"Falta un archivo en artifacts/ ({Path(e.filename).name}). "
             "Corre primero el pipeline desde la raíz del repo:  python main.py")
    st.stop()

PATAS = D["bt_config"]["patas"]
POLITICAS = list(D["bt_config"]["politicas"])


# Utilidades
def rangos_muestras(senales):
    """Fechas de inicio y fin de burn-in, train, test y val."""
    burn = senales["en_burn_in"].astype(str).str.lower().eq("true").to_numpy()
    r = {}
    if burn.any():
        r["burn-in"] = (senales.index[burn][0], senales.index[burn][-1])
    for m in MUESTRAS:
        idx = senales.index[(senales["muestra"] == m).to_numpy() & ~burn]
        r[m] = (idx[0], idx[-1])
    return r


def sombrear(fig, rangos, con_burn_in=True, **kw):
    """Sombra para burn-in, test y val (train queda sin sombra)."""
    for nombre, (ini, fin) in rangos.items():
        if nombre == "train" or (nombre == "burn-in" and not con_burn_in):
            continue
        fig.add_vrect(x0=ini, x1=fin, fillcolor=SOMBRA[nombre], line_width=0, layer="below",
                      annotation_text=nombre, annotation_position="top left",
                      annotation_font_size=11, annotation_font_color=GRIS, **kw)
    return fig


def estilo(fig, alto=360, **kw):
    layout = dict(height=alto, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    layout.update(kw)
    fig.update_layout(**layout)
    return fig


def pct(x, dec=2):
    return "n/d" if pd.isna(x) else f"{100 * x:.{dec}f}%"


def num(x, dec=2):
    if pd.isna(x):
        return "n/d"
    return "inf" if np.isinf(x) else f"{x:.{dec}f}"


def dinero(x):
    return "n/d" if pd.isna(x) else (f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}")


# Barra lateral
st.sidebar.title("Lab 03 MyST")
st.sidebar.caption("Equipo 5: Isabela Torres, Andrea Santoyo, Abdon Islas")
vista = st.sidebar.radio("Vista", ["Ejecutiva", "Técnica"],
                         captions=["Portfolio manager / comité", "Quant / gestor de riesgos"])
metodo = st.sidebar.selectbox("Método", ["kalman", "estatico"], format_func=NOMBRE_METODO.get)
politica = st.sidebar.selectbox("Política de rebalanceo", POLITICAS,
                                index=POLITICAS.index(D["bt_config"].get("politica_oficial", POLITICAS[0])),
                                format_func=lambda p: NOMBRE_POLITICA.get(p, p))
theta = D["theta_optimo"][metodo]
st.sidebar.divider()
st.sidebar.markdown(
    f"**Grupo:** {' + '.join(PATAS)}  \n"
    f"**Vector de Johansen (train):** {D['chosen_group']['vector']}  \n"
    f"**Vida media (train):** {D['chosen_group']['vida_media']} días  \n"
    f"**θ\\*:** z_in={theta['z_in']}, z_out={theta['z_out']}, "
    f"z_stop={theta['z_stop']}, h_max={theta['h_max']}")
st.sidebar.caption("Capital inicial de 1,000,000 por muestra; 80% del capital invertido; comisión "
                   "de 0.125% por pata; préstamo de 0.25% anual sobre los cortos. La señal del "
                   "cierre de t se ejecuta al cierre de t+1.")

senales = D["senales"][metodo]
rangos = rangos_muestras(senales)
cur = D["curvas"][(D["curvas"]["metodo"] == metodo) & (D["curvas"]["politica"] == politica)]
met = D["metricas"][(D["metricas"]["metodo"] == metodo) & (D["metricas"]["politica"] == politica)]
ops = D["ops"][(D["ops"]["metodo"] == metodo) & (D["ops"]["politica"] == politica)].copy()
ops["resultado"] = np.where(ops["pnl_neto"] > 0, "ganadora", "perdedora")

st.title(f"Pairs trading {' / '.join(PATAS)}")
st.caption(f"{NOMBRE_METODO[metodo]}  |  {NOMBRE_POLITICA.get(politica, politica)}  |  "
           "datos de artifacts/ (no se vuelve a correr el backtest)")


# VISTA EJECUTIVA
def vista_ejecutiva():
    # ---- Tarjeta de desempeno ----
    st.subheader("Tarjeta de desempeño")
    muestra = st.radio("Muestra", ["completa", "train", "test", "val"], horizontal=True,
                       format_func=NOMBRE_MUESTRA.get, key="muestra_kpi")
    fila = met[met["muestra"] == muestra].iloc[0]
    c = st.columns(6)
    c[0].metric("Sharpe", num(fila["sharpe"]))
    c[1].metric("Sortino", num(fila["sortino"]))
    c[2].metric("Calmar", num(fila["calmar"]))
    c[3].metric("Max drawdown", pct(fila["max_drawdown"]))
    c[4].metric("Win rate", pct(fila["win_rate"], 1))
    c[5].metric("Profit factor", num(fila["profit_factor"]))
    st.caption(f"Retorno total {pct(fila['retorno_total'])}  |  retorno anualizado "
               f"{pct(fila['retorno_anual'])}  |  {int(fila['n_operaciones'])} operaciones  |  "
               f"{int(fila['dias_con_posicion'])} de {int(fila['dias'])} días con posición")
    with st.expander("Todas las muestras en una tabla"):
        t = met.set_index("muestra").loc[["train", "test", "val", "completa"],
                                         ["sharpe", "sortino", "calmar", "max_drawdown", "win_rate",
                                          "profit_factor", "retorno_total", "n_operaciones"]]
        st.dataframe(t.style.format({"sharpe": "{:.2f}", "sortino": "{:.2f}", "calmar": "{:.2f}",
                                     "max_drawdown": "{:.2%}", "win_rate": "{:.1%}",
                                     "profit_factor": "{:.2f}", "retorno_total": "{:.2%}"}))

    #  Curva de capital con drawdown
    st.subheader("Curva de capital con drawdown")
    modo = st.radio("Curva", ["Encadenada", "Por muestra"], horizontal=True,
                    captions=["test arranca con el capital final de train, val con el de test",
                              "cada muestra arranca en 1,000,000"])
    col_cap, col_dd = (("capital_encadenado", "drawdown_encadenado") if modo == "Encadenada"
                       else ("capital", "drawdown"))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for m in MUESTRAS:
        s = cur[cur["muestra"] == m]
        fig.add_trace(go.Scatter(x=s["fecha"], y=s[col_dd], name="drawdown" if m == "train" else None,
                                 showlegend=(m == "train"), fill="tozeroy", line=dict(width=0),
                                 fillcolor="rgba(227,73,72,0.25)", hovertemplate="%{y:.2%}"),
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=s["fecha"], y=s[col_cap], name="capital" if m == "train" else None,
                                 showlegend=(m == "train"), line=dict(color=AZUL, width=2),
                                 hovertemplate="$%{y:,.0f}"), secondary_y=False)
    sombrear(fig, rangos, con_burn_in=False)
    fig.update_yaxes(title_text="capital", tickformat="$,.0f", secondary_y=False)
    dd_min = float(cur[col_dd].min())
    fig.update_yaxes(title_text="drawdown", tickformat=".0%", secondary_y=True,
                     range=[dd_min * 3, 0], showgrid=False)
    st.plotly_chart(estilo(fig, 420))

    #  Atribucion de costos
    st.subheader("Atribución de costos: de P&L bruto a P&L neto")
    a, b = st.columns([2, 1])
    with b:
        muestra_c = st.radio("Muestra", ["completa", "train", "test", "val"],
                             format_func=NOMBRE_MUESTRA.get, key="muestra_costos")
        f = met[met["muestra"] == muestra_c].iloc[0]
        st.metric("Comisiones totales pagadas", dinero(f["comisiones"]))
        st.metric("Costo total de préstamo", dinero(f["prestamo"]))
        st.metric("Comisiones por rebalanceo", dinero(f["comisiones_rebalanceo"]),
                  help=f"{int(f['n_rebalanceos'])} rebalanceos")
        if muestra_c == "completa":
            st.caption("En 'completa' los montos son la suma de las tres muestras, "
                       "cada una con 1,000,000 de capital inicial.")
    with a:
        fig = go.Figure(go.Waterfall(
            measure=["absolute", "relative", "relative", "total"],
            x=["P&L bruto", "Comisiones", "Préstamo", "P&L neto"],
            y=[f["pnl_bruto"], -f["comisiones"], -f["prestamo"], 0],
            text=[dinero(v) for v in [f["pnl_bruto"], -f["comisiones"], -f["prestamo"], f["pnl_neto"]]],
            textposition="outside", connector=dict(line=dict(color=GRIS, width=1)),
            increasing=dict(marker=dict(color=VERDE)), decreasing=dict(marker=dict(color=ROJO)),
            totals=dict(marker=dict(color=AZUL))))
        fig.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(estilo(fig, 380, hovermode="closest", showlegend=False))

    # Bitacora de operaciones
    st.subheader("Bitácora de operaciones")
    f1, f2, f3, f4 = st.columns(4)
    sel_m = f1.multiselect("Periodo (muestra)", MUESTRAS, default=MUESTRAS)
    sel_d = f2.multiselect("Dirección", ["largo spread", "corto spread"],
                           default=["largo spread", "corto spread"])
    sel_r = f3.multiselect("Resultado", ["ganadora", "perdedora"], default=["ganadora", "perdedora"])
    fmin, fmax = ops["fecha_entrada"].min().date(), ops["fecha_salida"].max().date()
    fechas = f4.date_input("Fechas de entrada", (fmin, fmax), min_value=fmin, max_value=fmax)
    filtro = ops["muestra"].isin(sel_m) & ops["direccion"].isin(sel_d) & ops["resultado"].isin(sel_r)
    if isinstance(fechas, (tuple, list)) and len(fechas) == 2:
        filtro &= ops["fecha_entrada"].dt.date.between(fechas[0], fechas[1])
    vis = ops[filtro]
    tabla = vis[["muestra", "operacion", "direccion", "fecha_entrada", "fecha_salida", "dias",
                 "motivo_cierre", "pnl_bruto", "comisiones", "prestamo", "pnl_neto", "retorno",
                 "n_rebalanceos", "resultado"]].assign(retorno=lambda x: 100 * x["retorno"])
    st.dataframe(tabla, hide_index=True, column_config={
        "operacion": st.column_config.NumberColumn("operación"),
        "direccion": st.column_config.TextColumn("dirección"),
        "fecha_entrada": st.column_config.DateColumn("entrada", format="YYYY-MM-DD"),
        "fecha_salida": st.column_config.DateColumn("salida", format="YYYY-MM-DD"),
        "dias": st.column_config.NumberColumn("días"),
        "motivo_cierre": st.column_config.TextColumn("motivo de cierre"),
        "pnl_bruto": st.column_config.NumberColumn("P&L bruto", format="dollar"),
        "comisiones": st.column_config.NumberColumn("comisiones", format="dollar"),
        "prestamo": st.column_config.NumberColumn("préstamo", format="dollar"),
        "pnl_neto": st.column_config.NumberColumn("P&L neto", format="dollar"),
        "retorno": st.column_config.NumberColumn("retorno", format="%.2f%%"),
        "n_rebalanceos": st.column_config.NumberColumn("rebalanceos"),
    })
    st.caption(f"{len(vis)} de {len(ops)} operaciones  |  P&L neto de la selección: "
               f"{dinero(vis['pnl_neto'].sum())}  |  win rate {pct(vis['resultado'].eq('ganadora').mean(), 1)}")

    # Distribucion de retornos por operacion
    st.subheader("Distribución de retornos por operación")
    r = 100 * vis["retorno"].to_numpy(dtype=float)
    if len(r) < 2:
        st.info("Se necesitan al menos 2 operaciones en la selección.")
        return
    bins = max(5, min(20, int(np.sqrt(len(r)) * 2)))
    cuentas, bordes = np.histogram(r, bins=bins)
    centros, ancho = (bordes[:-1] + bordes[1:]) / 2, bordes[1] - bordes[0]
    mu, sd = r.mean(), r.std(ddof=1)
    x = np.linspace(bordes[0] - ancho, bordes[-1] + ancho, 200)
    normal = len(r) * ancho * np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centros, y=cuentas, width=ancho * 0.9, name="operaciones",
                         marker_color=AZUL, hovertemplate="%{x:.2f}%: %{y} operaciones<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=normal, name=f"normal (media {mu:.2f}%, desv. {sd:.2f}%)",
                             line=dict(color=NARANJA, width=2), hoverinfo="skip"))
    fig.update_xaxes(title_text="retorno neto de la operación (%)", ticksuffix="%")
    fig.update_yaxes(title_text="número de operaciones")
    st.plotly_chart(estilo(fig, 340, hovermode="closest"))


# VISTA TECNICA
def vista_tecnica():
    burn_in = st.toggle("Incluir burn-in en las gráficas de los filtros", value=False)
    inicio = senales.index[0] if burn_in else rangos["train"][0]
    sen = senales.loc[inicio:]

    # Precios del grupo
    st.subheader("Precios de los activos del grupo (base 100)")
    px = D["precios"][PATAS].loc[senales.index[0]:]
    fig = go.Figure()
    for i, p in enumerate(PATAS):
        fig.add_trace(go.Scatter(x=px.index, y=100 * px[p] / px[p].iloc[0], name=p,
                                 line=dict(color=COLORES_PATAS[i], width=1.5)))
    st.plotly_chart(estilo(sombrear(fig, rangos)))

    #  Spread con senales
    st.subheader("Spread S_t con entradas y salidas")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sen.index, y=sen["s"], name="spread S_t", line=dict(color=AZUL, width=1.5)))
    fig.add_trace(go.Scatter(x=sen.index, y=sen["mu"], name="nivel esperado (μ)",
                             line=dict(color=GRIS, width=1, dash="dash")))
    s_en = senales["s"]
    for direccion, simbolo, color in [("largo spread", "triangle-up", VERDE),
                                      ("corto spread", "triangle-down", ROJO)]:
        o = ops[ops["direccion"] == direccion]
        fig.add_trace(go.Scatter(x=o["fecha_entrada"], y=s_en.reindex(o["fecha_entrada"]).to_numpy(),
                                 mode="markers", name=f"entrada {direccion}",
                                 marker=dict(symbol=simbolo, size=11, color=color)))
    fig.add_trace(go.Scatter(x=ops["fecha_salida"], y=s_en.reindex(ops["fecha_salida"]).to_numpy(),
                             mode="markers", name="salida", marker=dict(symbol="x", size=9, color="#52514e"),
                             customdata=np.stack([ops["motivo_cierre"], 100 * ops["retorno"]], axis=1),
                             hovertemplate="salida: %{customdata[0]}, retorno %{customdata[1]:.2f}%<extra></extra>"))
    st.caption("Kalman: S_t = log(AMT) - β_t (log(SBAC) - media del burn-in); estático: "
               "S_t = log(AMT) - 0.8631 log(SBAC). Las marcas van en la fecha de ejecución (t+1).")
    st.plotly_chart(estilo(sombrear(fig, rangos, con_burn_in=burn_in), 400))

    # z-score con umbrales
    st.subheader("z-score con umbrales de entrada, salida y stop")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sen.index, y=sen["z"], name="z_t", line=dict(color=AZUL, width=1.2)))
    for nivel, nombre, estilo_linea in [(theta["z_in"], "entrada", "dash"), (theta["z_out"], "salida", "dot"),
                                        (theta["z_stop"], "stop", "solid")]:
        for signo in (1, -1):
            if nivel == 0 and signo == -1:
                continue
            fig.add_hline(y=signo * nivel, line=dict(color=ROJO if nombre == "stop" else GRIS,
                                                    width=1, dash=estilo_linea),
                          annotation_text=f"{nombre} {signo * nivel:+g}", annotation_font_size=10)
    st.plotly_chart(estilo(sombrear(fig, rangos, con_burn_in=burn_in), 380))

    # Vector de pesos
    st.subheader("Evolución del vector de pesos (una línea por activo)")
    cols_w = [f"w_{p}" for p in PATAS]
    if all(c in sen.columns for c in cols_w):
        pesos = sen[cols_w].set_axis(PATAS, axis=1)
    else:
        pesos = pd.DataFrame({PATAS[0]: 1.0, PATAS[1]: -sen["beta"]}, index=sen.index)
    fig = go.Figure()
    for i, p in enumerate(PATAS):
        fig.add_trace(go.Scatter(x=pesos.index, y=pesos[p], name=f"w {p}",
                                 line=dict(color=COLORES_PATAS[i], width=1.5)))
    st.caption(f"Normalizado en {PATAS[0]} = 1. Kalman usa el β previo (β_t|t-1), que no ve el precio del día.")
    st.plotly_chart(estilo(sombrear(fig, rangos, con_burn_in=burn_in)))

    # Eigenvector movil
    st.subheader("Primer eigenvector de Johansen móvil (ventana de 252 días)")
    eig = D["eig"]
    completo = st.toggle("Escala completa", value=False,
                         help="Normalizar dividiendo entre el peso del primer activo explota cuando "
                              "ese peso es casi cero en la ventana.")
    fig = go.Figure()
    for i, p in enumerate([c for c in eig.columns if c in PATAS]):
        fig.add_trace(go.Scatter(x=eig.index, y=eig[p], name=p, mode="lines+markers",
                                 marker=dict(size=3), line=dict(color=COLORES_PATAS[i], width=1)))
    fuera = int((eig.drop(columns=PATAS[0], errors="ignore").abs() > 3).any(axis=1).sum())
    if not completo:
        fig.update_yaxes(range=[-3, 3])
    st.caption(f"{len(eig)} ventanas de {eig.index[0].date()} a {eig.index[-1].date()}; "
               f"{fuera} quedan fuera de [-3, 3].")
    st.plotly_chart(estilo(sombrear(fig, rangos)))

    # Ganancia de Kalman
    st.subheader("Ganancia de Kalman de ambos filtros")
    if metodo != "kalman":
        st.info("Los filtros de Kalman solo intervienen en el método Kalman; se muestran como referencia.")
    f1, f2 = D["f1"].loc[inicio:], D["f2"].loc[inicio:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Filtro 1: pesos (α, β)", "Filtro 2: nivel del spread"])
    fig.add_trace(go.Scatter(x=f1.index, y=f1["K_alpha"], name="K α", line=dict(color=AZUL, width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=f1.index, y=f1["K_beta"], name="K β", line=dict(color=NARANJA, width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=f2.index, y=f2["K2"], name="K filtro 2", line=dict(color=VERDE, width=1.2)), 2, 1)
    sombrear(fig, rangos, con_burn_in=burn_in, row="all", col=1)
    st.caption("K β cambia de signo porque el regresor está centrado: su signo sigue al de "
               "log(SBAC) menos su media del burn-in. En el filtro 2 la ganancia objetivo es 1/83.")
    st.plotly_chart(estilo(fig, 460))

    # Varianza posterior 
    st.subheader("Diagnóstico de convergencia: varianza posterior del estado")
    p2 = (1 - f2["K2"]) * f2["K2"] * f2["sigma"] ** 2          # P2_t|t = (1-K2) K2 S2
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Filtro 1: P de α y de β", "Filtro 2: P del nivel μ"])
    fig.add_trace(go.Scatter(x=f1.index, y=f1["P_alpha"], name="P α", line=dict(color=AZUL, width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=f1.index, y=f1["P_beta"], name="P β", line=dict(color=NARANJA, width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=p2.index, y=p2, name="P filtro 2", line=dict(color=VERDE, width=1.2)), 2, 1)
    fig.update_yaxes(type="log", exponentformat="e", dtick="D2")
    sombrear(fig, rangos, con_burn_in=burn_in, row="all", col=1)
    st.caption("Escala logarítmica. La P del filtro 2 se reconstruye como (1 - K2) K2 σ².")
    st.plotly_chart(estilo(fig, 460))

    # Sharpe movil
    st.subheader("Sharpe móvil de 252 días (curva encadenada)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cur["fecha"], y=cur["sharpe_movil_252"], name="Sharpe 252 días",
                             line=dict(color=AZUL, width=1.5)))
    fig.add_hline(y=0, line=dict(color=GRIS, width=1))
    st.caption("En los tramos sin posición los rendimientos son cero, así que el Sharpe queda plano.")
    st.plotly_chart(estilo(sombrear(fig, rangos, con_burn_in=False)))

    # Costo del rebalanceo
    st.subheader("Costo del rebalanceo contra no rebalancear")
    reb = D["rebalanceo"][D["rebalanceo"]["metodo"] == metodo]
    valores = {"n_rebalanceos": "rebalanceos", "comisiones_rebalanceo": "comisiones de rebalanceo",
               "dif_pnl_neto_vs_sin": "P&L neto vs sin rebalanceo", "sharpe": "Sharpe"}
    t = reb.pivot_table(index="politica", columns="muestra", values=list(valores))
    t = t.reindex([p for p in POLITICAS if p in t.index])
    t = t.reindex(columns=pd.MultiIndex.from_product([list(valores), MUESTRAS]))
    t = t.rename(index=lambda p: NOMBRE_POLITICA.get(p, p), columns=valores, level=0)
    dinero_cols = [c for c in t.columns if c[0] != "Sharpe"]
    st.dataframe(t.style.format("{:,.0f}", subset=dinero_cols)
                 .format("{:.3f}", subset=[c for c in t.columns if c[0] == "Sharpe"]))
    st.caption("P&L neto vs sin rebalanceo: P&L neto de la política menos el de dejar las acciones "
               "fijas. La banda mide la distancia entre la mezcla actual de nocionales y la objetivo "
               "(suma de diferencias absolutas de las proporciones).")


if vista == "Ejecutiva":
    vista_ejecutiva()
else:
    vista_tecnica()