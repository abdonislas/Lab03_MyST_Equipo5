# performance/reporte.py
# Genera docs/reporte.pdf. Todos los numeros salen de artifacts/ y las figuras de docs/figuras/
# (se regeneran al inicio), asi el reporte siempre coincide con la ultima corrida del pipeline.
# Los ejemplos numericos de los filtros se calculan con las mismas funciones del repo.
# Ejecutar desde la raiz:  python -m performance.reporte
import ast
import json
import os
import re
import warnings
from datetime import date
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from statsmodels.stats.diagnostic import acorr_ljungbox

from cointegration.screening import corr_movil_mediana
from kalman.filtro_pesos import correr_filtro_pesos
from kalman.filtro_spread import correr_filtro_spread, parametros_memoria
from performance import figuras

ART, FIG, SALIDA = Path("artifacts"), Path("docs/figuras"), Path("docs/reporte.pdf")
ANCHO = 17.0 * cm
TINTA, TINTA_2, LINEA, FONDO = "#0b0b0b", "#52514e", "#c9c8c2", "#f3f2ee"
METODOS = {"kalman": "Kalman", "estatico": "Estático"}
MUESTRAS = ["train", "test", "val"]
EVENTO = "2026-04-02"      # alza de SBAC por reportes de posible venta (data_jumps.csv)

# Corrida de control del 2026-11-08: python main.py en otra computadora (Linux, Python 3.13) con las
# versiones de requirements.txt y los mismos precios. Los trials de Optuna coinciden con los del commit
# hasta el 42 (Kalman) y el 57 (estatico); despues el TPE propone otros puntos porque el Sharpe difiere
# en el orden de 1e-14 por redondeo, y el theta* final cambia. Solo se reporta si theta_optimo.json es
# el mismo con el que se hizo la comparacion (si el pipeline cambia, hay que volver a medirla).
OTRA_CORRIDA = {
    "referencia": {"kalman": 1.9933704485338453, "estatico": 1.9859667813961646},
    "kalman": {"theta": (1.8273, 0.3580, 3.2290, 94), "extra": "δ = 10<super>−6</super>",
               "sharpe": (0.643, -0.183, 0.505), "coinciden_hasta": 42},
    "estatico": {"theta": (1.8672, 0.2068, 3.4542, 120), "extra": "L = 93",
                 "sharpe": (0.751, -0.056, 1.071), "coinciden_hasta": 57},
}

# ---------------------------------------------------------------- tipografia
_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for nombre, archivo in [("Serif", "STIXGeneral.ttf"), ("Serif-Bold", "STIXGeneralBol.ttf"),
                        ("Serif-Italic", "STIXGeneralItalic.ttf"), ("Serif-BoldItalic", "STIXGeneralBolIta.ttf"),
                        ("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                        ("Sans-Oblique", "DejaVuSans-Oblique.ttf")]:
    pdfmetrics.registerFont(TTFont(nombre, str(_ttf / archivo)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic",
                              boldItalic="Serif-BoldItalic")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique",
                              boldItalic="Sans-Bold")

E = {
    "titulo": ParagraphStyle("titulo", fontName="Sans-Bold", fontSize=16.5, leading=21, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName="Serif", fontSize=10.5, leading=14, textColor=TINTA_2),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=12.5, leading=16, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=10, leading=13.5, spaceBefore=9, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Sans-Bold", fontSize=8.8, leading=12, spaceBefore=6, spaceAfter=3,
                         textColor=TINTA_2),
    "p": ParagraphStyle("p", fontName="Serif", fontSize=10.2, leading=13.8, alignment=TA_JUSTIFY, spaceAfter=5),
    "item": ParagraphStyle("item", fontName="Serif", fontSize=10.2, leading=13.6, leftIndent=13, bulletIndent=3,
                           spaceAfter=3, alignment=TA_JUSTIFY),
    "pie": ParagraphStyle("pie", fontName="Serif", fontSize=9, leading=11.6, textColor=TINTA_2, spaceAfter=9,
                          spaceBefore=2, alignment=TA_JUSTIFY),
    "leyenda": ParagraphStyle("leyenda", fontName="Serif", fontSize=9, leading=11.6, textColor=TINTA_2,
                              spaceBefore=0, spaceAfter=0),
    "celda": ParagraphStyle("celda", fontName="Sans", fontSize=7.3, leading=9.1),
    "celda_d": ParagraphStyle("celda_d", fontName="Sans", fontSize=7.3, leading=9.1, alignment=TA_RIGHT),
    "celda_b": ParagraphStyle("celda_b", fontName="Sans-Bold", fontSize=7.3, leading=9.1),
    "celda_bd": ParagraphStyle("celda_bd", fontName="Sans-Bold", fontSize=7.3, leading=9.1, alignment=TA_RIGHT),
    "texto": ParagraphStyle("texto", fontName="Serif", fontSize=9.3, leading=11.8),
    "texto_b": ParagraphStyle("texto_b", fontName="Serif-Bold", fontSize=9.3, leading=11.8),
}


# ---------------------------------------------------------------- utilidades
def leer(nombre, **kw):
    return pd.read_csv(ART / nombre, **kw)


def jleer(nombre):
    return json.load(open(ART / nombre))


def a_bool(serie):
    return figuras.a_bool(serie)


def esc(texto):
    """& literal -> &amp; (P&L, AT&T) sin tocar las entidades que ya vienen escritas."""
    return re.sub(r"&(?!(amp|lt|gt|quot|#\d+);)", "&amp;", str(texto))


def num(x, d=2, signo=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    if isinstance(x, float) and np.isinf(x):
        return "inf"
    s = f"{x:+,.{d}f}" if signo else f"{x:,.{d}f}"
    return s.replace("-", "−")


def pct(x, d=1, signo=False):
    return "n/d" if pd.isna(x) else num(100 * x, d, signo) + "%"


def cient(x, d=3):
    """1.234e-05 -> 1.234×10⁻⁵ con marcado de reportlab."""
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}×10<super>{int(e)}</super>".replace("-", "−")


def P(texto, estilo="p"):
    return Paragraph(esc(texto), E[estilo])


def items(lista):
    return [Paragraph(esc(t), E["item"], bulletText="•") for t in lista]


def ecuacion(tex, tam=10.5):
    """Ecuacion centrada, renderizada con mathtext (STIX) a 300 dpi y escalada a tamano real."""
    with plt.rc_context({"mathtext.fontset": "stix"}):
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=tam)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=True)
        plt.close(fig)
    datos = buf.getvalue()
    w, h = ImageReader(BytesIO(datos)).getSize()
    escala = min(1.0, ANCHO / (w * 72 / 300))          # si no cabe en el ancho de texto, se reduce
    img = Image(BytesIO(datos), width=escala * w * 72 / 300, height=escala * h * 72 / 300)
    img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 2), img, Spacer(1, 4)])


def tabla(filas, anchos, derecha=(), texto=False, resaltar=()):
    """Especificacion de una tabla con encabezado; Numerador.tabla la arma con su leyenda.
    derecha: columnas numericas alineadas a la derecha. texto=True usa letra serif (tablas con prosa).
    resaltar: filas en negritas."""
    return {"filas": filas, "anchos": anchos, "derecha": derecha, "texto": texto, "resaltar": resaltar}


def armar_tabla(spec, leyenda):
    """La leyenda va como primera fila de la tabla: no se separa del encabezado y se repite si la tabla
    continua en la pagina siguiente."""
    filas, texto = spec["filas"], spec["texto"]
    filas_p = []
    for i, fila in enumerate(filas):
        celdas = []
        for j, c in enumerate(fila):
            if texto:
                est = "texto_b" if (i == 0 or j == 0 or i in spec["resaltar"]) else "texto"
            elif i == 0 or i in spec["resaltar"]:
                est = "celda_bd" if j in spec["derecha"] else "celda_b"
            else:
                est = "celda_d" if j in spec["derecha"] else "celda"
            celdas.append(Paragraph(esc(c), E[est]))
        filas_p.append(celdas)
    n_col = len(filas[0])
    datos = [[Paragraph(esc(leyenda), E["leyenda"])] + [""] * (n_col - 1)] + filas_p
    t = Table(datos, colWidths=spec["anchos"], repeatRows=2, hAlign="LEFT")
    pad = 3 if texto else 2.2
    estilo = [("SPAN", (0, 0), (-1, 0)),
              ("LEFTPADDING", (0, 0), (-1, 0), 0), ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
              ("TOPPADDING", (0, 0), (-1, 0), 0),
              ("LINEABOVE", (0, 1), (-1, 1), 0.8, colors.HexColor(TINTA)),
              ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.HexColor(TINTA)),
              ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor(TINTA)),
              ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("TOPPADDING", (0, 1), (-1, -1), pad), ("BOTTOMPADDING", (0, 1), (-1, -1), pad),
              ("LEFTPADDING", (0, 1), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    for i in range(3, len(datos), 2):
        estilo.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(FONDO)))
    t.setStyle(TableStyle(estilo))
    return t


def H(texto, nivel="h1"):
    """Titulo de seccion: si queda poco espacio en la pagina, empieza en la siguiente."""
    return [CondPageBreak({"h1": 6.5, "h2": 4.5, "h3": 3.5}[nivel] * cm), P(texto, nivel)]


def pot10(x):
    """1e-06 -> 10^-6 con marcado de reportlab (para delta)."""
    e = int(round(np.log10(x)))
    return f"10<super>{e}</super>".replace("-", "−")


class Numerador:
    """Numera figuras y tablas en orden de aparicion y guarda que figura del enunciado es cada una."""

    def __init__(self):
        self.f, self.t, self.mapa = 0, 0, []

    def figura(self, archivo, pie, enunciado=None, ancho=ANCHO):
        self.f += 1
        if enunciado:
            self.mapa.append((enunciado, self.f, archivo))
        ruta = FIG / archivo
        w, h = ImageReader(str(ruta)).getSize()
        return KeepTogether([Image(str(ruta), width=ancho, height=ancho * h / w),
                             P(f"<b>Figura {self.f}.</b> {pie}", "pie")])

    def tabla(self, leyenda, spec):
        self.t += 1
        t = armar_tabla(spec, f"<b>Tabla {self.t}.</b> {leyenda}")
        _, alto = t.wrap(ANCHO, 100 * cm)
        if alto < 8 * cm:                            # tabla chica: no se parte entre paginas
            return [KeepTogether([t]), Spacer(1, 8)]
        return [CondPageBreak(4.5 * cm), t, Spacer(1, 8)]   # grande: se parte y repite encabezado


def pie_pagina(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7)
    canvas.setFillColor(colors.HexColor(TINTA_2))
    canvas.drawString(2.3 * cm, 1.25 * cm, "Lab 03 MyST · Equipo 5 · Arbitraje estadístico con AMT y SBAC")
    canvas.drawRightString(19.3 * cm, 1.25 * cm, str(doc.page))
    canvas.restoreState()


# ---------------------------------------------------------------- calculos auxiliares
def diagnostico_innovaciones(sen, f1, f2):
    """Media, varianza, rho1, Ljung-Box y curtosis de las innovaciones estandarizadas."""
    burn = a_bool(sen["en_burn_in"])
    series = {"Filtro 1": (f1["e"] / np.sqrt(f1["S"])).to_numpy(), "Filtro 2": f2["z"].to_numpy()}
    filas = []
    for nombre, u in series.items():
        for m in MUESTRAS:
            x = u[(sen["muestra"].to_numpy() == m) & ~burn]
            lb = acorr_ljungbox(x, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
            filas.append({"filtro": nombre, "muestra": m, "n": len(x), "media": x.mean(),
                          "var": x.var(ddof=1), "rho1": np.corrcoef(x[:-1], x[1:])[0, 1], "lb10": lb,
                          "curtosis": pd.Series(x).kurt()})
    return pd.DataFrame(filas)


def atribucion_evento(diario, fecha=EVENTO):
    """Parte del P&L de val que se gana el dia del evento, por metodo."""
    out = {}
    d = diario[diario["muestra"] == "val"]
    for m, g in d.groupby("metodo"):
        g = g.sort_values("fecha")
        pnl = g["capital"].diff().fillna(0.0)
        total = g["capital"].iloc[-1] - g["capital"].iloc[0]
        dia = float(pnl[g["fecha"] == fecha].sum())
        out[m] = {"total": total, "dia": dia, "frac": dia / total if total else np.nan,
                  "ret_sin": (total - dia) / g["capital"].iloc[0]}
    return out


def operacion_ejemplo(sen, ops, diario, fills):
    """Primera operacion del Kalman en train que cierra por la regla de salida (si no hay, la primera)."""
    o = ops[(ops["metodo"] == "kalman") & (ops["muestra"] == "train")].copy()
    cand = o[o["motivo_cierre"] == "salida"]
    o = (cand if len(cand) else o).iloc[0]
    fe, fs = pd.Timestamp(o["fecha_entrada"]), pd.Timestamp(o["fecha_salida"])
    k = sen
    i_e, i_s = k.index.get_loc(fe), k.index.get_loc(fs)
    senal_e, senal_s = k.iloc[i_e - 1], k.iloc[i_s - 1]
    d = diario[(diario["metodo"] == "kalman") & (diario["muestra"] == "train")].set_index("fecha")
    f = fills[(fills["metodo"] == "kalman") & (fills["muestra"] == "train") & (fills["operacion"] == o["operacion"])]
    entrada = f[f["tipo"] == "entrada"].set_index("pata")
    salida = f[f["tipo"].isin(["salida", "liquidacion"])].set_index("pata")
    reb = f[f["tipo"] == "rebalanceo"]
    dia1 = k.index[i_e + 1]
    return {"op": o, "fecha_senal": k.index[i_e - 1], "senal": senal_e, "fecha_senal_salida": k.index[i_s - 1],
            "senal_salida": senal_s, "entrada": entrada, "salida": salida, "reb": reb, "dia1": dia1,
            "diario_dia1": d.loc[dia1], "beta_dia1": float(k.iloc[i_e]["beta"]),
            "V": float(o["capital_entrada"]), "dias_naturales_1": (dia1 - fe).days}


# ---------------------------------------------------------------- contenido
def construir():
    N = Numerador()
    chosen, kp, th = jleer("chosen_group.json"), jleer("kalman_params.json"), jleer("theta_optimo.json")
    cfg = jleer("bt_config.json")
    oficial = cfg.get("politica_oficial", "diario")
    v2, v1 = leer("screening_all_groups.csv"), leer("universo_v1/screening_all_groups.csv")
    eg, eg1 = leer("engle_granger_pairs.csv"), leer("universo_v1/engle_granger_pairs.csv")
    fp, oos, rob = leer("false_positives.csv"), leer("oos_check.csv"), leer("robustez_oos.csv")
    orden, saltos, huecos = leer("integration_order.csv"), leer("data_jumps.csv"), leer("data_gaps.csv")
    eig = leer("rolling_eigenvector.csv", index_col=0, parse_dates=True)
    valpha = leer("vecm_alpha.csv").set_index("ticker")
    cal, mom = leer("kalman_calibracion.csv"), leer("kalman_calibracion_momentos.csv")
    gk, ge = leer("grid_kalman.csv"), leer("grid_estatico.csv")
    met = leer("bt_metricas.csv")
    met = met[met["politica"] == oficial]
    reb = leer("bt_rebalanceo.csv")
    ops = leer("bt_operaciones.csv")
    ops = ops[ops["politica"] == oficial]
    diario = leer("bt_diario.csv", parse_dates=["fecha"])
    diario = diario[diario["politica"] == oficial]
    fills = leer("bt_fills.csv", parse_dates=["fecha"])
    fills = fills[fills["politica"] == oficial]
    sen = figuras.leer_senales("kalman")
    f1 = leer("kalman_filtro_pesos.csv", index_col=0, parse_dates=True)
    f2 = leer("kalman_filtro_spread.csv", index_col=0, parse_dates=True)
    px1 = leer("universo_v1/prices_train.csv", index_col=0, parse_dates=True)

    m_of = {(r["metodo"], r["muestra"]): r for _, r in met.iterrows()}
    o = {r["muestra"]: r for _, r in oos.iterrows()}
    rb = {r["sub_muestra"]: r for _, r in rob.iterrows()}
    fpt = {r["prueba"]: r for _, r in fp.iterrows()}
    par = eg[eg["par"] == "AMT-SBAC"].iloc[0]
    grupo = chosen["grupo"].replace("+", " y ")
    b1 = -chosen["vector"][1]
    burn = a_bool(sen["en_burn_in"])
    fechas_burn = sen.index[burn]
    rangos = {m: (g.index[0].date(), g.index[-1].date(), len(g)) for m, g in sen.groupby("muestra", sort=False)}
    ev = atribucion_evento(diario)
    g_ev = diario[(diario["metodo"] == "kalman") & (diario["muestra"] == "val")].sort_values("fecha").reset_index(drop=True)
    i_ev = g_ev.index[g_ev["fecha"] == EVENTO][0]
    lado_evento = {1: "largo el spread (largo AMT, corto SBAC)", -1: "corto el spread (corto AMT, largo SBAC)",
                   0: "fuera de posición"}[int(g_ev.loc[i_ev - 1, "d"])]
    diag = diagnostico_innovaciones(sen, f1, f2)
    dg = {(r["filtro"], r["muestra"]): r for _, r in diag.iterrows()}
    K_obj, q, lam = parametros_memoria(kp["L"])
    tk, te = th["kalman"], th["estatico"]
    ex = operacion_ejemplo(sen, ops, diario, fills)
    s = []

    # ============================================================== portada y resumen
    s += [P("Arbitraje estadístico con cointegración y filtros de Kalman: AMT y SBAC", "titulo"),
          P("Lab 03 · Microestructura y Sistemas de Trading · ITESO", "sub"),
          P("Equipo 5: Isabela Torres-Septien Uribe, Andrea Santoyo Vega y Abdon Islas León", "sub"),
          P(f"Generado el {date.today().isoformat()} con <i>python -m performance.reporte</i>; todas las cifras "
            f"salen de <i>artifacts/</i>, salvo la corrida de control de la sección 6.", "sub"), Spacer(1, 10)]
    s += [*H("Resumen", "h1")]
    s += items([
        f"Universo de seis acciones de torres y telefónicas de EE. UU. (AMT, CCI, SBAC, VZ, T y TMUS), precios "
        f"diarios del {rangos['train'][0]} al {rangos['val'][1]} con partición cronológica 60/20/20. Se probaron los "
        f"{len(v2)} grupos de 2 a 6 activos, más {len(v1)} de un universo anterior. Pasan el embudo "
        f"{int((v2['etapa_caida'] == 'PASA').sum())} grupos; se eligió <b>{grupo}</b>, el único sin bandera roja: vector "
        f"(1, {num(-b1, 4)}), ADF del spread p = {num(chosen['ADF_spread_p'], 3)} y vida media de "
        f"{chosen['vida_media']} días en train.",
        "La señal sale de dos filtros de Kalman: el filtro 1 estima los pesos (α<sub>t</sub>, β<sub>t</sub>) y el "
        "filtro 2 el nivel y la escala del spread para obtener un z-score causal. La línea base (estático) usa el "
        "vector fijo de Johansen y un z-score móvil. Los dos usan la misma política de umbrales, con θ elegido por "
        "optimización bayesiana en train y congelado.",
        f"Con rebalanceo diario, el Sharpe neto en train, test y val es {num(m_of[('kalman', 'train')]['sharpe'])}, "
        f"{num(m_of[('kalman', 'test')]['sharpe'])} y {num(m_of[('kalman', 'val')]['sharpe'])} para Kalman, y "
        f"{num(m_of[('estatico', 'train')]['sharpe'])}, {num(m_of[('estatico', 'test')]['sharpe'])} y "
        f"{num(m_of[('estatico', 'val')]['sharpe'])} para el estático. Kalman hace "
        f"{int(m_of[('kalman', 'test')]['n_operaciones'])} operaciones en test y "
        f"{int(m_of[('kalman', 'val')]['n_operaciones'])} en val.",
        f"La relación no es estable fuera de muestra: con el vector de train, el ADF del spread en test da "
        f"p = {num(o['test']['ADF_spread_p'], 2)} y la vida media sube a {num(o['test']['vida_media'], 0)} días; al "
        f"re-partir el propio train, el vector de sub-train tampoco es estacionario en sub-test ni en sub-val. En val, "
        f"un solo día ({EVENTO}, SBAC +18.9%) explica el {pct(ev['kalman']['frac'], 0)} del P&L del Kalman.",
        "Conclusión: el sistema es causal y su contabilidad cuadra, pero la evidencia no sostiene una relación "
        "estable entre AMT y SBAC y el desempeño fuera de muestra no es robusto, en línea con el sobreajuste "
        "señalado en la revisión del avance.",
    ])

    # ============================================================== 1. datos
    n_covid = int(((pd.to_datetime(saltos["fecha"]) >= "2020-02-01") & (pd.to_datetime(saltos["fecha"]) <= "2020-03-31")).sum())
    con_evento = saltos[saltos["evento"].notna()]
    s += [*H("1. Datos y partición", "h1"),
          P(f"Precios de cierre ajustados de Yahoo Finance (<i>yfinance</i>) para AMT, CCI, SBAC, VZ, T y TMUS, del "
            f"{rangos['train'][0]} al {rangos['val'][1]}: {int(huecos['n_obs'].iloc[0]):,} días de negociación comunes a "
            f"los seis activos, sin faltantes. La partición es cronológica, sin mezclar fechas. Los primeros "
            f"{len(fechas_burn)} días de train ({fechas_burn[0].date()} a {fechas_burn[-1].date()}) son el burn-in de "
            f"los filtros y no se operan. Todo lo que se estima (vector, filtros y umbrales) usa solo train. Los precios "
            f"quedan guardados en <i>artifacts/</i> porque Yahoo reajusta la historia con cada dividendo y sin el "
            f"snapshot los resultados no serían reproducibles.")]
    s += N.tabla("Partición de la muestra.", tabla(
        [["muestra", "desde", "hasta", "días", "uso"]] +
        [[m, str(rangos[m][0]), str(rangos[m][1]), f"{rangos[m][2]:,}", u] for m, u in
         zip(MUESTRAS, ["estimar todo (126 días de burn-in)", "medir, sin re-estimar", "medir, sin re-estimar"])],
        [2 * cm, 2.6 * cm, 2.6 * cm, 1.6 * cm, 8.2 * cm], derecha=(3,)))
    s += [P(f"Revisión de calidad: {len(saltos)} rendimientos diarios con |z| &gt; 6 (<i>data_jumps.csv</i>); "
            f"{n_covid} caen entre febrero y marzo de 2020 (COVID). Dos tienen un evento corporativo documentado: "
            + "; ".join(f"{r['ticker']} el {r['fecha']} ({r['evento'].lower()})" for _, r in con_evento.iterrows())
            + ". No se eliminó ni corrigió ningún dato: los saltos son movimientos reales del mercado."),
          N.figura("fig1_precios.png", "Precios de cierre ajustados de AMT y SBAC. Sombras: burn-in (gris), test "
                   "(naranja) y val (verde).", enunciado="1. Precios individuales del grupo")]

    # ============================================================== 2. seleccion
    s += [*H("2. Selección del grupo", "h1"),
          P("Las torres (AMT, CCI, SBAC) rentan espacio a las telefónicas (VZ, T, TMUS) con contratos de largo plazo: "
            "comparten clientes, flujos y sensibilidad a tasas. Esa razón económica es previa a las pruebas. "
            "El código está en <i>cointegration/</i> y corre solo sobre train."),
          *H("2.1 Orden de integración", "h2"),
          P("Se clasifica un activo como I(1) si el ADF en nivel no rechaza raíz unitaria, el ADF en diferencias sí la "
            "rechaza y el KPSS en nivel rechaza estacionariedad, todo al 5% sobre log-precios.")]
    s += N.tabla("Orden de integración en train (valores p).", tabla(
        [["activo", "ADF nivel", "ADF diferencias", "KPSS nivel", "clasificación"]] +
        [[r["ticker"], num(r["ADF_nivel_p"], 4), num(r["ADF_diff_p"], 4), num(r["KPSS_nivel_p"], 2),
          r["orden"]] for _, r in orden.iterrows()],
        [2.2 * cm, 2.6 * cm, 3 * cm, 2.6 * cm, 6.6 * cm], derecha=(1, 2, 3)))
    no_i1 = orden.loc[orden["orden"] != "I(1)", "ticker"].tolist()
    s += [P(f"{', '.join(no_i1)} queda en revisión: su ADF en nivel rechaza raíz unitaria "
            f"(p = {num(orden.set_index('ticker').loc[no_i1[0], 'ADF_nivel_p'], 3)}). Todo grupo que lo incluya lleva "
            f"bandera roja, porque mezclar un activo que podría ser estacionario con otros I(1) puede producir una "
            f"cointegración aparente." if no_i1 else "Los seis activos son I(1)."),
          *H("2.2 Prefiltro informativo: correlación y Engle-Granger", "h2"),
          P("Para los 15 pares se calcula la mediana de la correlación móvil de 126 días de los rendimientos (umbral "
            "0.70) y Engle-Granger en los dos sentidos. Ninguna de las dos bloquea: la correlación mide co-movimiento "
            "de corto plazo, no equilibrio de largo plazo, y Engle-Granger depende de qué activo va como dependiente.")]
    filas = [["par", "correlación móvil (mediana)", "EG p (A depende de B)", "EG p (B depende de A)", "pasa corr.", "pasa EG"]]
    for _, r in eg.iterrows():
        filas.append([r["par"], num(r["corr_movil_mediana"], 3), num(r["EG_p_AB"], 3), num(r["EG_p_BA"], 3),
                      "sí" if a_bool(pd.Series([r["pasa_corr"]]))[0] else "no",
                      "sí" if a_bool(pd.Series([r["pasa_EG"]]))[0] else "no"])
    s += N.tabla("Prefiltro de los 15 pares en train.", tabla(
        filas, [2.4 * cm, 3.6 * cm, 3.3 * cm, 3.3 * cm, 2.2 * cm, 2.2 * cm], derecha=(1, 2, 3)))
    n_rech_eg = int((eg["EG_p_AB"] < 0.05).sum() + (eg["EG_p_BA"] < 0.05).sum())
    s += [P(f"Solo los pares de torres pasan la correlación. Engle-Granger rechaza en {n_rech_eg} de "
            f"{int(fpt['Engle-Granger']['n_tests'])} pruebas, cuando el azar daría "
            f"{num(fpt['Engle-Granger']['FP_esperados'], 1)} al 5%, y ninguno de esos rechazos es de un par de torres. "
            f"En particular, AMT y SBAC <b>no pasan Engle-Granger al 5%</b> (p = {num(par['EG_p_AB'], 3)} con AMT como "
            f"dependiente y {num(par['EG_p_BA'], 3)} con SBAC)."),
          *H("2.3 Embudo bloqueante: Johansen, ADF y vida media", "h2"),
          P("Los 57 grupos (todas las combinaciones de 2 a 6 activos) pasan por tres etapas en orden: (1) Johansen con "
            "constante y un rezago en diferencias, con rango entre 1 y n − 1 tanto por traza como por máximo eigenvalor "
            "al 95%; (2) ADF del spread construido con el primer eigenvector, normalizado en la pata de mayor peso, con "
            "p &lt; 0.05; (3) vida media de un AR(1) del spread entre 5 y 60 días hábiles.")]
    cuenta = v2["etapa_caida"].value_counts()
    pasan = v2[v2["etapa_caida"] == "PASA"].sort_values("ADF_spread_p")
    filas = [["grupo", "rango traza / máx. eig.", "ADF p", "vida media", "bandera roja", "motivo de la bandera"]]
    for _, r in pasan.iterrows():
        roja = a_bool(pd.Series([r["bandera_roja"]]))[0]
        filas.append([r["grupo"].replace("+", " + "), f"{r['rank_traza']} / {r['rank_maxeig']}", num(r["ADF_spread_p"], 4),
                      f"{num(r['vida_media'], 1)} días", "sí" if roja else "no",
                      (f"incluye {', '.join(ast.literal_eval(r['activo_no_I1']))} (no I(1))" if roja else "")])
    s += N.tabla(f"Grupos que pasan el embudo ({len(pasan)} de {len(v2)}; los otros "
                 f"{int(cuenta.get('johansen', 0))} caen en Johansen). La tabla completa está en el anexo A.",
                 tabla(filas, [3 * cm, 3.2 * cm, 1.6 * cm, 2 * cm, 2 * cm, 5.2 * cm], derecha=(2, 3)))
    s += [P(f"Se eligió <b>{grupo}</b> (criterio en <i>chosen_group.json</i>: el único grupo que pasa sin bandera roja, "
            f"con la mejor justificación económica, porque las dos son empresas de torres). Su vector normalizado en AMT "
            f"es (1, {num(-b1, 4)}): el spread es log AMT − {num(b1, 4)} log SBAC. En el backtest los pesos se aplican "
            f"al nocional, así que por cada dólar largo en AMT se venden {num(b1, 4)} dólares de SBAC, y al revés."),
          *H("2.4 Pruebas múltiples y lectura honesta del embudo", "h2"),
          P(f"Johansen se aplicó a {int(fpt['Johansen']['n_tests'])} grupos: al 5% esperaríamos "
            f"{num(fpt['Johansen']['FP_esperados'], 2)} falsos positivos por puro azar y pasan "
            f"{int(((v2['rank_traza'] >= 1) & (v2['rank_maxeig'] >= 1)).sum())}. Los grupos comparten activos y las "
            f"pruebas no son independientes, así que la cuenta es una referencia, pero dice que tres aprobados de 57 es "
            f"justo lo que el azar produciría. Además, {int((v2['ADF_spread_p'] < 0.05).sum())} de los {len(v2)} grupos "
            f"tienen ADF del spread con p &lt; 0.05, incluidos casi todos los que fallan Johansen: el eigenvector se "
            f"estima para que el spread sea lo más estacionario posible en la misma muestra, y los valores críticos "
            f"estándar del ADF no corrigen eso. En la práctica la etapa que decide es Johansen, y la elección final "
            f"descansa en la razón económica.")]
    corr_v1 = {f"{a}-{b}": corr_movil_mediana(np.log(px1).diff(), a, b) for a, b in
               [p.split("-") for p in eg1["par"]]}
    filas = [["par (universo v1)", "correlación móvil (mediana)", "EG p (A depende de B)", "EG p (B depende de A)"]]
    for _, r in eg1.iterrows():
        filas.append([r["par"], num(corr_v1[r["par"]], 3), num(r["EG_p_AB"], 3), num(r["EG_p_BA"], 3)])
    s += [*H("2.5 Universo anterior (v1)", "h2"),
          P(f"Antes se probó un universo de telecomunicaciones y medios: VZ, T, BCE, DIS, AMX y NFLX. Con el embudo de "
            f"ese momento, que usaba la correlación como primera etapa bloqueante, los {len(v1)} grupos cayeron en "
            f"correlación y no llegaron a Johansen; Engle-Granger no rechazó en ninguna de las 30 pruebas. Como ningún "
            f"grupo avanzó, se cambió a un universo con un vínculo económico más directo (torres y sus clientes). Ese cambio se decidió "
            f"después de ver resultados, y los {len(v1)} grupos de v1 cuentan en el total de {len(v1) + len(v2)} "
            f"grupos probados.")]
    s += N.tabla("Universo v1: prefiltro de los 15 pares (correlación recalculada con la misma función).",
                 tabla(filas, [3.6 * cm, 4.2 * cm, 4.2 * cm, 4.2 * cm], derecha=(1, 2, 3)))
    a_amt, a_sb = valpha.loc["AMT"], valpha.loc["SBAC"]
    vel = a_amt["alpha"] - (-chosen["vector"][1]) * a_sb["alpha"]
    s += [*H("2.6 VECM", "h2"),
          P(f"VECM en train con rango 1, constante y un rezago, con el vector de Johansen. Los coeficientes de ajuste son "
            f"α<sub>AMT</sub> = {num(a_amt['alpha'], 4)} (p = {num(a_amt['pval'], 3)}) y α<sub>SBAC</sub> = "
            f"{num(a_sb['alpha'], 4)} (p = {num(a_sb['pval'], 3)}). Quien corrige el desequilibrio es SBAC: cuando el "
            f"spread está arriba de su equilibrio, SBAC sube y el spread regresa. La velocidad implícita del spread es "
            f"α<sub>AMT</sub> − {num(b1, 4)} α<sub>SBAC</sub> = {num(vel, 4)} por día, una vida media de "
            f"{num(np.log(2) / -vel, 1)} días, coherente con los {chosen['vida_media']} del AR(1)."),
          *H("2.7 Estabilidad del vector fuera de muestra", "h2")]
    s += N.tabla("Spread con el vector fijo de train evaluado en cada muestra (<i>oos_check.csv</i>).", tabla(
        [["muestra", "periodo", "ADF p", "vida media", "estacionario al 5%", "vida media en [5, 60]"]] +
        [[m, f"{o[m]['fecha_inicio']} a {o[m]['fecha_fin']}", num(o[m]["ADF_spread_p"], 4),
          f"{num(o[m]['vida_media'], 1)} días", "sí" if a_bool(pd.Series([o[m]["pasa_ADF"]]))[0] else "no",
          "sí" if a_bool(pd.Series([o[m]["pasa_vida_media"]]))[0] else "no"] for m in MUESTRAS],
        [1.8 * cm, 4.6 * cm, 1.8 * cm, 2.4 * cm, 3.2 * cm, 3.2 * cm], derecha=(2, 3)))
    s += N.tabla("Robustez dentro de train (<i>robustez_oos.csv</i>): el train se re-parte 60/20/20, el vector se "
                 "estima solo en sub-train y se congela.", tabla(
        [["sub-muestra", "días", "ADF p", "vida media", "estacionario al 5%"]] +
        [[m.replace("_", "-"), f"{int(rb[m]['n_obs']):,}", num(rb[m]["ADF_spread_p"], 4),
          f"{num(rb[m]['vida_media'], 1)} días", "sí" if a_bool(pd.Series([rb[m]["estacionario"]]))[0] else "no"]
         for m in ["sub_train", "sub_test", "sub_val"]],
        [3 * cm, 2 * cm, 2.4 * cm, 3 * cm, 6.6 * cm], derecha=(1, 2, 3)))
    fuera = int((eig["SBAC"].abs() > 3).sum())
    positivos = int((eig["SBAC"] > 0).sum())
    s += [P(f"Con el vector de train, el spread deja de ser estacionario en test y lo vuelve a ser en val. La misma "
            f"ruptura aparece dentro de train: el vector de sub-train no es estacionario ni en sub-test ni en sub-val. "
            f"El Johansen móvil (ventanas de 252 días con paso de 5, solo en train) lo confirma: de {len(eig)} ventanas, "
            f"en {positivos} el peso de SBAC sale positivo (ya no es un spread entre los dos) y en {fuera} se sale de "
            f"[−3, 3]. La mediana es {num(eig['SBAC'].median(), 2)}, cerca del vector de todo el train, pero la "
            f"dispersión muestra que el vector no está bien determinado ventana a ventana."),
          N.figura("fig4_johansen_movil.png", "Peso de SBAC en el primer eigenvector de Johansen en ventanas móviles "
                   "de 252 días dentro de train (normalizado en AMT = 1). Los rombos marcan ventanas fuera de escala.",
                   enunciado="4. Primer eigenvector de Johansen (ventana móvil)")]

    # ============================================================== 3. Powell
    s += [*H("3. Formulación con el marco de Powell", "h1"),
          P("El marco de decisiones secuenciales de Powell describe un sistema con cinco elementos: el estado S<sub>t</sub> "
            "(lo que se sabe en t), la decisión (aquí d<sub>t</sub>, para no confundirla con el regresor x<sub>t</sub>), la "
            "información exógena W<sub>t+1</sub> (lo que llega después de decidir), la función de transición "
            "S<sub>t+1</sub> = S<super>M</super>(S<sub>t</sub>, d<sub>t</sub>, W<sub>t+1</sub>) y la función objetivo. "
            "Cada formulación se construye en seis pasos: definir el problema, modelar el estado, modelar la transición, "
            "modelar la observación, definir la regla de decisión y definir el mecanismo de aprendizaje o actualización. "
            "Se aplica tres veces: al filtro 1, al filtro 2 y al sistema completo. La notación es "
            "y<sub>t</sub> = log AMT<sub>t</sub> y x<sub>t</sub> = log SBAC<sub>t</sub>; x<super>c</super><sub>t</sub> = "
            f"x<sub>t</sub> − {num(kp['x_media_burnin'], 4)} es el regresor centrado con la media del burn-in.")]
    filas = [["elemento", "Filtro 1: pesos", "Filtro 2: nivel del spread", "Sistema completo"],
             ["Estado",
              "Creencia sobre los pesos: media z<sub>t|t</sub> = (α<sub>t|t</sub>, β<sub>t|t</sub>) y covarianza "
              "P<sub>t|t</sub> de 2 × 2.",
              "Creencia sobre el nivel: μ<sub>t|t</sub>, su varianza P2<sub>t|t</sub> y la varianza estimada del ruido "
              "R2<sub>t</sub>.",
              "S<sub>t</sub> = (R<sub>t</sub>, I<sub>t</sub>, B<sub>t</sub>): físico (efectivo, acciones, días en la "
              "operación, bloqueo), información (precios, x<super>c</super><sub>t</sub>, s<sub>t</sub>) y creencia (los "
              "estados de los dos filtros)."],
             ["Decisión",
              "Ninguna en línea: es un estimador. Fuera de línea se eligen δ, R y la inicialización.",
              "Ninguna en línea: entrega z<sub>t</sub> a la política. Fuera de línea se eligen L, q y λ.",
              "d<sub>t</sub> ∈ {−1, 0, +1}: corto, fuera o largo el spread. El tamaño queda fijo por la regla de "
              "80% de exposición."],
             ["Información exógena",
              "W<sub>t+1</sub> = (AMT<sub>t+1</sub>, SBAC<sub>t+1</sub>): con ellos se forman y<sub>t+1</sub> y "
              "x<super>c</super><sub>t+1</sub>.",
              "s<sub>t+1</sub> = y<sub>t+1</sub> − β<sub>t+1|t</sub> x<super>c</super><sub>t+1</sub>, que usa los "
              "precios nuevos y el β previo del filtro 1.",
              "Los precios de cierre del día siguiente; los costos (comisión y préstamo) son parámetros conocidos."],
             ["Transición",
              "Predicción y actualización de Kalman (ecuaciones 3 a 5).",
              "Predicción, z-score y actualización del nivel local, más el EWMA de R2 (ecuaciones 7 a 10).",
              "Los dos filtros más la contabilidad: ejecución en t+1, mark-to-market, préstamo, comisiones y "
              "rebalanceo diario (ecuaciones 12 y 13)."],
             ["Objetivo",
              "Con δ y R dados, el error cuadrático medio mínimo del estado. Para calibrar: verosimilitud de las "
              "innovaciones con δ ≤ 10<super>−5</super> y R por momentos.",
              "Un z<sub>t</sub> calibrado (media 0, varianza 1) con la misma memoria que la ventana del estático.",
              "Máximo Sharpe neto en train con al menos 10 operaciones (ecuación 14), una versión ajustada por riesgo "
              "del objetivo general de Powell (ecuación 15)."]]
    s += N.tabla("Los cinco elementos de Powell en las tres formulaciones.",
                 tabla(filas, [2.4 * cm, 4.6 * cm, 4.6 * cm, 5.4 * cm], texto=True))

    # ---- filtro 1
    s += [*H("3.1 Filtro 1: pesos dinámicos (α<sub>t</sub>, β<sub>t</sub>)", "h2"),
          P("Observación y transición (A = I, paseo aleatorio; C<sub>t</sub> = (1, x<super>c</super><sub>t</sub>)). Los "
            "ruidos son gaussianos, independientes entre sí y en el tiempo:"),
          ecuacion(r"(1)\quad y_t = \alpha_t + \beta_t\, x^{c}_t + v_t,\qquad \mathrm{Var}(v_t) = R"),
          ecuacion(r"(2)\quad z_t = (\alpha_t,\ \beta_t)^{\top} = z_{t-1} + w_t,\qquad \mathrm{Var}(w_t) = Q = \dfrac{\delta}{1-\delta}\, I_2"),
          P("Cada día se predice antes de ver el dato y se actualiza después:"),
          ecuacion(r"(3)\quad \hat z_{t|t-1} = \hat z_{t-1|t-1},\qquad P_{t|t-1} = P_{t-1|t-1} + Q"),
          ecuacion(r"(4)\quad e_t = y_t - C_t\,\hat z_{t|t-1},\qquad S_t = C_t P_{t|t-1} C_t^{\top} + R,\qquad K_t = \dfrac{P_{t|t-1} C_t^{\top}}{S_t}"),
          ecuacion(r"(5)\quad \hat z_{t|t} = \hat z_{t|t-1} + K_t\, e_t,\qquad P_{t|t} = (I - K_t C_t)\, P_{t|t-1}")]
    filas = [["paso", "Filtro 1"],
             ["1. Definir el problema",
              "Estimar en línea la cobertura entre log AMT y log SBAC sin usar el futuro. Entre dos empresas del mismo "
              "negocio la cobertura es estructural: debe moverse en meses, no en días."],
             ["2. Modelar el estado",
              f"z<sub>t</sub> = (α<sub>t</sub>, β<sub>t</sub>). Se conoce solo su distribución: media y covarianza. "
              f"Inicio con el OLS del burn-in, z<sub>0</sub> = ({num(kp['z0'][0], 4)}, {num(kp['z0'][1], 4)}), y "
              f"P<sub>0</sub> = I. El regresor se centra porque sin centrar x ronda 5.2 y las columnas de C<sub>t</sub> "
              f"son casi colineales: el filtro confundía cambios de nivel con cambios de β."],
             ["3. Modelar la transición",
              f"Paseo aleatorio (ecuación 2) con δ = {pot10(kp['delta'])}, es decir Q = {cient(kp['Q'][0][0], 6)} I."],
             ["4. Modelar la observación",
              f"Ecuación 1 con R = {cient(kp['R'], 3)}, calibrada por método de momentos (sección 5.1)."],
             ["5. Definir la regla de decisión",
              "No hay decisión de trading. La regla es el estimador: se usa β<sub>t|t−1</sub>, el β previo, para "
              "formar el spread y la cobertura del día, así nada del día t entra a su propio peso."],
             ["6. Mecanismo de aprendizaje",
              "En línea, la recursión de Kalman (ecuaciones 3 a 5): nuevo estado = estado previo + ganancia × "
              "sorpresa. Fuera de línea, la calibración de δ y R en train (sección 5.1)."]]
    s += N.tabla("Los seis pasos aplicados al filtro 1.", tabla(filas, [4 * cm, 13 * cm], texto=True))
    s += [P(f"La ganancia K<sub>t</sub> reparte la sorpresa e<sub>t</sub> entre α y β según la incertidumbre de cada "
            f"uno y según x<super>c</super><sub>t</sub>: K<sub>β</sub> cambia de signo cuando log SBAC cruza su media "
            f"del burn-in. Con δ = {pot10(kp['delta'])} los pesos cambian despacio; β<sub>t|t−1</sub> se mueve entre "
            f"{num(sen.loc[~burn & (sen['muestra'] == 'train').to_numpy(), 'beta'].min(), 2)} y "
            f"{num(sen.loc[~burn & (sen['muestra'] == 'train').to_numpy(), 'beta'].max(), 2)} en train y entre "
            f"{num(sen.loc[(sen['muestra'] != 'train').to_numpy(), 'beta'].min(), 2)} y "
            f"{num(sen.loc[(sen['muestra'] != 'train').to_numpy(), 'beta'].max(), 2)} en test y val, contra "
            f"{num(b1, 4)} del vector fijo."),
          N.figura("fig3_pesos.png", "Peso de SBAC en el vector de cobertura normalizado en AMT = 1: Kalman "
                   "(β previo, con banda de dos desviaciones) contra el vector fijo de Johansen. El salto de abril "
                   f"de 2026 es la reacción del filtro al alza de SBAC del {EVENTO}.",
                   enunciado="3. Vector de pesos β<sub>t</sub> (filtro 1)")]

    # ---- filtro 2
    s += [*H("3.2 Filtro 2: nivel y escala del spread", "h2"),
          P("Con el β previo se forma el spread; el nivel α no se resta porque lo absorbe este filtro como un nivel "
            "local μ<sub>t</sub>:"),
          ecuacion(r"(6)\quad s_t = y_t - \hat\beta_{t|t-1}\, x^{c}_t"),
          ecuacion(r"(7)\quad s_t = \mu_t + v_t,\quad \mathrm{Var}(v_t) = R2_t;\qquad \mu_t = \mu_{t-1} + w_t,\quad \mathrm{Var}(w_t) = Q2_t = q\, R2_{t-1}"),
          ecuacion(r"(8)\quad P2_{t|t-1} = P2_{t-1|t-1} + Q2_t,\qquad S2_t = P2_{t|t-1} + R2_{t-1},\qquad z_t = \dfrac{s_t - \hat\mu_{t|t-1}}{\sqrt{S2_t}}"),
          ecuacion(r"(9)\quad K2_t = \dfrac{P2_{t|t-1}}{S2_t},\qquad \hat\mu_{t|t} = \hat\mu_{t|t-1} + K2_t\,(s_t - \hat\mu_{t|t-1}),\qquad P2_{t|t} = (1 - K2_t)\, P2_{t|t-1}"),
          ecuacion(r"(10)\quad R2_t = \lambda\, R2_{t-1} + (1-\lambda)\,(s_t - \hat\mu_{t|t-1})^2"),
          P("El z-score se calcula antes de actualizar, así que su nivel y su escala solo usan información hasta t − 1.")]
    filas = [["paso", "Filtro 2"],
             ["1. Definir el problema",
              "Saber cuándo el spread está lejos de su nivel normal, con un nivel que puede moverse despacio y una "
              "escala que cambia con la volatilidad."],
             ["2. Modelar el estado",
              f"(μ<sub>t</sub>, P2<sub>t</sub>, R2<sub>t</sub>). Inicio con el spread del burn-in calculado con el β del "
              f"OLS fijo: μ<sub>0</sub> = {num(kp['mu_0'], 4)}, R2<sub>0</sub> = {cient(kp['R2_0'], 3)} y "
              f"P2<sub>0</sub> = K R2<sub>0</sub> = {cient(kp['P2_0'], 3)}, el valor de estado estable."],
             ["3. Modelar la transición",
              f"Nivel local con Q2<sub>t</sub> = q R2<sub>t−1</sub> y q = {cient(q, 3)} (ecuación 7)."],
             ["4. Modelar la observación",
              f"s<sub>t</sub> = μ<sub>t</sub> + v<sub>t</sub>, con R2 estimada por EWMA con λ = {num(lam, 5)} "
              f"(ecuación 10)."],
             ["5. Definir la regla de decisión",
              "No hay decisión de trading: el filtro entrega z<sub>t</sub> (ecuación 8), que es la variable de estado "
              "sobre la que actúa la política."],
             ["6. Mecanismo de aprendizaje",
              f"En línea, la actualización del nivel y el EWMA de la escala (ecuaciones 9 y 10). Fuera de línea, L = "
              f"{kp['L']} días fija K = 1/L, q y λ (sección 5.2)."]]
    s += N.tabla("Los seis pasos aplicados al filtro 2.", tabla(filas, [4 * cm, 13 * cm], texto=True))
    s += [N.figura("fig5_ganancia.png", "Ganancia de Kalman de los dos filtros, después del burn-in. La del filtro 2 "
                   "oscila alrededor de 1/L: baja después de un salto de volatilidad (R2 sube) y sube cuando la "
                   "volatilidad se calma.", enunciado="5. Ganancia de Kalman de ambos filtros")]

    # ---- sistema completo
    s += [*H("3.3 Sistema completo", "h2"),
          P("El estado separa lo físico, la información y la creencia:"),
          ecuacion(r"(11)\quad S_t = \left(R_t,\ I_t,\ B_t\right),\qquad R_t = (\mathrm{efectivo}_t,\ q^{AMT}_t,\ q^{SBAC}_t,\ h_t,\ \mathrm{bloqueo}_t)"),
          ecuacion(r"I_t = (P^{AMT}_t,\ P^{SBAC}_t,\ s_t),\qquad B_t = (\hat z_{t|t},\ P_{t|t},\ \hat\mu_{t|t},\ P2_{t|t},\ R2_t)"),
          P(f"La señal se calcula con el cierre de t y se ejecuta al cierre de t + 1. Al abrir, y en cada rebalanceo "
            f"diario, el nocional por pata con signo es (V<sub>t</sub> es el valor de la cuenta y w el vector de pesos del "
            f"día de la señal):"),
          ecuacion(r"(12)\quad N_t = d_t \cdot 0.8\, V_t\, \dfrac{w_t}{|w_t|_1},\qquad w_t = \left(1,\ -\hat\beta_{t|t-1}\right),\qquad |w_t|_1 = 1 + \hat\beta_{t|t-1}"),
          ecuacion(r"(13)\quad V_{t+1} = V_t + q_t^{\top}\left(P_{t+1} - P_t\right) - \mathrm{pr\'estamo}_{t+1} - \mathrm{comisiones}_{t+1}"),
          P(f"con comisión de {num(100 * cfg['comision'], 3)}% del nocional por pata en cada operación y préstamo de "
            f"{num(100 * cfg['tasa_prestamo'], 2)}% anual sobre el nocional corto, por día natural con base "
            f"{cfg['base_dias']}. El objetivo con el que se eligió θ y su forma general en Powell son:"),
          ecuacion(r"(14)\quad \max_{\theta}\ \mathrm{Sharpe}(\theta) = \sqrt{252}\ \dfrac{\bar r(\theta)}{\sigma_r(\theta)}\qquad \mathrm{sujeto\ a}\quad n_{\mathrm{ops}}(\theta) \geq 10"),
          ecuacion(r"(15)\quad \max_{\pi}\ \mathrm{E}\left[\sum_{t=0}^{T} C\left(S_t, X^{\pi}(S_t)\right)\ \mid\ S_0\right]"),
          P("donde r son los rendimientos diarios netos de la cuenta y C la ganancia neta del día. El Sharpe divide la "
            "recompensa media entre su volatilidad: es la versión ajustada por riesgo del objetivo de Powell. La "
            "política es una función de política (PFA) con θ = (z<sub>in</sub>, z<sub>out</sub>, z<sub>stop</sub>, "
            "h<sub>max</sub>):")]
    s += N.tabla("Regla de decisión (PFA), <i>strategy/reglas.py</i>.", tabla(
        [["situación", "regla"],
         ["Fuera de posición", "Entra si z<sub>in</sub> &lt; |z<sub>t</sub>| ≤ z<sub>stop</sub>: d<sub>t</sub> = +1 "
                               "(largo AMT, corto SBAC) si z<sub>t</sub> &lt; 0 y d<sub>t</sub> = −1 si z<sub>t</sub> "
                               "&gt; 0. Si |z<sub>t</sub>| ya pasa z<sub>stop</sub>, no entra."],
         ["En posición", "Cierra por salida si |z<sub>t</sub>| &lt; z<sub>out</sub>, por stop si |z<sub>t</sub>| &gt; "
                         "z<sub>stop</sub> o por tiempo si h<sub>t</sub> ≥ h<sub>max</sub>; si no, mantiene."],
         ["Después de un stop", "No vuelve a entrar hasta que |z<sub>t</sub>| &lt; z<sub>in</sub>: si la relación se "
                                "rompió, no se apuesta otra vez a lo que acaba de fallar."]],
        [3.6 * cm, 13.4 * cm], texto=True))
    s += [P("Es la misma clase de política que la regla mín-máx de inventarios: dos umbrales sobre una variable del "
            "estado disparan la acción, y los umbrales se ajustan simulando la política y midiendo el objetivo.")]
    filas = [["paso", "Sistema completo"],
             ["1. Definir el problema",
              "Decidir cada día si estar largo, corto o fuera del spread AMT/SBAC para maximizar el rendimiento neto "
              "ajustado por riesgo, sin usar información futura."],
             ["2. Modelar el estado",
              "S<sub>t</sub> = (R<sub>t</sub>, I<sub>t</sub>, B<sub>t</sub>) de la ecuación 11. El estado inicial "
              "incluye los parámetros fijos: δ, R, L, θ, costos, capital de 1,000,000 y exposición de 80%."],
             ["3. Modelar la transición",
              "S<sup>M</sup>: el filtro 1 actualiza (α, β, P), el filtro 2 actualiza (μ, P2, R2) y produce "
              "z<sub>t+1</sub>, y la contabilidad mueve el estado físico (ecuaciones 12 y 13)."],
             ["4. Modelar la observación",
              "Se observan los cierres ajustados de AMT y SBAC; los pesos y el nivel de equilibrio no se observan y "
              "entran por los filtros."],
             ["5. Definir la regla de decisión",
              f"La PFA de la tabla anterior con θ* = ({num(tk['z_in'], 3)}, {num(tk['z_out'], 3)}, "
              f"{num(tk['z_stop'], 3)}, {tk['h_max']}) para Kalman."],
             ["6. Mecanismo de aprendizaje",
              "θ se busca una sola vez en train con optimización bayesiana y se congela (sección 6). En test y val los "
              "filtros siguen aprendiendo en línea, pero θ, δ, R y L no se re-estiman."]]
    s += N.tabla("Los seis pasos aplicados al sistema completo.", tabla(filas, [4 * cm, 13 * cm], texto=True))
    s += [N.figura("fig2_spread_senales.png", "Spread de cada método con su nivel estimado y las operaciones del "
                   "backtest oficial (marcas en la fecha de ejecución). El spread del Kalman incluye el nivel α porque "
                   "lo absorbe μ.", enunciado="2. Spread con señales"),
          N.figura("fig6_zscore.png", "z-score causal de cada método con los umbrales de su θ* (recortado a ±5).",
                   enunciado="6. z-score con umbrales")]

    # ============================================================== 4. ejemplos numericos
    y_e, x_e = np.array([1.6, 2.1, 0.4]), np.array([1.0, 2.0, -1.0])
    r1 = correr_filtro_pesos(y_e, x_e, 1 / 101, [1.0, 0.5], 0.1)
    s += [*H("4. Ejemplos numéricos", "h1"),
          *H("4.1 Filtro 1, tres periodos", "h2"),
          P("Números redondos: z<sub>0</sub> = (1, 0.5), P<sub>0</sub> = I, Q = 0.01 I (δ = 1/101) y R = 0.1, con "
            "observaciones (x<sub>1</sub>, y<sub>1</sub>) = (1, 1.6), (x<sub>2</sub>, y<sub>2</sub>) = (2, 2.1) y "
            "(x<sub>3</sub>, y<sub>3</sub>) = (−1, 0.4). La tabla sale de <i>correr_filtro_pesos</i>, la misma función "
            "del pipeline, y coincide con el cálculo a mano."),
          P("Periodo 1. Predicción: z<sub>1|0</sub> = (1, 0.5) y P<sub>1|0</sub> = I + 0.01 I = 1.01 I. Con "
            "C<sub>1</sub> = (1, 1) se espera y = 1 + 0.5 = 1.5, así que la sorpresa es e<sub>1</sub> = 1.6 − 1.5 = 0.1. "
            "S<sub>1</sub> = 1.01 + 1.01 + 0.1 = 2.12 y K<sub>1</sub> = (1.01, 1.01)/2.12 = (0.4764, 0.4764). "
            "Actualización: z<sub>1|1</sub> = (1, 0.5) + 0.1 (0.4764, 0.4764) = (1.0476, 0.5476). Con un solo dato en "
            "x = 1 el filtro no separa α de β, solo su suma: la covarianza entre los dos se vuelve negativa "
            "(−0.4812). En el periodo 3, con x del otro lado de cero, esa covarianza casi desaparece; es la misma razón "
            "por la que el modelo real centra x.")]
    filas = [["t", "x", "y", "α previo", "β previo", "e", "S", "K<sub>α</sub>", "K<sub>β</sub>", "α post.",
              "β post.", "P<sub>α</sub>", "P<sub>β</sub>"]]
    for i, r in r1.iterrows():
        filas.append([str(i + 1), num(x_e[i], 0), num(y_e[i], 1)] +
                     [num(r[c], 4) for c in ["alpha_prior", "beta_prior", "e", "S", "K_alpha", "K_beta", "alpha_post",
                                             "beta_post", "P_alpha", "P_beta"]])
    s += N.tabla("Filtro 1 paso a paso (P<sub>α</sub> y P<sub>β</sub> son la diagonal de P<sub>t|t</sub>).",
                 tabla(filas, [0.6 * cm, 0.8 * cm, 0.9 * cm] + [1.47 * cm] * 10, derecha=tuple(range(13))))
    s_e, s_ini, L_e = np.array([0.5, -0.3, 1.2]), np.array([-1.0, 0.0, 1.0]), 5
    r2, p2 = correr_filtro_spread(s_e, s_ini, L_e)
    # reconstruye las columnas intermedias para la tabla, con las mismas ecuaciones
    mu, R2, P2 = p2["mu_0"], p2["R2_0"], p2["P2_0"]
    filas = [["t", "s<sub>t</sub>", "μ previo", "Q2", "P2 previo", "S2", "e", "z<sub>t</sub>", "K2", "μ post.",
              "P2 post.", "R2"]]
    for t, st in enumerate(s_e):
        Q2 = p2["q"] * R2
        Pp = P2 + Q2
        S2 = Pp + R2
        e = st - mu
        K2 = Pp / S2
        mu_n, P2 = mu + K2 * e, (1 - K2) * Pp
        R2_n = p2["lambda"] * R2 + (1 - p2["lambda"]) * e ** 2
        assert np.isclose(mu_n, r2["mu_post"].iloc[t]) and np.isclose(e / np.sqrt(S2), r2["z"].iloc[t])
        filas.append([str(t + 1), num(st, 1), num(mu, 4), num(Q2, 4), num(Pp, 4), num(S2, 4), num(e, 4),
                      num(e / np.sqrt(S2), 4), num(K2, 4), num(mu_n, 4), num(P2, 4), num(R2_n, 4)])
        mu, R2 = mu_n, R2_n
    s += [*H("4.2 Filtro 2, tres periodos", "h2"),
          P(f"Memoria L = {L_e}: K = 1/L = {num(p2['K_obj'], 2)}, q = K²/(1 − K) = {num(p2['q'], 2)} y λ = 1 − 1/L = "
            f"{num(p2['lambda'], 1)}. El spread de inicialización es (−1, 0, 1), así que μ<sub>0</sub> = 0, "
            f"R2<sub>0</sub> = 1 y P2<sub>0</sub> = K R2<sub>0</sub> = 0.2. Observaciones: 0.5, −0.3 y 1.2."),
          P("Periodo 1. Q2 = 0.05 × 1 = 0.05, P2 previo = 0.2 + 0.05 = 0.25 y S2 = 0.25 + 1 = 1.25. La sorpresa es "
            "e = 0.5 − 0 = 0.5 y el z-score, calculado antes de actualizar, es 0.5/√1.25 = 0.4472. La ganancia es "
            "0.25/1.25 = 0.2, exactamente K, porque el filtro arrancó en su estado estable. Después: μ = 0 + 0.2 × 0.5 = "
            "0.1, P2 = 0.8 × 0.25 = 0.2 y R2 = 0.8 × 1 + 0.2 × 0.25 = 0.85. En los periodos 2 y 3 R2 baja y la "
            "ganancia sube por arriba de 0.2: P2 todavía trae la escala anterior. Es lo que se ve en la ganancia real "
            "del filtro 2 cuando cambia la volatilidad.")]
    s += N.tabla("Filtro 2 paso a paso (salida de <i>correr_filtro_spread</i>).",
                 tabla(filas, [0.6 * cm] + [1.49 * cm] * 11, derecha=tuple(range(12))))
    # sistema completo: una operacion real
    op, se, ss = ex["op"], ex["senal"], ex["senal_salida"]
    ent, sal = ex["entrada"], ex["salida"]
    V, bta = ex["V"], float(se["beta"])
    N_amt, N_sb = 0.8 * V / (1 + bta), -0.8 * V * bta / (1 + bta)
    d1 = ex["diario_dia1"]
    com_ent, com_sal, com_reb = float(ent["comision"].sum()), float(sal["comision"].sum()), float(ex["reb"]["comision"].sum())
    direccion = "largo" if op["direccion"] == "largo spread" else "corto"
    s += [*H("4.3 Sistema completo: una operación real", "h2"),
          P(f"Operación {int(op['operacion'])} del Kalman en train ({op['direccion']}), tomada de "
            f"<i>bt_operaciones.csv</i>, <i>bt_fills.csv</i> y <i>bt_diario.csv</i>. Recorre la transición completa: "
            f"filtros, política, ejecución y contabilidad.")]
    filas = [["momento", "qué pasa", "cálculo"],
             [f"Señal, cierre del {ex['fecha_senal'].date()}",
              f"Los filtros entregan s = {num(se['s'], 5)}, μ = {num(se['mu'], 5)} y σ = {num(se['sigma'], 5)}.",
              f"z = ({num(se['s'], 5)} − {num(se['mu'], 5)})/{num(se['sigma'], 5)} = {num(se['z'], 3)}; como "
              f"{num(tk['z_in'], 3)} &lt; |z| ≤ {num(tk['z_stop'], 3)} y z {'&lt;' if se['z'] < 0 else '&gt;'} 0, "
              f"d = {'+1' if se['z'] < 0 else '−1'} ({direccion} el spread)."],
             [f"Ejecución, cierre del {pd.Timestamp(op['fecha_entrada']).date()}",
              f"Se abre con V = {num(V, 2)} y el β previo del día de la señal, {num(bta, 5)}.",
              f"N<sub>AMT</sub> = 0.8 V/(1 + β) = {num(N_amt, 2)}; N<sub>SBAC</sub> = −0.8 V β/(1 + β) = {num(N_sb, 2)}. "
              f"Acciones: {num(ent.loc['AMT', 'cantidad'], 2)} de AMT a {num(ent.loc['AMT', 'precio'], 4)} y "
              f"{num(ent.loc['SBAC', 'cantidad'], 2)} de SBAC a {num(ent.loc['SBAC', 'precio'], 4)}. Comisión: "
              f"0.125% × {num(abs(N_amt) + abs(N_sb), 2)} = {num(com_ent, 2)}."],
             [f"Primer día, {ex['dia1'].date()}",
              "Mark-to-market, préstamo y rebalanceo hacia el β del día anterior.",
              f"P&L de precios = {num(d1['pnl_bruto'], 2)}; préstamo = 0.25% × corto × {ex['dias_naturales_1']}/365 = "
              f"{num(d1['prestamo'], 2)}; rebalanceo a β = {num(ex['beta_dia1'], 5)} con comisión {num(d1['comision'], 2)}."],
             [f"Salida, señal del {ex['fecha_senal_salida'].date()}",
              f"z = {num(ss['z'], 3)}.",
              f"|z| &lt; z<sub>out</sub> = {num(tk['z_out'], 3)}: cierre por {op['motivo_cierre']}, ejecutado el "
              f"{pd.Timestamp(op['fecha_salida']).date()} con comisión {num(com_sal, 2)}."],
             ["Resultado",
              f"{int(op['dias'])} días hábiles, {int(op['n_rebalanceos'])} días con rebalanceo.",
              f"P&L bruto {num(op['pnl_bruto'], 2)} − comisiones {num(op['comisiones'], 2)} (entrada {num(com_ent, 2)}, "
              f"{len(ex['reb'])} ajustes de rebalanceo {num(com_reb, 2)}, salida {num(com_sal, 2)}) − préstamo "
              f"{num(op['prestamo'], 2)} = {num(op['pnl_neto'], 2)}, un {pct(op['retorno'], 2)} del capital de entrada."]]
    s += N.tabla("Una operación real, de la señal al P&L neto.", tabla(filas, [3.3 * cm, 5 * cm, 8.7 * cm], texto=True))

    # ============================================================== 5. Q y R
    mejor_d = cal.loc[cal.groupby("delta")["log_verosimilitud"].idxmax()]
    libre = cal.loc[cal["log_verosimilitud"].idxmax()]
    perm = cal[a_bool(cal["max_LL_permitido"])].iloc[0]
    m6 = mom.set_index("delta")
    d_usado, d_otro = kp["delta"], [d for d in m6.index if not np.isclose(d, kp["delta"])][0]
    dif_ll = m6.loc[d_otro, "LL"] - m6.loc[d_usado, "LL"]
    s += [*H("5. Justificación de Q y R", "h1"),
          P("Q dice qué tanto se espera que cambie el estado de un día a otro y R qué tanto ruido trae cada observación. "
            "Su cociente fija la ganancia: con R grande el filtro desconfía de los datos y casi no se mueve; con Q "
            "grande el estado persigue cada dato. En pairs trading eso importa porque un β que persigue a los precios "
            "se come la reversión que se quiere operar."),
          *H("5.1 Filtro 1", "h2"),
          P(f"Q = δ/(1 − δ) I, la parametrización de Chan (2013): δ es la fracción de la varianza que se atribuye al "
            f"movimiento de los pesos. La calibración tiene cuatro pasos. (1) Rejilla de δ en {{10<super>−7</super>, …, "
            f"10<super>−3</super>}} por 10 valores de R entre R<sub>OLS</sub>/100 y R<sub>OLS</sub> = "
            f"{cient(kp['R_ols'], 3)} (la varianza de los residuales del OLS del burn-in), evaluando la log-verosimilitud "
            f"de las innovaciones en train sin burn-in. (2) Cota δ ≤ 10<super>−5</super>: con δ mayor, la verosimilitud "
            f"explica la reversión del spread como movimiento de α y β, justo lo que se quiere operar. (3) R por método "
            f"de momentos para cada δ admisible. (4) El δ final entre 10<super>−6</super> y 10<super>−5</super> lo "
            f"eligió la optimización de θ por Sharpe en train (sección 6).")]
    filas = [["δ", "permitido", "R con máxima verosimilitud", "log-verosimilitud máxima", "nota"]]
    for _, r in mejor_d.iterrows():
        nota = ""
        if np.isclose(r["R"], cal["R"].min()):
            nota = "R en el borde inferior de la rejilla"
        if bool(a_bool(pd.Series([r["max_LL_permitido"]]))[0]):
            nota = "máximo dentro de la cota"
        filas.append([f"{pot10(r['delta'])}", "sí" if a_bool(pd.Series([r["permitido"]]))[0] else "no",
                      cient(r["R"], 2), num(r["log_verosimilitud"], 1), nota])
    s += N.tabla("Mejor R de la rejilla para cada δ (<i>kalman_calibracion.csv</i>).",
                 tabla(filas, [1.8 * cm, 2 * cm, 4 * cm, 3.6 * cm, 5.6 * cm], derecha=(2, 3)))
    s += [N.figura("apoyo_calibracion_filtro1.png", "Log-verosimilitud del filtro 1 en train para cada δ y R. Línea "
                   "punteada: δ fuera de la cota. Estrella: máximo con δ ≤ 10<super>−5</super>. Cuadros: R por método de "
                   "momentos.")]
    filas = [["δ", "R por momentos", "log-verosimilitud", "var(e/√S) train", "test", "val", "curtosis en exceso"]]
    for d, r in m6.iterrows():
        filas.append([f"{pot10(d)}" + (" (usado)" if np.isclose(d, d_usado) else ""), cient(r["R_momentos"], 3),
                      num(r["LL"], 1), num(r["var_train"], 3), num(r["var_test"], 3), num(r["var_val"], 3),
                      num(r["curtosis"], 2)])
    s += [P(f"<b>Por qué R por momentos.</b> Las innovaciones tienen colas pesadas (curtosis en exceso de "
            f"{num(m6.loc[d_usado, 'curtosis'], 2)} con δ = {pot10(d_usado)} y {num(m6.loc[d_otro, 'curtosis'], 2)} con "
            f"{pot10(d_otro)}). Con colas pesadas la verosimilitud gaussiana prefiere una R chica que ajuste bien los días "
            f"normales y tolere los pocos errores grandes, y el filtro termina reportando menos incertidumbre de la real. "
            f"El método de momentos busca por bisección la R con var(e<sub>t</sub>/√S<sub>t</sub>) = 1 en train, para que "
            f"S<sub>t</sub> refleje el tamaño real de los errores. Con δ = {pot10(d_usado)} la R de momentos "
            f"({cient(m6.loc[d_usado, 'R_momentos'], 3)}) es "
            f"{num(m6.loc[d_usado, 'R_momentos'] / mejor_d.set_index('delta').loc[d_usado, 'R'], 1)} veces la de "
            f"máxima verosimilitud en la rejilla.")]
    s += N.tabla("R por método de momentos (<i>kalman_calibracion_momentos.csv</i>).",
                 tabla(filas, [2.6 * cm, 2.6 * cm, 3.0 * cm, 2.6 * cm, 1.6 * cm, 1.6 * cm, 2.8 * cm],
                       derecha=(1, 2, 3, 4, 5, 6)))
    s += [P(f"<b>Lo que dice la evidencia sobre δ.</b> El máximo sin restricción está en δ = {pot10(libre['delta'])} con R "
            f"en el borde inferior de la rejilla ({cient(libre['R'], 2)}): la verosimilitud quiere pesos que se muevan "
            f"rápido y no tiene un óptimo interior. Dentro de la cota prefiere δ = {pot10(perm['delta'])}. Con R por "
            f"momentos, δ = {pot10(d_otro)} tiene {num(dif_ll, 1)} puntos más de log-verosimilitud que el δ usado "
            f"({pot10(d_usado)}). El δ = {pot10(d_usado)} lo eligió la optimización de θ por Sharpe en train, no la "
            f"estadística del filtro: da un β más estable y menos rebalanceo, pero es un parámetro más ajustado al P&L "
            f"de train. Que los datos pidan pesos más móviles de lo que una cointegración estable permite es, en sí, "
            f"evidencia de que la relación no es fija."),
          *H("5.2 Filtro 2", "h2"),
          P("Si Q2 = q R2 y R2 se mantiene, la ganancia converge a un valor fijo. Con p = P2<sub>t|t−1</sub>/R2, en estado "
            "estable K = p/(p + 1) y P2<sub>t|t</sub>/R2 = (1 − K) p = K, así que p = K + q. Despejando:"),
          ecuacion(r"(16)\quad K^2 = q\,(1-K)\ \ \Longrightarrow\ \ q = \dfrac{K^2}{1-K},\qquad P2_{t|t-1} = \dfrac{K}{1-K}\, R2"),
          P(f"Con ganancia fija, μ<sub>t|t</sub> = μ<sub>t|t−1</sub> + K (s<sub>t</sub> − μ<sub>t|t−1</sub>) es un promedio "
            f"exponencial con memoria de 1/K días. Se fija K = 1/L con L = round(2 × {chosen['vida_media']}) = {kp['L']} "
            f"días, dos vidas medias, para que el filtro 2 recuerde lo mismo que la ventana móvil del estático; λ = 1 − 1/L "
            f"da la misma memoria a la escala. Así Q2 queda determinada por la memoria elegida y R2 por los datos. Una "
            f"primera versión con Q2 fija y grande hacía que μ persiguiera al spread y las operaciones duraban unos cinco "
            f"días, contra una vida media de 41.")]
    gan = {m: float(np.median(f2["K2"].to_numpy()[(sen["muestra"].to_numpy() == m) & ~burn])) for m in MUESTRAS}
    s += N.tabla("Parámetros del filtro 2 (<i>kalman_params.json</i>) y ganancia observada.", tabla(
        [["parámetro", "valor", "de dónde sale"],
         ["L", str(kp["L"]), f"2 × vida media de train ({chosen['vida_media']} días)"],
         ["K = 1/L", num(K_obj, 5), "ganancia objetivo"],
         ["q", cient(q, 3), "K²/(1 − K), ecuación 16"],
         ["λ", num(lam, 5), "1 − 1/L"],
         ["μ<sub>0</sub>, R2<sub>0</sub>", f"{num(kp['mu_0'], 4)}, {cient(kp['R2_0'], 3)}",
          "media y varianza del spread del burn-in con el β del OLS"],
         ["P2<sub>0</sub>", cient(kp["P2_0"], 3), "K R2<sub>0</sub>, estado estable"],
         ["mediana de K2", f"{num(gan['train'], 4)} / {num(gan['test'], 4)} / {num(gan['val'], 4)}",
          "train / test / val, cerca de 1/L"]],
        [3.4 * cm, 4.6 * cm, 9 * cm]))
    f1t, f2t = dg[("Filtro 1", "train")], dg[("Filtro 2", "train")]
    s += [*H("5.3 Prueba de las innovaciones (prueba 4)", "h2"),
          P("Si un filtro está bien especificado, sus innovaciones estandarizadas son ruido blanco con media 0 y "
            "varianza 1. En el filtro 2 la innovación estandarizada es el propio z-score.")]
    filas = [["filtro", "muestra", "días", "media", "varianza", "ρ<sub>1</sub>", "Ljung-Box (10) p", "curtosis en exceso"]]
    for _, r in diag.iterrows():
        filas.append([r["filtro"], r["muestra"], f"{r['n']:,}", num(r["media"], 3, True), num(r["var"], 3),
                      num(r["rho1"], 3), "&lt; 0.001" if r["lb10"] < 0.001 else num(r["lb10"], 3), num(r["curtosis"], 2)])
    s += N.tabla("Innovaciones estandarizadas (filtro 1: e<sub>t</sub>/√S<sub>t</sub>; filtro 2: z<sub>t</sub>), después "
                 "del burn-in.", tabla(filas, [2 * cm, 1.8 * cm, 1.5 * cm, 1.7 * cm, 1.9 * cm, 1.6 * cm, 3 * cm, 3.5 * cm],
                                       derecha=(2, 3, 4, 5, 6, 7)))
    s += [P(f"La varianza pasa en train: en el filtro 1 por construcción ({num(f1t['var'], 3)}, porque R se calibró para "
            f"eso) y en el filtro 2 queda en {num(f2t['var'], 3)}. La parte de ruido blanco no pasa: la autocorrelación "
            f"de orden 1 es {num(f1t['rho1'], 2)} en el filtro 1 y {num(f2t['rho1'], 2)} en el filtro 2, y Ljung-Box "
            f"rechaza en todas las muestras. La causa es de especificación: los dos filtros modelan el estado como paseo "
            f"aleatorio más ruido blanco, pero la desviación del spread respecto a su equilibrio es persistente (vida "
            f"media de {chosen['vida_media']} días) y ese componente no está en el estado, así que pasa a la "
            f"innovación. En el filtro 2 esa persistencia es justo la señal que se opera, pero un filtro con innovaciones "
            f"autocorrelacionadas no está bien especificado y su S<sub>t</sub> no mide bien la incertidumbre. La prueba "
            f"automática (<i>tests/test_calibracion_filtro.py</i>) solo revisa media y varianza, por eso pasa."),
          N.figura("apoyo_innovaciones_acf.png", "Autocorrelación de las innovaciones estandarizadas en train, 20 "
                   "rezagos. La banda es el intervalo de 95% para ruido blanco.")]

    # ============================================================== 6. optimizacion
    fk, fe_ = int(a_bool(gk["factible"]).sum()), int(a_bool(ge["factible"]).sum())
    top_k = gk[a_bool(gk["factible"])].nlargest(5, "sharpe")
    s += [*H("6. Optimización de θ (paso 6 de la política)", "h1"),
          P(f"θ se busca con optimización bayesiana (Optuna, muestreador TPE con semilla 42), "
            f"{len(gk)} evaluaciones por método, solo en train después del burn-in. Cada evaluación aplica las reglas "
            f"con un θ candidato y mide el Sharpe neto con el evaluador simple de <i>strategy/evaluador.py</i> (mismos "
            f"costos que el backtest, acciones fijas durante la operación). El objetivo es el Sharpe si hay al menos 10 "
            f"operaciones; si no, −10 + 0.01 n<sub>ops</sub>, que empuja la búsqueda hacia configuraciones que operen. "
            f"El TPE ajusta una densidad a los mejores trials y otra al resto, y propone el punto que más separa las dos: "
            f"concentra la búsqueda donde ya encontró buenos resultados.")]
    s += N.tabla("Espacio de búsqueda (<i>strategy/grid_search.py</i>).", tabla(
        [["parámetro", "Kalman", "Estático", "tipo"],
         ["z<sub>in</sub>", "[1.5, 2.5]", "[1.5, 2.5]", "continuo"],
         ["z<sub>out</sub>", "[0, 0.5]", "[0, 0.5]", "continuo"],
         ["z<sub>stop</sub>", "[3, 4]", "[3, 4]", "continuo"],
         ["h<sub>max</sub>", "41 a 124 días", "41 a 124 días", "entero (1 a 3 vidas medias)"],
         ["δ (filtro 1)", "{10<super>−6</super>, 10<super>−5</super>}", "no aplica", "categórico, cada uno con su R"],
         ["L (ventana)", f"fija en {kp['L']} (filtro 2)", "41 a 124 días", "entero"]],
        [3 * cm, 4.4 * cm, 4.4 * cm, 5.2 * cm]))
    s += N.tabla("θ* congelado (<i>theta_optimo.json</i>).", tabla(
        [["método", "z<sub>in</sub>", "z<sub>out</sub>", "z<sub>stop</sub>", "h<sub>max</sub>", "otros",
          "Sharpe en train (evaluador)", "operaciones", "trials con 10 o más"],
         ["Kalman", num(tk["z_in"], 4), num(tk["z_out"], 4), num(tk["z_stop"], 4), str(tk["h_max"]),
          f"δ = {pot10(tk['delta'])}", num(tk["sharpe_train"], 3), str(tk["n_ops_train"]), f"{fk} de {len(gk)}"],
         ["Estático", num(te["z_in"], 4), num(te["z_out"], 4), num(te["z_stop"], 4), str(te["h_max"]),
          f"L = {te['L']}", num(te["sharpe_train"], 3), str(te["n_ops_train"]), f"{fe_} de {len(ge)}"]],
        [1.7 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.3 * cm, 1.8 * cm, 3.2 * cm, 2 * cm, 2.8 * cm],
        derecha=(1, 2, 3, 4, 6, 7)))
    s += [N.figura("apoyo_optimizacion_theta.png", "Trayectoria de la optimización bayesiana: Sharpe en train de "
                   "cada trial y el mejor encontrado hasta ese momento."),
          P(f"Dos observaciones. Primera, los mejores trials del Kalman son casi el mismo punto: los cinco primeros "
            f"tienen z<sub>in</sub> entre {num(top_k['z_in'].min(), 2)} y {num(top_k['z_in'].max(), 2)}, h<sub>max</sub> "
            f"entre {int(top_k['h_max'].min())} y {int(top_k['h_max'].max())} y Sharpe entre "
            f"{num(top_k['sharpe'].min(), 3)} y {num(top_k['sharpe'].max(), 3)}; elegir entre ellos es elegir ruido. "
            f"Segunda, el óptimo se encontró cerca del final de la búsqueda y depende del camino que siguió el TPE.")]
    ref = OTRA_CORRIDA["referencia"]
    if np.isclose(tk["z_in"], ref["kalman"]) and np.isclose(te["z_in"], ref["estatico"]):
        filas = [["método", "corrida", "θ* (z<sub>in</sub>, z<sub>out</sub>, z<sub>stop</sub>, h<sub>max</sub>)", "otros",
                  "Sharpe train", "test", "val"]]
        for m, t in [("kalman", tk), ("estatico", te)]:
            oc = OTRA_CORRIDA[m]
            filas.append([METODOS[m], "commit actual", f"{num(t['z_in'], 2)}, {num(t['z_out'], 2)}, "
                          f"{num(t['z_stop'], 2)}, {t['h_max']}", f"δ = {pot10(t['delta'])}" if m == "kalman" else f"L = {t['L']}",
                          num(m_of[(m, 'train')]['sharpe'], 3), num(m_of[(m, 'test')]['sharpe'], 3),
                          num(m_of[(m, 'val')]['sharpe'], 3)])
            filas.append([METODOS[m], "otra computadora", ", ".join(num(v, 2) if i < 3 else str(v)
                                                                     for i, v in enumerate(oc["theta"])),
                          oc["extra"]] + [num(v, 3) for v in oc["sharpe"]])
        s += [P("<b>Sensibilidad a la plataforma.</b> Al correr <i>python main.py</i> en otra computadora (Linux, Python "
                "3.13) con las versiones fijadas en <i>requirements.txt</i> y los mismos precios, los trials coinciden con "
                f"los del commit hasta el {OTRA_CORRIDA['kalman']['coinciden_hasta']} en el Kalman y el "
                f"{OTRA_CORRIDA['estatico']['coinciden_hasta']} en el estático, aunque sus Sharpe ya difieren en el orden de "
                "10<super>−14</super> por redondeo. En el trial siguiente esa diferencia basta para que el TPE proponga otro "
                "punto, y la búsqueda termina en otro θ*. Cada "
                "corrida es reproducible en su propia máquina, pero el θ* no es estable entre máquinas, y con él "
                "cambian los resultados fuera de muestra (Sharpe del backtest oficial):")]
        s += N.tabla("Mismo código y mismos datos, dos computadoras.", tabla(
            filas, [1.7 * cm, 2.7 * cm, 3.8 * cm, 1.9 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm], derecha=(4, 5, 6)))
        s += [P("Una búsqueda cuyo resultado cambia con diferencias de redondeo está optimizando ruido: el Sharpe en train "
                "es casi plano en una región amplia de θ y el punto elegido es arbitrario dentro de ella. Es la misma "
                "señal de sobreajuste que muestra el contraste entre train y test.")]

    # ============================================================== 7. backtest
    s += [*H("7. Backtest y resultados", "h1"),
          P("El backtest oficial (<i>backtest/motor.py</i>) es independiente del evaluador de la optimización:")]
    s += items([
        f"Capital inicial de {num(cfg['capital'], 0)} en cada muestra, que arranca sin posición; train arranca después "
        f"del burn-in. Al final de cada muestra se liquida lo abierto.",
        f"Exposición bruta de {num(100 * cfg['exposicion'], 0)}% repartida según el vector de pesos (ecuación 12). La "
        f"señal del cierre de t se ejecuta al cierre de t + 1.",
        f"Comisión de {num(100 * cfg['comision'], 3)}% del nocional por pata al entrar, salir y rebalancear. Préstamo de "
        f"{num(100 * cfg['tasa_prestamo'], 2)}% anual sobre el nocional corto del cierre anterior, por día natural.",
        "Política oficial: rebalanceo diario hacia los pesos del día. También se corren bandas de 1%, 2% y 5% y sin "
        "rebalanceo para medir su costo.",
    ])
    filas = [["método", "muestra", "Sharpe", "Sortino", "Calmar", "máx. DD", "win rate", "profit factor", "retorno",
              "operaciones"]]
    for m in METODOS:
        for mu in MUESTRAS + ["completa"]:
            r = m_of[(m, mu)]
            filas.append([METODOS[m], mu, num(r["sharpe"]), num(r["sortino"]), num(r["calmar"]), pct(r["max_drawdown"]),
                          pct(r["win_rate"], 0), num(r["profit_factor"]), pct(r["retorno_total"]), str(int(r["n_operaciones"]))])
    s += N.tabla("Métricas netas con rebalanceo diario (<i>bt_metricas.csv</i>). <i>completa</i> encadena las tres "
                 "muestras.", tabla(filas, [1.7 * cm, 1.7 * cm, 1.4 * cm, 1.5 * cm, 1.4 * cm, 1.7 * cm, 1.6 * cm,
                                            2 * cm, 1.7 * cm, 2.3 * cm], derecha=tuple(range(2, 10))))
    s += [P(f"En train el estático supera al Kalman (Sharpe {num(m_of[('estatico', 'train')]['sharpe'])} contra "
            f"{num(m_of[('kalman', 'train')]['sharpe'])}) y fuera de muestra también: en test el Kalman pierde "
            f"({pct(m_of[('kalman', 'test')]['retorno_total'])}) y el estático apenas gana "
            f"({pct(m_of[('estatico', 'test')]['retorno_total'])}). Con {int(m_of[('kalman', 'test')]['n_operaciones'])} "
            f"operaciones en test y {int(m_of[('kalman', 'val')]['n_operaciones'])} en val, las métricas del Kalman fuera "
            f"de muestra tienen muy poca precisión."),
          N.figura("fig7_capital.png", "Capital por muestra con rebalanceo diario; cada muestra arranca en 1,000,000.",
                   enunciado="7. Curvas de capital (train, test, val)"),
          N.figura("fig8_retornos_operacion.png", "Distribución del retorno neto por operación, todas las muestras.",
                   enunciado="8. Distribución de retornos por operación")]
    filas = [["método", "muestra", "P&L bruto", "comisiones", "de ellas, rebalanceo", "préstamo", "P&L neto"]]
    for m in METODOS:
        for mu in MUESTRAS:
            r = m_of[(m, mu)]
            filas.append([METODOS[m], mu, num(r["pnl_bruto"], 0), num(r["comisiones"], 0),
                          num(r["comisiones_rebalanceo"], 0), num(r["prestamo"], 0), num(r["pnl_neto"], 0)])
    s += N.tabla("Del P&L bruto al neto (USD).", tabla(filas, [2 * cm, 1.8 * cm, 2.6 * cm, 2.5 * cm, 3.2 * cm,
                                                                2.2 * cm, 2.7 * cm], derecha=(2, 3, 4, 5, 6)))
    s += [N.figura("extra_costos.png", "Atribución de costos en las tres muestras juntas: las comisiones se comen "
                   f"{pct(m_of[('kalman', 'completa')]['comisiones'] / m_of[('kalman', 'completa')]['pnl_bruto'], 0)} del P&L "
                   f"bruto del Kalman y {pct(m_of[('estatico', 'completa')]['comisiones'] / m_of[('estatico', 'completa')]['pnl_bruto'], 0)} "
                   "del estático; el préstamo es marginal.", enunciado="Extra: atribución de costos"),
          N.figura("extra_sharpe_movil.png", "Sharpe móvil de 252 días sobre la curva encadenada.",
                   enunciado="Extra: Sharpe móvil de 252 días")]
    rbx = reb.set_index(["metodo", "politica", "muestra"])
    filas = [["método", "política", "rebalanceos (train)", "P&L neto contra no rebalancear: train", "test", "val"]]
    for m in METODOS:
        for pol in ["diario", "banda_1", "banda_2", "banda_5"]:
            filas.append([METODOS[m], pol.replace("_", " ") + ("%" if pol != "diario" else ""),
                          str(int(rbx.loc[(m, pol, "train"), "n_rebalanceos"]))] +
                         [num(rbx.loc[(m, pol, mu), "dif_pnl_neto_vs_sin"], 0, True) for mu in MUESTRAS])
    s += [*H("7.1 Costo del rebalanceo", "h2"),
          P("Seguir los pesos cada día cuesta comisiones; una banda solo ajusta cuando la mezcla se aleja más que la "
            "banda. Diferencia de P&L neto contra no rebalancear nunca (USD):")]
    s += N.tabla("Políticas de rebalanceo (<i>bt_rebalanceo.csv</i>).", tabla(
        filas, [2 * cm, 2 * cm, 3 * cm, 4.6 * cm, 2.7 * cm, 2.7 * cm], derecha=(2, 3, 4, 5)))
    dif_e = rbx.loc[("estatico", "diario", "train"), "dif_pnl_neto_vs_sin"]
    sin_e = rbx.loc[("estatico", "sin_rebalanceo", "train"), "pnl_neto"]
    s += [P(f"En el estático, rebalancear a diario cuesta {num(-dif_e, 0)} USD en train, el {pct(-dif_e / sin_e, 0)} del "
            f"P&L neto sin rebalanceo; en el Kalman las diferencias son menores porque opera menos días. Ninguna banda "
            f"domina en las tres muestras. Se mantiene el rebalanceo diario como oficial porque es el que replica el "
            f"vector de la estrategia."),
          *H("7.2 Dependencia de un evento", "h2"),
          P(f"El {EVENTO} SBAC subió 18.9% por reportes de una posible venta de la empresa. El Kalman estaba "
            f"{lado_evento} y ese día ganó {num(ev['kalman']['dia'], 0)} USD, el {pct(ev['kalman']['frac'], 0)} de su "
            f"P&L de val; sin ese día su retorno en val sería {pct(ev['kalman']['ret_sin'])} en lugar de "
            f"{pct(m_of[('kalman', 'val')]['retorno_total'])}. Para el estático ese día es el "
            f"{pct(ev['estatico']['frac'], 0)} del P&L de val. Ese movimiento es una noticia corporativa, no una "
            f"reversión del spread: el resultado de val depende de un evento que la estrategia no predice.")]

    # ============================================================== 8. pruebas
    s += [*H("8. Pruebas automáticas", "h1")]
    s += N.tabla("Pruebas del enunciado y su estado (<i>python -m pytest</i>).", tabla(
        [["prueba", "qué se verifica en el repositorio", "estado"],
         ["1. Causalidad", "<i>test_causalidad.py</i>: al cortar los datos a la mitad de test, z, β y la señal antes del "
                           "corte son idénticos a los de la corrida completa, en Kalman y estático.", "pasa"],
         ["2. Normalización", "No hay prueba automática. La normalización se cumple por construcción: AMT = 1 en el "
                              "vector de Johansen (<i>normalizar_vector</i>) y w = (1, −β) en el backtest.",
          "sin prueba"],
         ["3. Contabilidad", "<i>test_contabilidad.py</i>: identidades de efectivo, posiciones, P&L y costos en un caso "
                             "sintético de 3 patas y con las señales reales, tamaño y signos al entrar, y que sin "
                             "rebalanceo el backtest reproduce al evaluador simple.", "pasa"],
         ["4. Filtros", "<i>test_calibracion_filtro.py</i>: media y varianza de las innovaciones en train (y varianza "
                        "del filtro 1 en test). No revisa autocorrelación; con ella la prueba no pasaría (sección 5.3).",
          "pasa media y varianza; falla ruido blanco"]],
        [2.9 * cm, 10.5 * cm, 3.6 * cm], texto=True))

    # ============================================================== 9. discusion
    s += [*H("9. ¿Relación estructural o artefacto?", "h1"),
          P("La revisión del avance señaló que el modelo se sobreajusta: se rompe en test, y al re-entrenar solo con una "
            "parte de train vuelve a romperse en la parte que queda fuera. Los resultados de este reporte lo confirman. A favor de una relación real hay una razón "
            "económica previa (mismo negocio, mismos clientes), Johansen con rango 1 y ADF significativo en train, un VECM "
            "en el que SBAC corrige el desequilibrio con una velocidad coherente con la vida media, y un spread que vuelve "
            "a ser estacionario en val con el vector de train. En contra:")]
    contra = [
        f"AMT y SBAC no pasan Engle-Granger al 5% (p = {num(par['EG_p_AB'], 3)} y {num(par['EG_p_BA'], 3)}).",
        f"Se probaron {len(v1) + len(v2)} grupos en dos universos, y el universo se cambió después de ver resultados. "
        f"Pasan Johansen 3 de 57 grupos del universo final, cuando el azar daría {num(fpt['Johansen']['FP_esperados'], 2)}.",
        f"El vector se rompe fuera de muestra (ADF p = {num(o['test']['ADF_spread_p'], 2)} en test) y también dentro de "
        f"train (sub-test p = {num(rb['sub_test']['ADF_spread_p'], 2)}, sub-val p = {num(rb['sub_val']['ADF_spread_p'], 2)}).",
        f"El Johansen móvil no es estable: el peso de SBAC cambia de signo en {positivos} de {len(eig)} ventanas.",
        f"La verosimilitud del filtro 1 pide pesos más móviles que los usados (δ = {pot10(d_otro)} supera a {pot10(d_usado)} por "
        f"{num(dif_ll, 0)} puntos de log-verosimilitud), y sin la cota pediría todavía más.",
        "Las innovaciones de los dos filtros están autocorrelacionadas: la prueba 4 no se cumple en su parte de ruido "
        "blanco.",
        f"θ* depende de diferencias de redondeo en la optimización, y los mejores trials son prácticamente empates.",
        f"Val depende de un día: el {EVENTO} explica el {pct(ev['kalman']['frac'], 0)} del P&L del Kalman en val.",
    ]
    s += items(contra)
    s += [P("Con esta evidencia no se puede distinguir la relación de un artefacto de pruebas múltiples con periodos "
            "favorables. El sistema sí cumple lo que promete en su construcción: es causal, la contabilidad cuadra y los "
            "filtros reproducen sus ecuaciones. Lo que no sostiene es la premisa estadística de un equilibrio estable."),
          *H("Limitaciones", "h2")]
    s += items([
        "Un solo grupo: sin un segundo grupo no hay diversificación.",
        "Ejecución al cierre de t + 1 sin impacto de mercado; costo de préstamo fijo.",
        "Los filtros suponen errores gaussianos; con colas pesadas un filtro robusto reaccionaría menos a días como el "
        f"{EVENTO}.",
        "θ se optimizó contra un evaluador simple sin rebalanceo y el backtest oficial rebalancea a diario; la diferencia "
        "en train es chica, pero existe.",
    ])

    # ============================================================== 10. reproducibilidad
    s += [*H("10. Reproducibilidad", "h1"),
          P("Desde la raíz del repositorio, con Python 3.11 o superior: <i>pip install -r requirements.txt</i>; "
            "<i>python main.py</i> corre los 11 pasos con los precios guardados y semilla 42; <i>python -m pytest</i> "
            "corre las pruebas; <i>streamlit run dashboard/app.py</i> abre el dashboard; y <i>python -m "
            "performance.reporte</i> regenera las figuras de <i>docs/figuras/</i> y este documento desde "
            "<i>artifacts/</i>. Como se explica en la sección 6, otra computadora puede llegar a otro θ*; los artifacts "
            "del repositorio son los de la corrida oficial. El uso de herramientas de IA está documentado en el README.")]

    # ============================================================== anexos
    s += [PageBreak(), *H("Anexo A. Los 57 grupos del universo final", "h1"),
          P("Ordenados por la etapa del embudo donde caen y por ADF. Rango por traza / por máximo eigenvalor al 95%; "
            "la vida media es del spread con el primer eigenvector, en días hábiles.")]
    filas = [["grupo", "n", "corr. mínima", "rango", "bandera roja", "ADF p", "vida media", "etapa donde cae"]]
    for _, r in v2.iterrows():
        filas.append([r["grupo"].replace("+", " + "), str(r["n"]), num(r["corr_min_par"], 3),
                      f"{r['rank_traza']} / {r['rank_maxeig']}",
                      "sí" if a_bool(pd.Series([r["bandera_roja"]]))[0] else "no", num(r["ADF_spread_p"], 4),
                      num(r["vida_media"], 1), "pasa" if r["etapa_caida"] == "PASA" else r["etapa_caida"]])
    s += N.tabla("Embudo completo del universo final (<i>screening_all_groups.csv</i>).",
                 tabla(filas, [5 * cm, 0.8 * cm, 1.9 * cm, 1.4 * cm, 1.9 * cm, 1.6 * cm, 1.8 * cm, 2.6 * cm],
                       derecha=(1, 2, 5, 6)))
    s += [*H("Anexo B. Bitácora de operaciones", "h1"),
          P("Backtest oficial con rebalanceo diario; fechas de ejecución.")]
    filas = [["método", "muestra", "#", "dirección", "entrada", "salida", "días", "motivo", "P&L neto", "retorno",
              "rebal."]]
    for _, r in ops.iterrows():
        filas.append([METODOS[r["metodo"]], r["muestra"], str(int(r["operacion"])), r["direccion"],
                      str(r["fecha_entrada"])[:10], str(r["fecha_salida"])[:10], str(int(r["dias"])),
                      str(r["motivo_cierre"]).replace("_", " "), num(r["pnl_neto"], 0), pct(r["retorno"], 2),
                      str(int(r["n_rebalanceos"]))])
    s += N.tabla("Operaciones de los dos métodos (<i>bt_operaciones.csv</i>).",
                 tabla(filas, [1.5 * cm, 1.5 * cm, 0.7 * cm, 2.1 * cm, 1.9 * cm, 1.9 * cm, 0.9 * cm, 1.9 * cm, 1.8 * cm,
                               1.5 * cm, 1.2 * cm], derecha=(2, 6, 8, 9, 10)))
    s += [KeepTogether([*H("Anexo C. Figuras del enunciado", "h1")] + N.tabla("Dónde está cada figura pedida (en <i>docs/figuras/</i> y en <i>Figuras_Lab03.ipynb</i>).", tabla(
        [["figura del enunciado", "en este reporte", "archivo"]] +
        [[e, f"Figura {n}", a] for e, n, a in sorted(N.mapa, key=lambda x: (not x[0][0].isdigit(), x[0][:2] if x[0][0].isdigit() else '', x[1]))], [8 * cm, 3 * cm, 6 * cm])))]
    return s


def main():
    figuras.main()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(SALIDA), pagesize=letter, leftMargin=2.25 * cm, rightMargin=2.25 * cm,
                            topMargin=1.9 * cm, bottomMargin=2.1 * cm, title="Lab 03 MyST, Equipo 5: reporte",
                            author="Isabela Torres-Septien Uribe, Andrea Santoyo Vega, Abdon Islas León")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        contenido = construir()
    doc.build(contenido, onFirstPage=pie_pagina, onLaterPages=pie_pagina)
    print(f"Guardado {SALIDA} ({os.path.getsize(SALIDA) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
