# Lab 03 MyST: Pairs trading con cointegración y filtros de Kalman

Microestructura y Sistemas de Trading, ITESO. Equipo 5.

## Integrantes

- Isabela Torres
- Andrea Santoyo
- Abdon Islas

## Nivel de alcance

**Nivel avanzado.** Además del alcance base (selección del par por cointegración, estrategia de reversión a la media y backtest con costos), el proyecto incluye:

- Doble filtro de Kalman: pesos dinámicos (α, β) y nivel del spread, comparado contra una línea base estática con el vector de Johansen.
- Optimización bayesiana de θ = (z_in, z_out, z_stop, h_max) con Optuna (TPE), solo en train.
- Backtest multi-pata con cinco políticas de rebalanceo: diario, bandas de 1%, 2% y 5%, y sin rebalanceo.
- Pruebas de robustez fuera de muestra: re-split del train, vector de train evaluado en test y val, y Johansen móvil.
- Formulación de la estrategia con el marco de decisiones secuenciales de Powell ([docs/powell.md](docs/powell.md)).
- Pruebas automáticas de causalidad (sin lookahead), contabilidad y calibración del filtro.
- Dashboard interactivo de una página con vista ejecutiva y vista técnica.

## Descripción del proyecto

Estrategia de *pairs trading* sobre un universo de seis activos de torres de telecomunicaciones y operadores en EE. UU.: AMT, CCI, SBAC, VZ, T y TMUS. Se usan precios de cierre ajustados de Yahoo Finance del 2015-10-01 al 2026-10-01, divididos en orden cronológico en train, test y validación (60/20/20).

El pipeline tiene estas etapas:

1. **Datos.** Descarga, revisión de huecos y de saltos (|z| > 6) contra eventos corporativos conocidos, y split 60/20/20.
2. **Orden de integración.** ADF y KPSS para confirmar que cada serie es I(1).
3. **Screening.** Embudo de correlación móvil, Engle-Granger y Johansen sobre los 57 grupos posibles, con conteo de falsos positivos esperados. Pasan tres grupos; se eligió **AMT + SBAC** (vector `[1, -0.8631]`, ADF del spread p = 0.012, vida media de 41.3 días), el único sin bandera roja y con la mejor justificación económica: ambas son REITs de torres.
4. **Robustez y fuera de muestra.** Re-split del train, evaluación del vector de train en test y val, y primer eigenvector de Johansen en ventanas móviles de 252 días.
5. **VECM** con el vector de Johansen.
6. **Señales.** Filtro 1 para los pesos (α_t, β_t) y filtro 2 para el nivel del spread, con ganancia 1/83 para que tenga la misma memoria que la ventana del modelo estático. El z-score solo usa información hasta t-1.
7. **Optimización de θ** en train, maximizando el Sharpe neto con al menos 10 operaciones.
8. **Backtest y métricas.** Capital de 1,000,000 por muestra, 80% invertido, comisión de 0.125% por pata y costo de préstamo de 0.25% anual sobre los cortos. La señal del cierre de t se ejecuta al cierre de t+1. Se reportan Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor y la atribución de costos.

Hallazgo importante: la relación AMT/SBAC **no es estable fuera de muestra**. El spread deja de ser estacionario en test (ADF p = 0.88, vida media de 150 días) y vuelve a serlo en val. El re-split del train muestra la misma ruptura. Esto se refleja en un Sharpe negativo en test para la estrategia con Kalman.

### Estructura

```
data/            descarga y split de precios
cointegration/   orden de integración, screening, robustez OOS, Johansen móvil
kalman/          filtro 1 (pesos) y filtro 2 (spread)
strategy/        VECM, reglas, modelo estático, generación de señales, optimización de θ
backtest/        motor de backtest multi-pata y políticas de rebalanceo
performance/     métricas de desempeño, figuras y reporte PDF
dashboard/       app de Streamlit (solo lee artifacts/)
tests/           pruebas con pytest
artifacts/       snapshot de precios y todos los resultados intermedios
docs/            reporte (reporte.pdf y figuras/), formulación de Powell y notas de entrega
project_management/  charter, sprints y postmortem
main.py          punto de entrada del pipeline
```

## Instalación

Requisitos: Python 3.11 o superior (probado con 3.12) y git.

```bash
git clone https://github.com/abdonislas/Lab03_MyST_Equipo5.git
cd Lab03_MyST_Equipo5
python -m venv .venv
```

Activar el entorno virtual:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Instalar dependencias (las versiones están fijadas en [requirements.txt](requirements.txt)):

```bash
pip install -r requirements.txt
```

## Reproducción

Desde la raíz del repositorio, con el entorno activado:

```bash
python main.py
```

Este comando corre los 11 pasos del pipeline en unos 20 segundos y regenera todo `artifacts/`. Usa el snapshot de precios guardado en `artifacts/prices_*.csv`, así que no necesita internet y los resultados son idénticos en cada corrida en la misma computadora: la semilla está fijada en 42, incluida la del sampler de Optuna. En otra computadora, diferencias de redondeo pueden llevar a la optimización bayesiana a otro θ\* (sección 6 del reporte); los artifacts del repositorio son los de la corrida oficial.

Opciones:

| Comando | Qué hace |
|---|---|
| `python main.py --desde grid_search` | Corre desde un paso intermedio en adelante (`python main.py --help` muestra los nombres de los pasos) |
| `python main.py --descargar` | Vuelve a descargar los precios con yfinance en lugar de usar el snapshot. Yahoo puede ajustar precios históricos, así que los resultados pueden cambiar |
| `python -m pytest` | Corre las pruebas de causalidad, contabilidad y calibración del filtro |
| `python -m performance.reporte` | Regenera las figuras de `docs/figuras/` y el reporte `docs/reporte.pdf` a partir de `artifacts/` (no vuelve a correr el pipeline) |

## Dashboard

```bash
streamlit run dashboard/app.py
```

Se abre en el navegador en `http://localhost:8501`. El dashboard solo lee `artifacts/` y no vuelve a correr el backtest. Si falta algún archivo, muestra un aviso para correr primero `python main.py`.

### Guía de uso

**Barra lateral (controles globales)**

- **Vista:** *Ejecutiva* para portfolio manager o comité; *Técnica* para quant o gestor de riesgos.
- **Método:** *Kalman (pesos dinámicos)* o *Estático (vector fijo de Johansen)*.
- **Política de rebalanceo:** diario (la oficial), bandas de 1%, 2% o 5%, o sin rebalanceo.
- Debajo aparecen el grupo, el vector de Johansen de train, la vida media, el θ\* del método elegido y los supuestos de costos.

Todas las gráficas sombrean el periodo de cada muestra: gris para burn-in, naranja para test y verde para validación. Train va sin sombra.

**Vista ejecutiva**

1. **Tarjeta de desempeño:** Sharpe, Sortino, Calmar, max drawdown, win rate y profit factor para la muestra elegida (completa, train, test o val). El expander *Todas las muestras en una tabla* las compara lado a lado.
2. **Curva de capital con drawdown.** *Encadenada*: test arranca con el capital final de train y val con el de test. *Por muestra*: cada muestra arranca en 1,000,000.
3. **Atribución de costos:** cascada de P&L bruto a P&L neto (comisiones y préstamo), con el total de comisiones por rebalanceo.
4. **Bitácora de operaciones:** tabla filtrable por muestra, dirección (largo o corto spread), resultado (ganadora o perdedora) y rango de fechas de entrada. Al pie aparecen el P&L neto y el win rate de la selección.
5. **Distribución de retornos por operación:** histograma de la selección de la bitácora con una normal de referencia. Cambia al mover los filtros de la bitácora.

**Vista técnica**

El interruptor *Incluir burn-in* agrega los primeros 126 días del train, que no se operan, a las gráficas de los filtros.

1. **Precios del grupo** en base 100.
2. **Spread S_t con entradas y salidas:** triángulos verdes para largo spread, rojos para corto spread y cruces para las salidas. Al pasar el cursor sobre una salida se ven el motivo de cierre y el retorno.
3. **z-score con umbrales** de entrada, salida y stop del θ\* elegido.
4. **Evolución del vector de pesos**, normalizado en AMT = 1.
5. **Eigenvector de Johansen móvil** (ventanas de 252 días). El interruptor *Escala completa* muestra las ventanas que salen de [-3, 3].
6. **Ganancia de Kalman** de ambos filtros.
7. **Varianza posterior del estado** (escala logarítmica), como diagnóstico de convergencia.
8. **Sharpe móvil de 252 días** sobre la curva encadenada.
9. **Costo del rebalanceo:** tabla por política y muestra con el número de rebalanceos, sus comisiones, la diferencia de P&L neto contra no rebalancear y el Sharpe.

## Uso de Inteligencia Artificial

Usamos asistentes de IA generativa, en su mayoría GitHub Copilot y Claude (Anthropic), como apoyo en cuatro partes del proyecto:

- **Código del pipeline:** escribir y depurar los módulos de datos, cointegración, filtros de Kalman, generación de señales, optimización de θ con Optuna, backtest y métricas.
- **Dashboard:** construir la app de Streamlit ([dashboard/app.py](dashboard/app.py)), con sus gráficas de Plotly y sus filtros interactivos.
- **Notebooks:** generar las gráficas de [Analisis_Lab03.ipynb](Analisis_Lab03.ipynb) y [Figuras_Lab03.ipynb](Figuras_Lab03.ipynb).
- **Reporte y gestión:** escribir el generador del reporte PDF y sus figuras a partir de `artifacts/` ([performance/reporte.py](performance/reporte.py) y [performance/figuras.py](performance/figuras.py)) y redactar los borradores de charter, sprints y postmortem.

Además, con Claude Code:

- Se redactó este README a partir del código y de los resultados del repositorio.
- Se restauró el ciclo principal de [main.py](main.py), que se había borrado por accidente.
- Se agregó `optuna` a [requirements.txt](requirements.txt).
- Se renombró `strategy/VECM.py` a `strategy/vecm.py` para que el import funcione en Windows.
- Se revisó qué pruebas automáticas existen y cuáles faltan.

Las decisiones de modelado son del equipo: la selección del grupo AMT + SBAC, los supuestos de costos y la interpretación de los resultados. El equipo revisó el código generado o sugerido por IA y lo validó con:

- las pruebas automáticas (`python -m pytest`: causalidad, contabilidad y calibración del filtro);
- la reproducibilidad del pipeline (dos corridas de `python main.py` producen artifacts idénticos);
- la comparación entre el backtest y el evaluador independiente de `strategy/`.
