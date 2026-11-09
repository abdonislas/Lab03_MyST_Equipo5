# Project charter: Lab 03 MyST, Equipo 5

Microestructura y Sistemas de Trading, ITESO.

| | |
|---|---|
| Proyecto | Arbitraje estadístico con cointegración y filtros de Kalman |
| Equipo | Isabela Torres-Septien Uribe, Andrea Santoyo Vega, Abdon Islas León |
| Repositorio | https://github.com/abdonislas/Lab03_MyST_Equipo5 |
| Inicio | 2026-10-29 (estructura inicial del repositorio) |
| Entrega | Extendida dos semanas después de la revisión del avance |

## Objetivo

Construir y evaluar de forma honesta una estrategia de pairs trading: elegir un grupo de activos cointegrado con un embudo de pruebas, estimar sus pesos y su spread con dos filtros de Kalman, decidir con una política de umbrales, y medir el resultado con un backtest causal y con costos. El sistema se compara contra una línea base con el vector fijo de Johansen.

## Alcance

Incluye:

- Universo de seis acciones de torres y telefónicas (AMT, CCI, SBAC, VZ, T, TMUS), precios diarios del 2015-10-01 al 2026-09-30, partición cronológica 60/20/20.
- Orden de integración, embudo de cointegración sobre los 57 grupos, conteo de falsos positivos esperados, VECM y pruebas de estabilidad fuera de muestra.
- Filtro 1 (pesos α y β) y filtro 2 (nivel y escala del spread), formulados con el marco de Powell.
- Política de umbrales (PFA) con θ = (z_in, z_out, z_stop, h_max) elegido en train y congelado.
- Backtest multi-pata con comisión de 0.125% por pata, préstamo de 0.25% anual sobre los cortos, 80% de exposición y cinco políticas de rebalanceo.
- Métricas (Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor), dashboard de Streamlit y reporte PDF.
- Pruebas automáticas de causalidad, normalización de pesos, contabilidad y calibración de los filtros.

No incluye: más de un grupo, ejecución intradía, impacto de mercado, costo de préstamo variable ni operación en vivo.

## Entregables

| entregable | dónde |
|---|---|
| Pipeline reproducible | `main.py` y módulos en `data/`, `cointegration/`, `kalman/`, `strategy/`, `backtest/`, `performance/` |
| Resultados intermedios y finales | `artifacts/` |
| Reporte PDF (Powell ×3, ejemplos numéricos, Q y R, grupos probados, resultados) | `docs/reporte.pdf`, generado con `python -m performance.reporte` |
| Figuras del enunciado | `docs/figuras/` y `Figuras_Lab03.ipynb` |
| Dashboard | `dashboard/app.py` |
| Pruebas | `tests/` (`python -m pytest`) |
| README con uso de IA y reproducción | `README.md` |
| Gestión | `project_management/` (este charter, sprints y postmortem) |

## Roles

| integrante | rol | responsabilidades |
|---|---|---|
| Isabela Torres-Septien Uribe | A: datos y cointegración | descarga y partición de precios, orden de integración, primer screening (universo v1), Johansen móvil, VECM, re-split del train (`robustez_oos`), optimización bayesiana de θ, notebooks de análisis y figuras, README |
| Andrea Santoyo Vega | B: filtros y política | universo v2 y embudo corregido, elección de AMT y SBAC, chequeo fuera de muestra, filtros de Kalman, reglas PFA, línea base estática, primera búsqueda de θ, pruebas de causalidad y calibración, documento de Powell |
| Abdon Islas León | C: backtest y presentación | estructura del repositorio, motor de backtest y políticas de rebalanceo, métricas, prueba de contabilidad, `main.py`, dashboard, reporte PDF y documentos de gestión |

Las decisiones de modelado (grupo, costos, criterios del embudo, interpretación) se toman en equipo.

## Hitos

| fecha | hito |
|---|---|
| 2026-10-29 | Estructura del repositorio |
| 2026-10-30 | Datos, orden de integración y primer screening (universo v1) |
| 2026-11-01 | Universo v2, grupo AMT y SBAC, filtros, reglas, búsqueda de θ y documento de Powell |
| 2026-11-02 al 04 | Backtest, métricas, pruebas de contabilidad, `main.py` y dashboard |
| 2026-11-04 | Primera corrida completa |
| Después del 2026-11-04 | Revisión del avance: el modelo se sobreajusta (se rompe en test, y al re-entrenar con una parte de train vuelve a romperse) |
| 2026-11-08 | Re-split del train, optimización bayesiana de θ, notebooks, README, reporte y gestión |
| Entrega final | Cambios pendientes del equipo, reporte regenerado y revisión final |

## Criterios de éxito

- `python main.py` corre de punta a punta con los precios guardados y regenera `artifacts/`.
- Nada usa información futura: la prueba de causalidad pasa.
- La contabilidad cuadra: la prueba de contabilidad pasa.
- El reporte cubre lo que pide el enunciado y presenta los resultados como son, incluidos los negativos.
- Cada integrante puede explicar cualquier parte del sistema.

## Riesgos

| riesgo | probabilidad | impacto | efecto | mitigación |
|---|---|---|---|---|
| La relación entre AMT y SBAC no es estable fuera de muestra | alta | alto | pérdidas fuera de muestra | pruebas de estabilidad (oos_check, re-split del train, Johansen móvil) y reporte honesto |
| Sobreajuste de θ y de los parámetros de los filtros | alta | alto | métricas de train que no se repiten | θ elegido solo en train y congelado antes de ver test; mínimo de 10 operaciones |
| Pruebas múltiples (114 grupos y cambio de universo) | media | alto | grupo elegido por azar | conteo de falsos positivos esperados y razón económica previa |
| Divergencia o mala calibración de los filtros de Kalman | media | alto | pesos o z-score sin sentido | burn-in, x centrado, R por momentos, δ acotado y pruebas de calibración |
| Yahoo reajusta precios históricos | media | medio | resultados que cambian solos | snapshot de precios en `artifacts/` |
| Resultados distintos entre computadoras | media | medio | artifacts que no se pueden reproducir exactos | versiones fijas, semilla 42 y θ* congelado por defecto en `main.py` |
| Pocas operaciones fuera de muestra | alta | medio | métricas con poca precisión | reportar el número de operaciones junto a cada métrica |

Probabilidad e impacto en tres niveles: baja, media y alta.
