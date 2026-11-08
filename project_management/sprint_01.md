# Sprint 1: primera versión completa

| | |
|---|---|
| Periodo | 2026-10-29 al 2026-11-04 |
| Objetivo | Tener el pipeline completo de punta a punta: datos, selección del grupo, filtros, política, backtest, métricas y dashboard |
| Resultado | Cumplido. Primera corrida completa el 2026-11-04 |

## Backlog y estado

| tarea | responsable | estado | evidencia (commit) |
|---|---|---|---|
| Estructura del repositorio | Abdon | hecho | 3927ee5 |
| Descarga, revisión de huecos y saltos, split 60/20/20 | Isabela | hecho | 3535f26 |
| Orden de integración (ADF y KPSS) | Isabela | hecho | 3535f26 |
| Screening del universo v1 (VZ, T, BCE, DIS, AMX, NFLX) | Isabela | hecho, ningún grupo pasa | 3535f26, 31db627 |
| Johansen móvil y VECM | Isabela | hecho | 3535f26 |
| Versiones fijas en `requirements.txt` | Andrea | hecho | 04ee5f7 |
| Universo v2 (torres y telefónicas), embudo corregido, grupo AMT y SBAC, chequeo fuera de muestra | Andrea | hecho | b4bc0ef |
| Filtro 1 (pesos) y filtro 2 (spread), reglas PFA, línea base estática | Andrea | hecho | 50ccd95 |
| Pruebas de causalidad y de calibración de los filtros | Andrea | hecho | 50ccd95 |
| Búsqueda de θ en rejilla, θ congelado y métricas preliminares | Andrea | hecho | d048feb |
| Documento de Powell y nota de entrega al integrante C | Andrea | hecho | fe71ff4 |
| Motor de backtest multi-pata con políticas de rebalanceo | Abdon | hecho | 753ce5d, c4be3d9 |
| Métricas y costo del rebalanceo | Abdon | hecho | 71f8c77, f53cc74 |
| Prueba de contabilidad multi-pata | Abdon | hecho | 36a88e9 |
| `main.py` con el pipeline completo | Abdon | hecho | 0ae54bb |
| Dashboard de Streamlit | Abdon | hecho | f067c22, 7df3807 |
| Corrida completa y resultados | Abdon, Isabela | hecho | 850be9a, 7ef039e |

## Decisiones

- Cambio de universo: en v1 ningún grupo pasó la correlación ni Engle-Granger, así que se pasó a torres y telefónicas, que tienen un vínculo económico directo. Los 57 grupos de v1 cuentan como probados.
- Grupo elegido: AMT y SBAC, el único de los tres que pasan el embudo sin bandera roja.
- Correlación y Engle-Granger quedan como prefiltro informativo; el embudo bloqueante es Johansen, ADF del spread y vida media.
- R del filtro 1 por método de momentos y δ acotado a 10⁻⁵; el filtro 2 con ganancia 1/L, L = 83 días, para tener la misma memoria que el estático.
- La señal del cierre de t se ejecuta al cierre de t + 1 y cada muestra arranca plana con 1,000,000.

## Resultado al cierre

El pipeline corre completo y las pruebas de causalidad, calibración y contabilidad pasan. El chequeo fuera de muestra ya mostraba que el spread de AMT y SBAC deja de ser estacionario en test (ADF p = 0.88).

## Retrospectiva

- Funcionó: dividir el trabajo por etapas con un contrato claro entre ellas (la nota de entrega del integrante B al C fijó columnas, convenciones de tiempo y métricas de referencia).
- Funcionó: guardar los precios en `artifacts/` para no depender de los reajustes de Yahoo.
- No funcionó: el universo se cambió después de ver resultados, lo que aumenta el riesgo de quedarnos con un grupo elegido por azar.
- No funcionó: θ se eligió solo con el Sharpe de train, sin medir qué tan estable era ese óptimo.
- No funcionó: casi todo se subió directo a `main`, con un solo pull request.
