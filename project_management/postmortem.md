# Postmortem (borrador)

Se cierra al terminar el sprint 3. Lo que sigue refleja el estado al 2026-11-08.

## Resumen

El sistema está completo y es causal: datos, selección del grupo, dos filtros de Kalman, política de umbrales, backtest con costos, métricas, dashboard y reporte. El problema está en la premisa estadística: la relación entre AMT y SBAC no es estable. Se rompe en test (ADF p = 0.88), también se rompe dentro de train al re-partirlo, y el θ elegido en train no se sostiene fuera de muestra. La revisión del avance lo diagnosticó como sobreajuste.

## Qué salió bien

- Pipeline modular con un solo punto de entrada (`python main.py`) y contratos claros entre etapas.
- Precios guardados en `artifacts/`: los resultados no dependen de los reajustes de Yahoo.
- Pruebas automáticas de causalidad y contabilidad desde la primera versión.
- Resultados negativos documentados en lugar de ocultos: chequeo fuera de muestra, re-split del train, conteo de falsos positivos.
- Reporte y figuras generados desde `artifacts/`: se actualizan solos cuando cambian los resultados.

## Qué salió mal

| problema | causa raíz | efecto |
|---|---|---|
| Grupo elegido con evidencia débil | se probaron 114 grupos y el universo se cambió después de ver resultados; AMT y SBAC no pasan Engle-Granger al 5% | pasan Johansen 3 de 57 grupos, lo mismo que daría el azar (2.85) |
| θ sobreajustado | el objetivo es el Sharpe de una sola muestra de train, y las diferencias entre los mejores θ son ruido | test negativo para el Kalman y θ* que cambia entre computadoras |
| δ elegido por P&L | se dejó δ dentro de la búsqueda de θ | un parámetro estadístico del filtro quedó ajustado al Sharpe de train |
| Prueba 4 incompleta | el estado de los filtros no incluye la desviación persistente del spread | innovaciones autocorrelacionadas; la prueba automática solo revisa media y varianza |
| Prueba 2 sin automatizar | se dio por cumplida por construcción | no hay evidencia ejecutable de la normalización |
| Documentos desfasados | `docs/powell.md` se escribió con la rejilla y no se actualizó al cambiar a optimización bayesiana | dos versiones distintas del paso 6 en el repositorio |
| Artifacts de varias computadoras | cada integrante regeneró artifacts en su máquina | la corrida oficial no se puede reproducir exacta en otra máquina |

## Lecciones

1. Fijar el diseño de validación (qué muestra decide qué) antes de ver resultados, y contar todos los grupos y configuraciones probadas.
2. Un óptimo que cambia con diferencias de redondeo no es un óptimo: el objetivo debe premiar configuraciones que funcionen en varios periodos, no la mejor en uno.
3. Los parámetros estadísticos de los filtros se calibran con criterios estadísticos; el P&L solo decide la política.
4. Una prueba que solo revisa media y varianza no prueba que un filtro esté bien especificado.
5. Una sola corrida oficial, en una sola computadora, y todo lo demás (dashboard, reporte, README) leído de esos artifacts.

## Acciones

Ver `sprint_03.md`. Al cerrar el proyecto se completa esta sección con lo que se hizo y lo que quedó pendiente.
