# Sprint 2: respuesta a la revisión del avance

| | |
|---|---|
| Periodo | Después de la revisión del avance al 2026-11-08 |
| Objetivo | Atender la revisión (el modelo se sobreajusta) y completar los entregables que faltaban: reporte PDF, figuras, README y gestión |
| Resultado | Parcial. Los entregables están completos; el sobreajuste está diagnosticado pero no resuelto |

## Contexto

En la revisión del avance se señaló que el modelo se sobreajusta: se rompe en test, y al re-entrenar solo con una parte de train vuelve a romperse en la parte que queda fuera. Se sugirió revisar los hiperparámetros y se permitió usar optimización bayesiana en lugar de la rejilla. La entrega se extendió dos semanas.

## Backlog y estado

| tarea | responsable | estado | evidencia |
|---|---|---|---|
| Re-split del train en sub-train, sub-test y sub-val con el vector de sub-train congelado (`cointegration/robustez_oos.py`) | Isabela | hecho: la relación se rompe en sub-test y sub-val | d6477ca |
| Optimización bayesiana de θ con Optuna (TPE, semilla 42, 150 trials por método) en lugar de la rejilla | Isabela | hecho | 5ba6ea6 |
| Notebooks de análisis y de figuras del enunciado | Isabela | hecho | b55844d, bc09266 |
| Renombrar `strategy/VECM.py` a `strategy/vecm.py` (import en Windows) | Isabela | hecho | 362c67a |
| README con descripción, reproducción, guía del dashboard y uso de IA; ciclo de `main.py` restaurado; `optuna` en requirements | Isabela | hecho | 3b58c03 |
| Reporte PDF generado desde `artifacts/` (`performance/reporte.py`) con Powell ×3, ejemplos numéricos, justificación de Q y R, tabla de grupos probados y resultados | Abdon | hecho | `docs/reporte.pdf` |
| Figuras del reporte (`performance/figuras.py`, `docs/figuras/`) | Abdon | hecho | `docs/figuras/` |
| Charter, sprints y postmortem | Abdon | hecho (postmortem en borrador) | `project_management/` |
| Cambios de Andrea: θ* congelado por defecto en `main.py`, prueba de normalización y causalidad con tres cortes | Andrea | hecho (al cierre del sprint) | 863cc63 |

## Hallazgos del sprint

- El re-split del train confirma la ruptura: con el vector de sub-train, el ADF del spread da p = 0.61 en sub-test y 0.85 en sub-val.
- La optimización bayesiana encontró un θ* distinto al de la rejilla, con Sharpe de train de 0.56 (Kalman) y 0.74 (estático), pero test sigue negativo para el Kalman (Sharpe −0.38).
- Al repetir `python main.py` en otra computadora con las mismas versiones, la búsqueda de θ se separa del commit a partir del trial 43 (Kalman) por diferencias de redondeo y termina en otro θ*, con otros resultados fuera de muestra. El θ* no es estable.
- Las innovaciones de los dos filtros están autocorrelacionadas (ρ₁ de 0.90 y 0.95 en train): la prueba 4 pasa en media y varianza pero no en ruido blanco.
- No hay prueba automática de normalización (prueba 2).
- En val, el alza de SBAC del 2026-04-02 explica el 74% del P&L del Kalman.
- `docs/powell.md` todavía describe la rejilla de 81 combinaciones y el θ anterior en su paso 6.

## Retrospectiva

- Funcionó: medir la ruptura con una prueba propia (re-split del train) en lugar de discutirla en abstracto.
- Funcionó: generar el reporte desde `artifacts/`, así se actualiza solo cuando cambian los resultados.
- No funcionó: cambiar el optimizador no cambia el problema de fondo: el objetivo sigue siendo el Sharpe de una sola muestra de train.
- No funcionó: los artifacts se regeneraron en computadoras distintas y no se pueden reproducir exactos en otra máquina.
