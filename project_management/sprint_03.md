# Sprint 3: cierre y entrega final

| | |
|---|---|
| Periodo | 2026-11-08 (cierre de la entrega) |
| Objetivo | Revisar el repositorio completo contra el enunciado y cerrar lo que faltaba antes de la entrega. Se priorizó lo que el enunciado pide de forma explícita (estadísticos de Johansen, pruebas, reproducibilidad y gestión) sobre mejoras al modelo. |
| Resultado | Cumplido en los entregables. Los problemas de fondo del modelo quedan documentados, no resueltos. |

## Backlog y estado

| tarea | responsable | estado | evidencia |
|---|---|---|---|
| `main.py` usa el θ* congelado por defecto (`--reoptimizar` para volver a buscarlo), para que cualquier computadora reproduzca las cifras del reporte | Andrea | hecho | 863cc63 |
| Prueba 1 con tres cortes (dos en test y uno en val) | Andrea | hecho | 863cc63 |
| Prueba 2 (normalización del vector de pesos) automatizada en `tests/test_normalizacion.py` | Andrea | hecho | 863cc63 |
| Quitar la nota interna de entrega de `docs/` | Andrea | hecho | 863cc63 |
| Reportar los estadísticos de traza y de máximo eigenvalor con sus valores críticos (tabla 4 y anexo A) | Andrea | hecho | este cierre |
| Rango de Johansen probado en escalera, como se vio en clase (antes se contaban los rechazos): AMT+SBAC+T deja de pasar y quedan 2 de 57 grupos; el grupo elegido no cambia | Andrea | hecho | este cierre |
| Registro de riesgos con probabilidad e impacto | Andrea | hecho | este cierre |
| Paso 6 de `docs/powell.md` actualizado a la optimización bayesiana | Andrea | hecho | este cierre |
| Reporte más corto: se quitó la bitácora de operaciones del anexo (está completa en el dashboard y en `bt_operaciones.csv`) | Andrea | hecho | este cierre |
| Postmortem cerrado | Andrea | hecho | este cierre |

## Decisiones sobre los problemas abiertos

1. **θ* inestable entre computadoras:** se resolvió la reproducibilidad congelando θ*, pero no la fragilidad del óptimo: los mejores trials siguen siendo casi empates. Se reporta en la sección 6 del reporte.
2. **Prueba 4 (ruido blanco):** no se resuelve. Las innovaciones están autocorrelacionadas porque el estado de los filtros no incluye la desviación persistente del spread. Se documenta en la sección 5.3 del reporte.
3. **Prueba 2:** resuelta con una prueba automática.
4. **δ elegido por Sharpe:** se mantiene y se documenta como limitación.
5. **Relación inestable:** se reporta como sobreajuste de la relación en train, siguiendo la prueba que sugirió la revisión del avance (re-split del train).

## Lista de verificación de la entrega

- [x] `pip install -r requirements.txt` en un entorno limpio
- [x] `python main.py` corre sin errores y reproduce los artifacts del repositorio
- [x] `python -m pytest` pasa
- [x] `streamlit run dashboard/app.py` abre las dos vistas
- [x] `python -m performance.reporte` regenera `docs/reporte.pdf` y `docs/figuras/`
- [x] Charter, sprints y postmortem completos
