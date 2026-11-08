# Sprint 3 (plan): cierre y entrega final

| | |
|---|---|
| Periodo | Del 2026-11-09 a la entrega extendida |
| Objetivo | Integrar los cambios pendientes, decidir qué se hace con los problemas abiertos y entregar una versión final consistente |
| Estado | Plan, por confirmar en la reunión del equipo |

## Backlog

| tarea | responsable | prioridad | criterio de terminado |
|---|---|---|---|
| Cambios de Andrea | Andrea | alta | en `main` y `python -m pytest` pasa |
| Decidir cómo atender los problemas abiertos (abajo) | equipo | alta | decisión anotada en este archivo |
| Correr `python main.py` una sola vez, en una sola computadora, y subir esos artifacts como la corrida oficial | por definir | alta | artifacts en `main` y la computadora anotada en el README |
| Regenerar figuras y reporte con `python -m performance.reporte` y revisar el texto contra los nuevos resultados | Abdon | alta | `docs/reporte.pdf` sin cifras viejas |
| Poner al día el paso 6 de `docs/powell.md` (hoy describe la rejilla de 81 combinaciones y el θ anterior) | Andrea | media | coincide con `theta_optimo.json` |
| Revisar el README: URL del repositorio y nivel de alcance declarado | Isabela | media | README coincide con lo entregado |
| Cerrar el postmortem | equipo | media | secciones completas |
| Revisión final: pipeline limpio, pruebas, dashboard y reporte | equipo | alta | lista de verificación de abajo completa |

## Problemas abiertos (del reporte)

1. θ* no es estable: cambia entre computadoras por diferencias de redondeo y los mejores trials son prácticamente empates.
2. Prueba 4: las innovaciones de los dos filtros están autocorrelacionadas (ρ₁ de 0.90 y 0.95 en train).
3. Prueba 2: no hay prueba automática de normalización.
4. δ se eligió por Sharpe; la verosimilitud prefiere δ = 10⁻⁵ por más de 1,000 puntos de log-verosimilitud.
5. La relación se rompe fuera de muestra y dentro de train; el resultado de val depende del 2026-04-02.

## Lista de verificación de la entrega

- [ ] `pip install -r requirements.txt` en un entorno limpio
- [ ] `python main.py` corre sin errores
- [ ] `python -m pytest` pasa
- [ ] `streamlit run dashboard/app.py` abre y muestra los artifacts finales
- [ ] `python -m performance.reporte` regenera `docs/reporte.pdf` y `docs/figuras/`
- [ ] Las cifras del README, del reporte y del dashboard coinciden
- [ ] Charter, sprints y postmortem completos
