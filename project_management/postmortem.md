# Postmortem

## Resumen

El sistema está completo y es causal: datos, selección del grupo, dos filtros de Kalman, política de umbrales, backtest con costos, métricas, dashboard y reporte. El problema está en la premisa estadística: la relación entre AMT y SBAC no es estable. Se rompe en test (ADF p = 0.88), también se rompe dentro de train al re-partirlo, y el θ elegido en train no se sostiene fuera de muestra. La revisión del avance lo diagnosticó como sobreajuste, y nuestra prueba de re-split lo confirma.

## Hitos: qué se cumplió a tiempo y qué se retrasó

| hito | plan | real | comentario |
|---|---|---|---|
| Estructura del repositorio | 2026-10-29 | 2026-10-29 | a tiempo |
| Pipeline de datos completo | 2026-10-30 | 2026-10-30 | a tiempo |
| Prueba de cointegración completa | 2026-10-30 | 2026-11-01 | retrasado: el embudo del universo v1 tenía un error (la correlación se calculaba sobre niveles y con el mínimo del grupo) que eliminaba los 57 grupos antes de llegar a Johansen; se corrigió y se cambió de universo |
| Filtro 1 validado | 2026-11-01 | 2026-11-01 | a tiempo, pero con varias iteraciones de calibración el mismo día |
| Backtest completo | 2026-11-04 | 2026-11-03 | a tiempo |
| Dashboard funcionando | 2026-11-04 | 2026-11-04 | a tiempo |
| Charter, sprints y postmortem | durante el proyecto | 2026-11-08 | retrasado: se redactaron al final, de forma retrospectiva, a partir del historial de commits |
| Reporte PDF | 2026-11-08 | 2026-11-08 | a tiempo con la entrega extendida |

## Cuellos de botella

**Técnicos**

- **Calibración de los filtros.** Fueron necesarias cuatro iteraciones: centrar el regresor (sin centrar, α y β tenían correlación de −0.98), acotar δ (la máxima verosimilitud llevaba a un β que perseguía los precios), calibrar R por momentos (colas pesadas) y fijar la memoria del filtro 2 en L = 83 días (la primera versión perseguía al spread y las operaciones duraban 5 días).
- **Reproducibilidad.** La optimización bayesiana dio otro θ* en otra computadora por diferencias de redondeo del orden de 1e-14. Se resolvió congelando θ* por defecto en `main.py`.
- **Errores de integración.** El ciclo principal de `main.py` se borró por accidente y `strategy/VECM.py` no importaba en Windows; los dos se corrigieron.
- **Johansen.** El rango se calculaba contando todos los rechazos en lugar de probar en escalera; se corrigió al cierre (un grupo con bandera roja dejó de pasar).

**Organizacionales**

- **Dependencia en cadena.** El integrante C dependía de las señales del B, y el B del grupo del A. El retraso de la selección se propagó.
- **Flujo de git.** Casi todo se subió directo a `main`, con un solo pull request, y los artifacts se regeneraron en computadoras distintas.
- **Gestión al final.** El charter y los sprints no guiaron el trabajo porque se escribieron al cierre.
- **Comunicación en la revisión.** En la revisión del avance se dijo que Johansen solo se había corrido sobre los seis activos juntos, cuando se probaron las 57 combinaciones. Faltó que cada integrante conociera el trabajo de los demás.

## Qué haríamos distinto en producción

1. Fijar el diseño de validación antes de ver resultados y contar todos los grupos y configuraciones probadas.
2. Validación walk-forward con recalibración periódica del vector y de los filtros, en lugar de una sola partición.
3. Operar solo cuando la cointegración está activa, con un filtro de régimen (por ejemplo, el ADF móvil del spread) que apague la estrategia cuando la relación se rompe.
4. Un universo más amplio y relaciones con sustento fundamental más directo, para no depender de un solo par.
5. Calibrar los parámetros de los filtros con criterios estadísticos y dejar el P&L solo para la política; monitorear las innovaciones en vivo.
6. Ejecutar con precios de apertura o intradía y modelar costos de préstamo variables y el impacto de mercado.
7. Una sola corrida oficial en integración continua, con los artifacts versionados, y todo el trabajo por pull requests revisados.

## Cómo se materializaron los riesgos del charter

| riesgo | ¿se materializó? | qué pasó |
|---|---|---|
| Relación AMT y SBAC inestable | sí | el ADF del spread pasa de 0.012 en train a 0.88 en test, y el re-split del train muestra la misma ruptura |
| Sobreajuste de θ y de los filtros | sí | θ* cambia entre computadoras y los mejores trials son casi empates; el test es negativo para Kalman |
| Pruebas múltiples | sí | pasan Johansen 2 de 57 grupos, cuando el azar daría 2.85 |
| Divergencia o mala calibración de los filtros | parcialmente | las primeras versiones fallaban las pruebas de calibración y se corrigieron; las innovaciones siguen autocorrelacionadas |
| Reajustes de Yahoo | no | el snapshot de precios en `artifacts/` lo evitó |
| Resultados distintos entre computadoras | sí | se mitigó congelando θ* |
| Pocas operaciones fuera de muestra | sí | Kalman hace 3 operaciones en test y 3 en val; además, un solo día (2026-04-02) explica el 74% de su P&L en val |

## Lecciones

1. Un óptimo que cambia con diferencias de redondeo no es un óptimo: el objetivo debe premiar configuraciones que funcionen en varios periodos, no la mejor en uno.
2. Una prueba que solo revisa media y varianza no prueba que un filtro esté bien especificado.
3. Un resultado negativo bien documentado vale más que uno positivo que no se puede defender.
