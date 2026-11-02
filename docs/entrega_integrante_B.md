# Entrega del integrante B al integrante C

Esta nota resume lo que necesitas para el backtest oficial: qué archivos leer, qué significa cada columna, las convenciones de tiempo y las métricas de referencia.

## Qué leer

| archivo | para qué |
|---|---|
| `artifacts/senales_kalman.csv` | señales de la estrategia con Kalman, una fila por día |
| `artifacts/senales_estatico.csv` | señales de la línea base con vector fijo de Johansen |
| `artifacts/theta_optimo.json` | $\theta^*$ congelado de cada método, $\delta^*$ de Kalman y $L^*$ del estático |
| `artifacts/kalman_params.json` | todos los parámetros de los filtros y los $\theta$ usados para generar las señales |
| `artifacts/metricas_preliminares.csv` | métricas de referencia por método y muestra, para validar tu backtest |
| `artifacts/grid_kalman.csv`, `artifacts/grid_estatico.csv` | grid search completo en train, por si quieres revisar la sensibilidad |

Para regenerar todo desde la raíz del repo:

```
.venv/bin/python -m strategy.grid_search
.venv/bin/python -m strategy.generar_senales
```

El primer comando busca $\theta^*$ y escribe `theta_optimo.json` y `metricas_preliminares.csv`; el segundo lee ese archivo y escribe los dos CSV de señales.

## Columnas de `senales_kalman.csv` y `senales_estatico.csv`

Los dos archivos tienen las mismas columnas.

| columna | significado |
|---|---|
| `fecha` | día de negociación |
| `muestra` | `train`, `test` o `val` |
| `en_burn_in` | `True` en los primeros 126 días del train; ahí no se opera y `d = 0` |
| `AMT`, `SBAC` | precios de cierre ajustados, en nivel (no en logaritmo) |
| `beta` | cobertura del día. Kalman: $\hat\beta_{t\mid t-1}$, el beta previo, que no usa el precio de hoy. Estático: 0.8631 fijo |
| `s` | spread del día. Kalman: $\log \text{AMT} - \beta\,(\log \text{SBAC} - 4.5230)$, con $x$ centrado. Estático: $\log \text{AMT} - 0.8631 \log \text{SBAC}$ |
| `mu` | nivel esperado del spread con información hasta $t-1$. Kalman: el filtro 2. Estático: media móvil de 83 días con `shift(1)` |
| `sigma` | escala del spread con información hasta $t-1$, calculada igual que `mu` |
| `z` | z-score causal, $(s - \mu)/\sigma$ |
| `d` | señal: `+1` largo el spread (largo AMT, corto SBAC), `-1` corto el spread, `0` fuera |
| `h` | días desde la entrada; el día del cierre trae la duración total de la operación |
| `motivo_cierre` | vacío, `salida`, `stop_z` o `stop_tiempo`; solo aparece el día en que se cierra |

Centrar $x$ solo cambia el nivel de `s`, y ese nivel lo absorbe `mu`. La cobertura sigue siendo `beta` acciones de SBAC por cada unidad de AMT, en términos de nocional.

## Convenciones

- **Tiempo de ejecución.** La señal `d` se calcula con el cierre del día $t$ y se ejecuta al cierre de $t+1$. No tenemos precios de apertura, así que no hay ejecución intradía.
- **Cada muestra arranca plana.** Como en el ejemplo del profe, train, test y val se evalúan por separado, cada una con 1,000,000 y sin posición abierta. Train empieza después del burn-in.
- **Ojo: los CSV traen posición continua entre muestras.** Las señales se generaron recorriendo todo el histórico de corrido, así que el primer día de test puede traer una posición heredada del final de train. Para reproducir las métricas por muestra hay que reiniciar el estado: vuelve a aplicar `strategy.reglas.aplicar_reglas` sobre la columna `z` de esa muestra con el $\theta^*$ de `theta_optimo.json`. Eso fue lo que hice en `metricas_preliminares.csv`. El z del primer día de test sí usa la historia previa, y eso es correcto: no es lookahead.
- **Tamaño y costos usados en el grid.** Al entrar se usa el 80% del capital, con $N_{\text{AMT}} = 0.8V/(1+\beta)$ y $N_{\text{SBAC}} = 0.8V\beta/(1+\beta)$, y el $\beta$ del día de la señal. Las acciones se quedan fijas mientras dura la operación. La comisión es de 0.125% del nocional de cada pata al entrar y al salir. El préstamo es de 0.25% anual sobre el nocional de la pata corta, por día natural con base 365. Al final de cada muestra se liquida lo que esté abierto.

## Sobre `strategy/evaluador.py`

Ese evaluador es una versión simple que escribí solo para comparar configuraciones en el grid search. No es el backtest oficial. Tu backtest, con la contabilidad completa, debería reproducir aproximadamente estos números:

| método | muestra | Sharpe | retorno | max drawdown | operaciones | win rate | comisiones | préstamo |
|---|---|---|---|---|---|---|---|---|
| Kalman | train | 0.483 | 8.94% | −6.26% | 11 | 72.7% | 22,901 | 940 |
| Kalman | test | −0.262 | −1.81% | −3.22% | 3 | 33.3% | 6,113 | 387 |
| Kalman | val | 0.886 | 11.32% | −3.65% | 3 | 66.7% | 6,320 | 423 |
| Estático | train | 0.570 | 16.18% | −4.95% | 30 | 56.7% | 64,643 | 2,372 |
| Estático | test | −0.628 | −6.06% | −7.60% | 13 | 46.2% | 26,137 | 1,195 |
| Estático | val | 0.766 | 8.78% | −3.04% | 9 | 66.7% | 18,366 | 638 |

Si tus resultados se alejan mucho de estos, lo primero que hay que revisar son el desfase de un día en la ejecución, que cada muestra arranque plana y que en Kalman se use el `beta` del día de la señal.

Kalman solo hace 3 operaciones en test y 3 en val, así que sus métricas fuera de muestra son muy ruidosas.

## Lo que te toca

La comparación entre rebalanceo diario de la cobertura (seguir `beta` todos los días mientras la posición está abierta) y rebalanceo por banda (ajustar solo cuando `beta` se aleja más de cierto umbral del beta de entrada) es tuya. El evaluador del grid no rebalancea: deja las acciones fijas desde la entrada hasta la salida.
