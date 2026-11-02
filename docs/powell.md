# Pairs trading AMT y SBAC con Kalman: formulación de Powell

Este documento describe la estrategia del integrante B (filtros de Kalman y política de trading) con el marco de decisiones secuenciales de Powell. Todos los valores numéricos vienen de `artifacts/kalman_params.json`, `artifacts/theta_optimo.json`, `artifacts/kalman_calibracion_momentos.csv` y `artifacts/metricas_preliminares.csv`.

Notación general. $y_t = \log(\text{AMT}_t)$ es la observación y $x_t = \log(\text{SBAC}_t)$ el regresor. Usamos el regresor centrado

$$x^c_t = x_t - \bar{x}_{\text{burn-in}}, \qquad \bar{x}_{\text{burn-in}} = 4.5230,$$

donde $\bar{x}_{\text{burn-in}}$ es la media de $\log(\text{SBAC})$ en los primeros 126 días del train. Esos 126 días no se operan.

Recordatorio de los cinco elementos de Powell: estado $S_t$, decisión $x_t$ (aquí la llamaremos $d_t$ para no confundirla con el precio), información exógena $W_{t+1}$, función de transición $S_{t+1} = S^M(S_t, d_t, W_{t+1})$ y función objetivo. Los seis pasos del modelado son: (1) narrativa, (2) identificar las métricas del objetivo, (3) identificar las decisiones, (4) identificar las fuentes de incertidumbre, (5) escribir el modelo matemático y (6) diseñar y ajustar la política.

---

## a) Filtro 1: pesos dinámicos $(\alpha_t, \beta_t)$

### Modelo

Suponemos que la relación entre los dos log-precios es lineal con coeficientes que cambian lentamente:

$$y_t = \alpha_t + \beta_t\, x^c_t + v_t, \qquad v_t \sim N(0, R)$$

$$z_t = \begin{pmatrix}\alpha_t \\ \beta_t\end{pmatrix} = z_{t-1} + w_t, \qquad w_t \sim N(0, Q), \qquad Q = \frac{\delta}{1-\delta} I_2.$$

Es decir, $A = I$ (paseo aleatorio) y $C_t = (1,\ x^c_t)$.

### Los cinco elementos

**Estado.** El estado del filtro es su creencia sobre los pesos: la media $\hat z_{t|t} = (\hat\alpha_t, \hat\beta_t)$ y la matriz de covarianza $P_{t|t}$ de $2 \times 2$. No conocemos los pesos verdaderos, solo una distribución sobre ellos, así que el estado es un estado de creencia.

**Decisión.** No hay decisión. El filtro es un estimador: dado lo que observa, calcula la mejor estimación posible y no elige nada. Aun así conviene escribirlo en el lenguaje de Powell por dos razones. Primero, su salida es justamente la parte de creencia del estado del sistema completo, y la política de trading decide con ella. Segundo, los parámetros $\delta$ y $R$ sí se escogen, y esa elección es un problema de optimización con la misma estructura que el paso 6 de Powell: proponer una familia, medir un objetivo y elegir. El filtro es entonces un caso particular del marco en el que el conjunto de decisiones es vacío.

**Información exógena.** $W_{t+1} = (\text{AMT}_{t+1},\ \text{SBAC}_{t+1})$, el par de precios de cierre del día siguiente. Con eso se forma $y_{t+1}$ y $x^c_{t+1}$.

**Transición.** Primero se predice y luego se actualiza:

$$\text{Predicción:}\qquad \hat z_{t|t-1} = \hat z_{t-1|t-1}, \qquad P_{t|t-1} = P_{t-1|t-1} + Q$$

$$\text{Innovación:}\qquad e_t = y_t - C_t\, \hat z_{t|t-1}, \qquad S_t = C_t P_{t|t-1} C_t^\top + R$$

$$\text{Ganancia:}\qquad K_t = \frac{P_{t|t-1} C_t^\top}{S_t}$$

$$\text{Actualización:}\qquad \hat z_{t|t} = \hat z_{t|t-1} + K_t\, e_t, \qquad P_{t|t} = (I - K_t C_t)\, P_{t|t-1}$$

**Objetivo.** Para $\delta$ y $R$ fijos, el filtro de Kalman da la estimación de mínimo error cuadrático medio de $z_t$ dado lo observado. Para escoger $\delta$ y $R$ el objetivo natural es la log-verosimilitud de las innovaciones en el train (después del burn-in):

$$\ell(\delta, R) = \sum_t -\tfrac{1}{2}\left(\log(2\pi S_t) + \frac{e_t^2}{S_t}\right).$$

Más abajo explicamos por qué, al final, $R$ no se tomó del máximo de $\ell$ sino por método de momentos.

### Los seis pasos aplicados

1. Narrativa: AMT y SBAC son dos empresas de torres de telecomunicaciones; su relación de precios es estable a largo plazo pero no fija, y queremos seguirla día a día sin mirar el futuro.
2. Métricas: error de predicción de $y_t$, medido con la log-verosimilitud de las innovaciones y con que $e_t/\sqrt{S_t}$ se parezca a una normal estándar.
3. Decisiones: ninguna en línea; fuera de línea se eligen $\delta$, $R$ y la inicialización.
4. Incertidumbre: el ruido de observación $v_t$, los cambios de los pesos $w_t$ y los precios futuros.
5. Modelo: las ecuaciones de arriba, con $\hat z_0 = (4.2814,\ 0.3952)$ del OLS del burn-in y $P_0 = I$.
6. Ajuste: rejilla de $\delta \in \{10^{-7}, \ldots, 10^{-3}\}$ y $R$, restricción $\delta \le 10^{-5}$, $R$ por momentos y $\delta^* = 10^{-6}$ elegido por Sharpe en el grid search.

Valores finales: $\delta = 10^{-6}$, $Q = 1.000001 \times 10^{-6}\, I_2$, $R = 4.847 \times 10^{-4}$ (frente a $R_{\text{OLS}} = 1.307 \times 10^{-3}$).

---

## b) Filtro 2: nivel y escala del spread

### Modelo

Con el $\beta$ previo (el de antes de ver el dato de hoy) se forma el spread

$$s_t = y_t - \hat\beta_{t|t-1}\, x^c_t.$$

El nivel $\alpha$ no se resta: lo absorbe este segundo filtro como un nivel local $\mu_t$:

$$s_t = \mu_t + v_t,\quad v_t \sim N(0, R2_t), \qquad \mu_t = \mu_{t-1} + w_t,\quad w_t \sim N(0, Q2_t).$$

### Los cinco elementos

**Estado.** $(\hat\mu_t,\ P2_t,\ R2_t)$: la media estimada del nivel, su varianza y la varianza estimada del ruido del spread.

**Decisión.** Tampoco hay decisión; es otro estimador. Su producto es el z-score que usa la política.

**Información exógena.** El mismo par de precios del día siguiente, que junto con $\hat\beta_{t+1|t}$ produce $s_{t+1}$.

**Transición.** En cada día:

$$Q2_t = q\, R2_{t-1}, \qquad \hat\mu_{t|t-1} = \hat\mu_{t-1|t-1}, \qquad P2_{t|t-1} = P2_{t-1|t-1} + Q2_t$$

$$S2_t = P2_{t|t-1} + R2_{t-1}, \qquad z_t = \frac{s_t - \hat\mu_{t|t-1}}{\sqrt{S2_t}}$$

El z-score se calcula antes de actualizar, así que solo usa información hasta $t-1$ para el nivel y la escala. Después:

$$K2_t = \frac{P2_{t|t-1}}{S2_t}, \qquad \hat\mu_{t|t} = \hat\mu_{t|t-1} + K2_t\,(s_t - \hat\mu_{t|t-1}), \qquad P2_{t|t} = (1 - K2_t)\, P2_{t|t-1}$$

$$R2_t = \lambda\, R2_{t-1} + (1-\lambda)\,(s_t - \hat\mu_{t|t-1})^2.$$

**Objetivo.** Que $z_t$ esté bien calibrado: media cercana a cero y varianza cercana a uno en train, con la persistencia suficiente para que valga la pena tradearlo.

### Parámetros

Con $L = \text{round}(2 \times 41.3) = 83$ días:

$$K_{\text{obj}} = \frac{1}{L} = 0.01205, \qquad q = \frac{K_{\text{obj}}^2}{1-K_{\text{obj}}} = 1.469 \times 10^{-4}, \qquad \lambda = 1 - \frac{1}{L} = 0.98795.$$

Inicialización con el spread del burn-in calculado con el $\beta$ del OLS fijo: $\hat\mu_0 = 4.2814$, $R2_0 = 1.307 \times 10^{-3}$, $P2_0 = K_{\text{obj}} R2_0 = 1.574 \times 10^{-5}$. La ganancia mediana observada fue 0.0134 en train, 0.0125 en test y 0.0131 en val, muy cerca del objetivo.

Los seis pasos son análogos a los del filtro 1. La narrativa es que el spread oscila alrededor de un nivel que se mueve despacio; la métrica es la calibración de $z_t$; no hay decisiones en línea; la incertidumbre es el ruido del spread y el cambio del nivel; el modelo son las ecuaciones de arriba; y el ajuste fue escoger $q$ y $\lambda$ para tener la misma memoria que la ventana del modelo estático.

---

## c) El sistema completo

### Estado

Separamos el estado en sus tres partes:

$$S_t = (R_t,\ I_t,\ B_t).$$

- **Físico** $R_t$: efectivo, número de acciones de AMT y de SBAC (con signo, positivo largo y negativo corto) y $h_t$, los días que lleva abierta la posición.
- **Información** $I_t$: precios de cierre de AMT y SBAC, el regresor centrado $x^c_t$ y el spread $s_t$.
- **Creencia** $B_t$: $(\hat\alpha_t, \hat\beta_t, P_t)$ del filtro 1 y $(\hat\mu_t, P2_t, R2_t)$ del filtro 2.

El estado inicial $S_0$ contiene además los parámetros que no cambian: $\delta = 10^{-6}$, $R = 4.847 \times 10^{-4}$, $\bar{x}_{\text{burn-in}} = 4.5230$, $L = 83$, $\theta^*$ y los costos (comisión de 0.125% por pata, préstamo de 0.25% anual sobre la pata corta, capital inicial de 1,000,000 y exposición de 80%).

### Decisión

$$d_t \in \{-1,\ 0,\ +1\}.$$

$d_t = +1$ significa largo el spread (comprar AMT y vender SBAC), $d_t = -1$ lo contrario y $d_t = 0$ estar fuera. Al abrir una posición con capital $V$ y el $\beta$ del día de la señal, el tamaño es

$$N_{\text{AMT}} = 0.8\, V \frac{1}{1+\beta}, \qquad N_{\text{SBAC}} = 0.8\, V \frac{\beta}{1+\beta},$$

y las acciones se quedan fijas mientras la posición está abierta. La señal se calcula con el cierre de $t$ y se ejecuta al cierre de $t+1$.

### Información exógena

$W_{t+1} = (\text{AMT}_{t+1},\ \text{SBAC}_{t+1})$, los precios del día siguiente.

### Transición

La función de transición tiene tres piezas: el filtro 1 actualiza $(\hat\alpha, \hat\beta, P)$, el filtro 2 actualiza $(\hat\mu, P2, R2)$ y produce $z_{t+1}$, y la contabilidad actualiza el estado físico. En la contabilidad el valor de la cuenta cambia por la ganancia de las acciones, menos el préstamo y las comisiones:

$$V_{t+1} = V_t + n^{A}_t\,\Delta P^{A}_{t+1} + n^{S}_t\,\Delta P^{S}_{t+1} - \text{préstamo}_{t+1} - \text{comisiones}_{t+1}.$$

### Objetivo

Buscamos la política con el mayor Sharpe neto de costos en train:

$$\max_{\theta}\ \text{Sharpe}(\theta) = \sqrt{252}\ \frac{\overline{r}(\theta)}{\sigma_r(\theta)},$$

donde $r$ son los rendimientos diarios de la cuenta. En la forma general de Powell esto es buscar la política que maximiza el valor esperado de la suma de recompensas,

$$\max_{\pi}\ \mathbb{E}\left[\sum_{t=0}^{T} C\big(S_t, X^{\pi}(S_t)\big) \,\middle|\, S_0\right],$$

con $C$ la ganancia neta del día. El Sharpe es una versión ajustada por riesgo del mismo objetivo: divide la recompensa media entre su volatilidad.

### Política: PFA

La política es una función de política (PFA) con parámetros $\theta = (z_{\text{in}}, z_{\text{out}}, z_{\text{stop}}, h_{\max})$:

- Si estamos fuera y $z_{\text{in}} < |z_t| \le z_{\text{stop}}$, entramos: $d_t = +1$ si $z_t < 0$ y $d_t = -1$ si $z_t > 0$.
- Si estamos dentro, cerramos por salida si $|z_t| < z_{\text{out}}$, por stop si $|z_t| > z_{\text{stop}}$ o por tiempo si $h_t \ge h_{\max}$.
- Después de un cierre por stop o por tiempo no se vuelve a entrar hasta que $|z_t| < z_{\text{in}}$.

Es la misma clase de política que la regla min-max de inventarios. En inventarios, cuando el nivel baja de un mínimo se pide hasta un máximo; aquí, cuando el spread se aleja más de $z_{\text{in}}$ se abre y cuando regresa a $z_{\text{out}}$ se cierra. En los dos casos la política es un par de umbrales sobre una variable del estado que disparan una acción, y los umbrales se ajustan simulando la política y midiendo el objetivo.

### Paso 6: ajustar la política

Se hizo un grid search en train (después del burn-in) con $z_{\text{in}} \in \{1.5, 2.0, 2.5\}$, $z_{\text{out}} \in \{0, 0.25, 0.5\}$, $z_{\text{stop}} \in \{3.0, 3.5, 4.0\}$ y $h_{\max} \in \{41, 83, 124\}$, que son 81 combinaciones. Para Kalman se probaron además $\delta \in \{10^{-6}, 10^{-5}\}$ (162 configuraciones, 61 con al menos 10 operaciones) y para el estático la ventana $L \in \{41, 83, 124\}$ (243 configuraciones, 236 factibles). Se exigió un mínimo de 10 operaciones para no premiar resultados de dos o tres apuestas afortunadas.

Resultado congelado:

| método | $z_{\text{in}}$ | $z_{\text{out}}$ | $z_{\text{stop}}$ | $h_{\max}$ | otros | Sharpe train | operaciones |
|---|---|---|---|---|---|---|---|
| Kalman | 2.0 | 0.5 | 4.0 | 41 | $\delta^* = 10^{-6}$ | 0.483 | 11 |
| Estático | 2.0 | 0.25 | 3.0 | 41 | $L^* = 83$ | 0.570 | 30 |

Desde aquí $\theta^*$ ya no se toca: test y val solo se usan para medir.

---

## d) La ganancia de Kalman en palabras

Toda la actualización se resume en una frase:

$$\text{nuevo estado} = \text{estado anterior} + \text{ganancia} \times \text{sorpresa}.$$

La sorpresa es la innovación $e_t$: lo que vimos menos lo que esperábamos ver. La ganancia $K_t$ dice cuánto de esa sorpresa nos creemos. Es un cociente entre lo que no sabemos del estado ($P_{t|t-1}$, que crece con $Q$) y lo que no sabemos en total ($S_t$, que incluye a $R$).

- Si $R$ es muy grande, el filtro piensa que los datos son puro ruido. $S_t$ crece, $K_t$ se hace pequeña y el estado casi no se mueve.
- Si $Q$ es muy grande, el filtro piensa que el estado cambia mucho cada día. $P_{t|t-1}$ crece, $K_t$ se acerca a su máximo y el estado persigue cada dato nuevo.

Esto explica nuestras tres decisiones principales:

- **R por momentos.** La máxima verosimilitud gaussiana elegía una $R$ demasiado pequeña. Con eso el filtro se creía más preciso de lo que era ($e_t/\sqrt{S_t}$ tenía varianza 1.58). Al subir $R$ hasta que esa varianza es 1, la ganancia refleja el tamaño real de los errores.
- **$\delta$ acotado.** Un $\delta$ grande es un $Q$ grande: $\beta$ perseguiría los precios y se comería la reversión del spread que queremos tradear.
- **$L = 83$.** En el filtro 2 fijamos la ganancia en $K = 1/83$. Así cada día el nivel se mueve solo una ochenta y tresava parte de la sorpresa y recuerda aproximadamente los últimos 83 días, igual que la ventana del modelo estático.

---

## e) Ejemplo numérico a mano (filtro 1, tres periodos)

Usamos números redondos: $\hat z_0 = (1,\ 0.5)$, $P_0 = I$, $Q = 0.01\, I$ y $R = 0.1$. Las observaciones son $(x^c_1, y_1) = (1,\ 1.6)$, $(x^c_2, y_2) = (2,\ 2.1)$ y $(x^c_3, y_3) = (-1,\ 0.4)$. Todos los números están redondeados a cuatro decimales y se verificaron con un script de numpy que aplica exactamente las ecuaciones de la sección a).

### Periodo 1: $x^c_1 = 1$, $y_1 = 1.6$

Predicción: el estado no cambia y la incertidumbre crece en $Q$.

$$\hat z_{1|0} = \begin{pmatrix}1 \\ 0.5\end{pmatrix}, \qquad P_{1|0} = \begin{pmatrix}1.01 & 0 \\ 0 & 1.01\end{pmatrix}$$

Con $C_1 = (1,\ 1)$ la predicción de $y$ es $1 + 0.5 = 1.5$, así que la sorpresa es $e_1 = 1.6 - 1.5 = 0.1$.

$$P_{1|0} C_1^\top = \begin{pmatrix}1.01 \\ 1.01\end{pmatrix}, \qquad S_1 = 1.01 + 1.01 + 0.1 = 2.12$$

$$K_1 = \frac{1}{2.12}\begin{pmatrix}1.01 \\ 1.01\end{pmatrix} = \begin{pmatrix}0.4764 \\ 0.4764\end{pmatrix}$$

$$\hat z_{1|1} = \begin{pmatrix}1 \\ 0.5\end{pmatrix} + 0.1 \begin{pmatrix}0.4764 \\ 0.4764\end{pmatrix} = \begin{pmatrix}1.0476 \\ 0.5476\end{pmatrix}, \qquad P_{1|1} = \begin{pmatrix}0.5288 & -0.4812 \\ -0.4812 & 0.5288\end{pmatrix}$$

La covarianza fuera de la diagonal se vuelve negativa: con un solo dato en $x = 1$ el filtro no puede separar $\alpha$ de $\beta$, solo su suma.

### Periodo 2: $x^c_2 = 2$, $y_2 = 2.1$

$$\hat z_{2|1} = \begin{pmatrix}1.0476 \\ 0.5476\end{pmatrix}, \qquad P_{2|1} = \begin{pmatrix}0.5388 & -0.4812 \\ -0.4812 & 0.5388\end{pmatrix}$$

Con $C_2 = (1,\ 2)$ la predicción es $1.0476 + 2(0.5476) = 2.1429$ y la sorpresa $e_2 = 2.1 - 2.1429 = -0.0429$.

$$P_{2|1} C_2^\top = \begin{pmatrix}-0.4235 \\ 0.5965\end{pmatrix}, \qquad S_2 = 0.7694 + 0.1 = 0.8694$$

$$K_2 = \begin{pmatrix}-0.4872 \\ 0.6861\end{pmatrix}, \qquad K_2 e_2 = \begin{pmatrix}0.0209 \\ -0.0294\end{pmatrix}$$

$$\hat z_{2|2} = \begin{pmatrix}1.0686 \\ 0.5182\end{pmatrix}, \qquad P_{2|2} = \begin{pmatrix}0.3325 & -0.1906 \\ -0.1906 & 0.1296\end{pmatrix}$$

Ahora hay datos en dos valores distintos de $x$ y la incertidumbre de $\beta$ baja de 0.5388 a 0.1296.

### Periodo 3: $x^c_3 = -1$, $y_3 = 0.4$

$$\hat z_{3|2} = \begin{pmatrix}1.0686 \\ 0.5182\end{pmatrix}, \qquad P_{3|2} = \begin{pmatrix}0.3425 & -0.1906 \\ -0.1906 & 0.1396\end{pmatrix}$$

Con $C_3 = (1,\ -1)$ la predicción es $1.0686 - 0.5182 = 0.5504$ y la sorpresa $e_3 = 0.4 - 0.5504 = -0.1504$.

$$P_{3|2} C_3^\top = \begin{pmatrix}0.5331 \\ -0.3302\end{pmatrix}, \qquad S_3 = 0.8633 + 0.1 = 0.9633$$

$$K_3 = \begin{pmatrix}0.5534 \\ -0.3428\end{pmatrix}, \qquad K_3 e_3 = \begin{pmatrix}-0.0832 \\ 0.0515\end{pmatrix}$$

$$\hat z_{3|3} = \begin{pmatrix}0.9853 \\ 0.5697\end{pmatrix}, \qquad P_{3|3} = \begin{pmatrix}0.0475 & -0.0079 \\ -0.0079 & 0.0264\end{pmatrix}$$

Con un dato del otro lado de la media ($x = -1$) la covarianza entre $\alpha$ y $\beta$ casi desaparece. Es la misma idea por la que centramos $x$ en el modelo real: si todos los $x$ estuvieran del mismo lado y lejos de cero, $\alpha$ y $\beta$ seguirían confundidos.

---

## f) Decisiones de diseño y por qué

**Centrar $x$.** Sin centrar, $x = \log(\text{SBAC}) \approx 5.2$ y $C_t = (1,\ 5.2)$, así que las dos columnas son casi colineales. El filtro no distinguía "subió el nivel" de "subió $\beta$" y los cambios de nivel se colaban en $\beta$; la correlación entre $\hat\alpha$ y $\hat\beta$ en train era $-0.976$. Siguiendo la pista del profe de revisar el escalamiento de los log-precios en $C_t$ cuando el filtro no se comporta bien, restamos la media del burn-in. Con eso el spread dejó de derivar. En cambios diarios, la correlación entre $\hat\alpha$ y $\hat\beta$ pasó de $-0.675$ a $0.089$ con el modelo final ($\delta = 10^{-6}$), así que el filtro ya no confunde día a día un cambio de nivel con un cambio de $\beta$. En niveles quedó en $0.70$, positiva: ahora refleja que los dos derivan juntos a lo largo de los años, no una confusión del filtro.

**R por método de momentos.** Las innovaciones tienen colas pesadas: curtosis en exceso de 1.22 con $\delta = 10^{-6}$ y de 4.45 con $\delta = 10^{-5}$. Con colas pesadas, la log-verosimilitud gaussiana prefiere una $R$ pequeña que ajuste bien los días normales y tolera los pocos errores grandes. El filtro termina reportando menos incertidumbre de la real. Por eso escogemos $R$ tal que $\operatorname{var}(e_t/\sqrt{S_t}) = 1$ en train, buscándola por bisección. La tabla de máxima verosimilitud se conserva como evidencia en `kalman_calibracion.csv`. Fuera de muestra, la varianza quedó en 1.13 en test y 1.62 en val.

**$\delta$ acotado a $10^{-5}$.** Con $\delta \ge 10^{-4}$ la máxima verosimilitud empujaba a una solución degenerada: $R$ al mínimo de la rejilla y $\beta$ moviéndose entre 0.39 y 0.93 en val. El hedge ratio entre dos empresas del mismo negocio es un parámetro estructural que debe cambiar en meses, no en días. Con $\delta$ grande la verosimilitud explica la reversión del spread como movimientos de $\alpha$ y $\beta$, que es justo lo que queremos tradear. Entre $10^{-6}$ y $10^{-5}$ decidió el grid search, y ganó $10^{-6}$.

**$Q2$ proporcional a $R2$ con $K = 1/L$.** En el modelo de nivel local, si $Q2 = q\, R2$ la ganancia converge a un valor fijo, y con $q = K^2/(1-K)$ ese valor es exactamente $K$. Con ganancia fija, $\hat\mu$ es un promedio exponencial con memoria de aproximadamente $1/K$ días. Al usar $K = 1/83$ el filtro 2 recuerda lo mismo que la ventana de 83 días del modelo estático, y la comparación entre los dos es justa. El $\lambda$ del EWMA de $R2$ se fijó en $1 - 1/L$ por la misma razón. La primera versión usaba un $Q2$ fijo y grande; con eso $\hat\mu$ perseguía al spread y las operaciones duraban unos 5 días, contra una vida media de 41.

**No volver a entrar después de un stop.** Si el spread salió por $|z| > z_{\text{stop}}$ o por tiempo, lo más probable es que la relación esté en un régimen distinto. Volver a entrar en cuanto $|z|$ baje un poco de $z_{\text{stop}}$ sería apostar otra vez a lo que acaba de fallar. Por eso exigimos que $|z|$ regrese por debajo de $z_{\text{in}}$ antes de permitir otra entrada. Por la misma lógica, tampoco se abre una posición si $|z|$ ya está más allá de $z_{\text{stop}}$.

**Criterio de la media en la prueba del filtro 2.** La prueba original pedía $|\text{media}(z)| < 0.1$, que supone observaciones independientes. Pero $z_t$ es muy persistente: su autocorrelación de orden 1 es $\rho_1 \approx 0.95$. Con 1533 días hay solo unos 41 episodios independientes:

$$n_{\text{ef}} = n\,\frac{1-\rho_1}{1+\rho_1} \approx 41, \qquad EE_{\text{ef}} = \sqrt{\frac{\operatorname{var}(z)}{n_{\text{ef}}}} \approx 0.15.$$

El criterio corregido es $|\text{media}| < 2\, EE_{\text{ef}} \approx 0.30$; la media observada fue 0.14. El criterio de varianza en $[0.7,\ 1.3]$ no se cambió.
