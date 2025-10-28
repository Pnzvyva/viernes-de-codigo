---
marp: true
theme: einstein
paginate: true
math: MathJax
---

<!-- _class: lead -->
# Viernes de código
## Riesgo y estrategias **Low Volatility** en acciones 📚♦📚
**Juan Camilo Pinedo Campo** · Universidad del Norte  
_26 septiembre 2025_

---

<!-- _class: white-slide -->

# Objetivo de la clase

<div class="multicolumn Vcenter"><div>



- Entender **qué es el riesgo** y cómo se **mide**.
- Ver cómo cambian las **distribuciones de retornos** en una estrategia **low-vol**.
- Comparar **low-vol vs. high-vol**: retorno esperado, volatilidad, colas y drawdowns.
- Conectar con **decisiones prácticas** de portafolio.


</div><div>


- **Retorno esperado**: $\mathbb{E}[R]$
- **Riesgo (variabilidad)**: $\sigma(R)$
- **Trade-off** clásico: más retorno potencial suele implicar más riesgo.  
- Pero… **la forma** de la distribución importa (colas, sesgo).
</div></div>


---

# Medidas de riesgo

- **Volatilidad**: $\sigma = \sqrt{\mathrm{Var}(R)}$
- **Semivolatilidad**: dispersión solo de pérdidas.
- **VaR (nivel $\alpha$)**: pérdida tal que $P(R \leq \mathrm{VaR}_\alpha) = \alpha$.
- **Expected Shortfall (ES)**: $\mathbb{E}[R \mid R \leq \mathrm{VaR}_\alpha]$.
- **Drawdown máx.**: peor caída desde máximo histórico.
- **Skew / Curtosis**: forma y “colas” de la distribución.

---

# Fuentes de riesgo en acciones

- **Sistemático (beta)**: mercado general.
- **Idiosincrático**: empresa/sector.
- **Liquidez, ejecución, modelo**.
- **Colas** por eventos: resultados, noticias, shocks macro.


> Diferencia clave: en **bonos** se cubre nivel con DV01; en **acciones** se cubre **beta al mercado/sector**.


---

# Estrategia Low Volatility

- Seleccionar activos con **baja volatilidad histórica** (o **beta** bajo).
- Portafolio **minimum variance** o **equal-weight low-vol**.
- Expectativa: **menor dispersión** y **drawdowns** más contenidos. 

  > Puede sacrificar parte del **upside**, pero mejora la **estabilidad**.



---

<!-- _class: white-slide -->

# Diseño del experimento

<div class="multicolumn Vcenter"><div>



1. Universo de acciones (Stooq).  
2. Calcular **retornos log** diarios.  
3. Volatilidad **rolling** (p. ej., 60 días).  
4. Formar cestas:
   - **LOW-VOL**: las $k$ menos volátiles.
   - **HIGH-VOL**: las $k$ más volátiles.
5. **Rebalanceo** (diario/semanal/mensual).  
6. Comparar: métricas y **distribuciones**.


</div><div>


Lo que observamos (distribuciones)

- **Histogramas**: LOW-VOL más **estrechos** (menor $\sigma$).
- **ECDF**: colas izquierdas menos pesadas en LOW-VOL.
- **Boxplots**: menor rango intercuartílico y menos outliers.

> La **forma de la distribución** cambia la experiencia del inversionista.  

</div></div>

---

<!-- _class: white-slide -->

# Métricas clave (ejemplo)

<div class="multicolumn Vcenter"><div>





- **Vol anual**: menor en LOW-VOL.  
- **Sharpe**: puede mejorar si el retorno no cae tanto.  
- **VaR/ES**: límites de pérdidas diarias menos severos.  
- **MaxDD**: caídas acumuladas más moderadas.



</div><div>


Riesgos y sesgos

- **Look-ahead / Survivorship**: ¡evitar!  
- **Overfitting** de la ventana/umbral.  
- **Turnover y costos**: más rebalanceo, más fricción.  
- **Régimen de mercado**: resultados pueden **no generalizar**.

</div></div>


---

# Conclusiones

- Low-vol ≠ “mejor siempre”, pero **mejora la estabilidad**.
- La **distribución** (colas, sesgo) importa tanto como $\sigma$.
- Decisiones reales: frecuencia de rebalanceo, límites de riesgo (VaR/ES), control de drawdowns y costos.


---

Próximos pasos
----

 Extender a **beta-low** vs **min variance**.  
Probar **rebalanceo mensual** y **límites de turnover**.  
Añadir **costos** y **slippage**.  
 Validación *out-of-sample* y *stress tests*.
---

  🐍 🐍 🐍 🐍
=
