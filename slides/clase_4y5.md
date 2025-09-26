---
marp: true
theme: einstein
paginate: true
math: MathJax
---

<!-- _class: lead -->
# Viernes de código
## Backtesting en bonos, acciones y pair trading. 📚♦📚
**Juan Camilo Pinedo Campo** · Universidad del Norte  
_19 septiembre 2025_

---

<!-- _class: white-slide -->

# ¿Qué es el *backtesting*?

<div class="multicolumn Vcenter"><div>



- **Definición:** Simular una estrategia con **datos históricos** para estimar cómo habría funcionado.
- **Idea central:** mismas **reglas** que usarías en “vivo”, aplicadas sobre el pasado.
- **Ingredientes:**
  - Datos limpios (precios, tasas, volúmenes, costos).
  - Reglas claras (entradas, salidas, sizing, stops).
  - Métricas (PnL, Sharpe, *drawdown*, *hit ratio*, *turnover*).



</div><div>


 **Valida hipótesis**

 **Mide trade-offs**

 **Compara variantes**: ventanas, umbrales, filtros.
 **Detecta sesgos**:
  - *Look-ahead bias* (usar info del futuro).
  - *Survivorship bias* (ignorar activos que “murieron”).
  - *Overfitting* (demasiados parámetros).

 **Buenas prácticas:**
  - Añadir **costos y slippage** realistas.

</div></div>


---

# Acciones: características clave

- **Propiedad** de una empresa (derechos económicos y de voto).
- **Rentabilidad total** = precio + dividendos.
- **Riesgos**: de negocio, de mercado, de liquidez, dilución.
- **Drivers**: beneficios, múltiplos (P/E, EV/EBITDA), tipos de interés, news/flows.
- **Horizontes típicos**: *swing* (días-semanas), *position* (meses), *intraday* (minutos) si hay datos HFT..

---

# Acciones vs. otros activos (visión rápida)

| Activo         | Qué compras                  | Sensibilidad principal            | Costos típicos | Liquidez |
|---|---|---|---:|---|
| **Acciones**   | Propiedad (equity)           | Resultados y múltiplos            | Bajas-medias   | Alta (grandes) |
| **Bonos**      | Deuda (flujo cupón)          | **Nivel de tasas** y *credit*     | Bajas          | Muy alta (soberanos) |
| **FX**         | Moneda contra otra           | Macro / diferenciales de tasas    | Muy bajas      | Muy alta |
| **Commodities**| Bien físico / futuros        | Oferta/demanda, *storage*, estacionalidad | Medias | Media-alta |
| **Cripto**     | Token nativo                 | Flujos especulativos, adopción    | Variables      | Alta (top), baja (colas) |

> Diferencia clave: en **bonos** se cubre nivel con DV01; en **acciones** se cubre **beta al mercado/sector**.


---

# ¿Qué es el *pair trading*?


 **Estrategia de valor relativo**: largo un activo “barato” y corto otro “caro” dentro de un **par relacionado**.


**Relación estadística**:
Correlación alta **no basta**; se busca **cointegración** (spread estacionario).



---

<!-- _class: white-slide -->

# Estrategia de Cointegración (Backtesting)

<div class="multicolumn Vcenter"><div>



**Paso 1: Pipeline conceptual**

1. Elegir par (mismo sector/negocio).  
2. Regr/cointegración → construir **spread**:  

$$
S_t = Y_t - (c + \beta X_t)
$$

**Paso 2: Reglas con Z-score**

- $$ z > +z_{\text{in}} $$  
  → *short spread* (vender caro / comprar barato).  

- $$ z < -z_{\text{in}} $$  
  → *long spread* (comprar barato / vender caro).  



</div><div>


- **Salida:**  
  $$ |z| < z_{\text{out}} $$  
  → cerrar la posición.  

  **Paso 3: Market-neutral**

- Acciones → cubrir **beta**.  
- Bonos → cubrir **DV01**.  
- Objetivo: neutralidad frente al riesgo sistémico.  

</div></div>

---

En backtesting es importante enfatizar que “pasado ≠ futuro”, pero sí descarta estrategias frágiles.
----

En acciones se remarca la diferencia de neutralidad que parcen iguales pero no lo son. con bonos (beta vs. DV01).
---

  🐍 🐍 🐍 🐍
=
