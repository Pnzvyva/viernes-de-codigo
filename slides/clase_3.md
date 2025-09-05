---
marp: true
theme: sparta
paginate: true
math: katex
---

<!-- _class: lead -->
# Viernes de código
## Bonos, estrategias y valoracion. 💹🪙
**Juan Camilo Pinedo Campo** · Universidad del Norte  
_5 septiembre 2025_

---

<!-- _class: white-slide -->

# ¿Que es un bono financiero?

<div class="multicolumn vcenter"><div>

- Los estados **siempre** han necesito financiarse
- Las personas con mucho dinero reciben un **beneficio** por esto.
- podemos mapear esos elementos a **conceptos financieros reales.**  



</div><div>

![left w:650](https://scripofilia.it/1729-large_default/prestito-a-premj-citta-di-venezia-1-obbligazione-1869.jpg)

</div></div>


---

# Bonos: conceptos clave

- **Valor nominal ($FV$)**: monto a recibir al vencimiento.  
- **Cupón \($C$\)**: pago periódico de intereses.  
- **Tasa de cupón**: \( C/FV \).  
- **Vencimiento \($n$\)**: número de periodos restantes.  
- **Rendimiento \(Yield, $y$\)**: retorno exigido por el mercado.  
- **Frecuencia**: anual, semestral, etc.  
- **Precio**: valor presente de **todos** los flujos.

---

# Tipos de bonos

- **Estándar (cupón fijo)**  
- **Cero cupón** (descuento puro)  
- **Perpetuo** (con cupones infinitos)  
- **Callable/Putable** (opción embebida)  
- **Indexados**  (p. ej., UVR/IPC)  
- **Convertibles** (a acciones)

---

# Valoración: bono con cupón fijo

**Precio (frecuencia anual para simplicidad):**
$$
P=\sum_{t=1}^{n}\frac{C}{(1+r)^t}+\frac{FV}{(1+r)^n}
$$

**Relaciones:**
$$
\begin{aligned}
r > \dfrac{C}{FV} &\;\;\Rightarrow\;\; \text{descuento } (P < FV) \\
r < \dfrac{C}{FV} &\;\;\Rightarrow\;\; \text{prima } (P > FV) \\
r = \dfrac{C}{FV} &\;\;\Rightarrow\;\; \text{par } (P = FV)
\end{aligned}
$$


---

# Valoración: cero cupón y perpetuo

**Cero cupón:**
$$
P=\frac{FV}{(1+r)^n}
$$

**Perpetuo (con cupón fijo \(C\)):**
$$
P=\frac{C}{r}
$$

> Intuición: el perpetuo es una **anualidad perpetua** con tasa de descuento \($r$\).

---

# Medidas de rendimiento (1/2)

**Current Yield (CY):**
$$
\mathrm{CY}=\frac{\text{Cupón anual}}{\text{Precio}}
$$
**Yield to Maturity (YTM):** tasa \($r^\ast$\) tal que
$$
P=\sum_{t=1}^{n}\frac{C}{(1+r^\ast)^t}+\frac{FV}{(1+r^\ast)^n}
$$

> La YTM es la **TIR** del bono si se mantienen los supuestos (hold to maturity, reinversión a YTM, etc.).

---

# Medidas de rendimiento (2/2)

**Yield to Call (YTC)** — si el bono es **calleable** a precio \($K$\) en \($n_c$\) periodos:
$$
P=\sum_{t=1}^{n_c}\frac{C}{(1+r_c)^t}+\frac{K}{(1+r_c)^{n_c}}
$$

> La **YTC** asume que el emisor ejerce el **call** en la primera fecha óptima (típicamente cuando \(r\) cae lo suficiente).

---

# Estrategia a seguir, Curva de rendimientos


- Relación entre madurez \($T$\) y rendimiento \($r$($T$)\).
- Formas típicas:
  - Normal (positiva)
  - Invertida
  - Plana

$$
r : T \mapsto y(T)
$$

**Interpretación:**
- Expectativas de tasas de interés
- Señales de inflación y ciclos económicos
- Punto de partida para estrategias cuantitativas

---
# Estrategias con Cointegración la idea clave


- Aunque cada precio \($p_{i,t}$\) es no estacionario $I(1)$, puede existir una combinación lineal que sí lo sea.

- Formalmente:
$$
\beta^\top p_t \sim I(0)
$$

- El spread:
$$
s_t = \beta^\top p_t
$$
es **estacionario**, es decir, revierte a la media.

**Intuición:** activos relacionados pueden desviarse en el corto plazo,  
pero mantienen un equilibrio de largo plazo.

---

# Estrategias con Cointegración

**Método 1: Siempre en largo**
- Tomar posición larga en el spread \($s_t$\).
- Basado en reversión a la media.
- Mayor riesgo, sin periodos flat.



---

# Estrategias con Cointegración


**Método 2: Basado en z-score**
- Definición:
$$
z_t = \frac{s_t - \mu_s}{\sigma_s}
$$

1. Tomar yields \($y_{1y}$, $y_{5y}$, $y_{10y}$\).  
2. Probar **cointegración** (Johansen) para hallar combinación estacionaria:
   $$
   s_t=\beta_{1}\,y_{1y,t}+\beta_{5}\,y_{5y,t}+\beta_{10}\,y_{10y,t}
   $$
3. Estandarizar con **z-score** y definir reglas:  
   - \(z>+2\): **vender** spread  
   - \(z<-2\): **comprar** spread  
   - \(|z|<0.5\): **cerrar**  
4. Ponderar posiciones con **DV01-neutralidad**.


---

# DV01-neutralidad (resumen)

- **DV01** (por 1 bp) de cada tramo \($1y$, $5y$, $10y$\).  
- Buscar \($w$\) cercano a \($\beta$\) con restricciones:
  $$
  \min\|w-\beta\|^2\quad
  \text{s.a.}\;\; \text{DV01}^\top w=0,\;\; w_{10y}=1
  $$
- PnL aproximado por duración:
  $$
  \Delta P\approx -\sum_i w_i\cdot \text{DV01}_i\cdot \Delta y_i(\text{bp})
  $$


---

# Caso Colombia: TES (COP vs UVR)

- **TES COP**: nominal en pesos.  
- **TES UVR**: indexados a inflación (UVR).  
- **Implicación**: separar curvas **nominal** y **real**; Fisher:
  $$
  1+i=(1+r)(1+\pi)\;\;\Rightarrow\;\; i\approx r+\pi
  $$
- Datos típicos disponibles: tasas **cero cupón** a plazos 1, 5, 10 años (COP y UVR).



---


# DV01 y Ajuste del Portafolio

**DV01 (Dollar Value of 1 bp):**
- Cambio en el precio por un movimiento de 1 punto básico en la tasa.
- Mide la sensibilidad a la curva de rendimientos.

**Optimización:**
$$
\min\|w-\beta\|^2
\quad \text{s.a.}\;\; \text{DV01}^\top w=0,\;\; w_{10y}=1
$$

- Mantener vector cercano a \($\beta$\).
- Neutralizar sensibilidad a movimientos paralelos de la curva.

**PnL aproximado:**
$$
\Delta P \approx -\sum_i w_i\cdot \text{DV01}_i\cdot \Delta y_i
$$

---

# PnL: interpretación y escalamiento

- Backtest reporta PnL **por 100** de nocional (base).  
- Escalar a dinero real:
  $$
  \text{PnL real}=\text{PnL base}\times\frac{\text{Nocional real}}{100}
  $$
- Ej.: PnL base \(=16.61\), nocional \(=300\times 10^9\) COP  
  $$
  16.61\times \frac{300\times 10^9}{100}=4.983\times 10^{10}\;\text{COP}
  $$

---

Vamos a python y veamos esto mucho mejor. 🐍 🐍 🐍 🐍 

