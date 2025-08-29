---
marp: true
theme: sparta
paginate: true
math: katex
---

<!-- _class: lead -->
# Viernes de código
## El valor del dinero en el tiempo ⌛💸
**Juan Camilo Pinedo Campo** · Universidad del Norte  
_29 agosto 2025_

---

<!-- _class: white-slide -->

# ¿Alguna vez han escuchado que el tiempo es dinero?

<div class="multicolumn vcenter"><div>

-  La moneda **es el tiempo**: cada minuto es vida. 
- Hay **precios**, **intereses**, **peajes** y **“policía del tiempo”**.
- podemos mapear esos elementos a **conceptos financieros reales.**  



</div><div>

![center w:350](https://m.media-amazon.com/images/I/516z8CTk8sL._SY300_SX300_QL70_ML2_.jpg)

<figcaption>El precio del mañana - Andrew nicoll</figcaption>

</div></div>

---


<!-- _class: white-slide -->

# Empecemos viendo la definicion matematica de algunos conceptos

<div class="multicolumn vcenter"><div>

- **VF**: capital que tendrá un monto tras capitalizarse.  
  $$\mathrm{VF} = \mathrm{VP}(1+i)^n$$
- **VP**: valor hoy de un monto futuro.  
  $$\mathrm{VP} = \frac{\mathrm{VF}}{(1+i)^n}$$

----

# ¿En el contexto financiero y de la pelicula como podriamos interpretar estos conceptos?

<div class="multicolumn vcenter"><div>



- Si guardas **120 min** a \(i=5\%\) semanal durante \(n=6\) semanas:   
$$\mathrm{VF}=120(1.05)^6$$

- Si necesitas **200 min** en 6 semanas con \(i=5\%\):  
  $$\mathrm{VP}=\frac{200}{(1.05)^6}$$

> Interpretación: ¿conviene guardar minutos hoy o buscarlos prestados?

----

# Tasa efectiva anual (TEA)

Si tienes una tasa nominal mensual \(i_m\), la **efectiva anual** es:
$$
\mathrm{TEA}=\left(1+i_m\right)^{12}-1
$$

**Ejemplo:** \(i_m=2\%\) ⇒ 
$$
\mathrm{TEA}=(1.02)^{12}-1\approx \mathbf{26.82\%}
$$ 

**Equivalente con tasa nominal anual \(j\) convertible mensual:**
$$
\mathrm{TEA}=\left(1+\frac{j}{12}\right)^{12}-1
$$

> En *In Time*: si los **precios en minutos** suben “un 2% mensual”, el costo anual se dispara (capitalización).

---

# Tasa real (Fisher)

Relación de Fisher:
$$
1+i = (1+r)(1+\pi) \quad\Rightarrow\quad r=\frac{1+i}{1+\pi}-1
$$
- \($i$): tasa nominal  
- \($\pi$): inflación  
- \($r$): **tasa real** (poder adquisitivo)

**Ejemplo:** \($i=20\%$), \($\pi=12\%$) ⇒ \($r\approx 7.14\%$).  
> En *In Time*: si “suben los precios de minutos, tu tasa **real** de ganancia de tiempo cae.” 

$$
\pi↑
$$

---

# VAN (Valor Actual Neto)

Regla de decisión:
$$
\mathrm{VAN}=\sum_{t=1}^{n}\frac{CF_t}{(1+i)^t}-C_0
$$
- Si **VAN > 0** ⇒ crea valor (aceptar).
- Si **VAN < 0** ⇒ destruye valor (rechazar).

**Datos:** \(C_0=60\) min, \(CF_1=CF_2=CF_3=25\) min, \(i=10\%\)

$$
\begin{aligned}
\mathrm{VAN}
&= -60 + \frac{25}{1.1} + \frac{25}{1.1^2} + \frac{25}{1.1^3} \\
&= -60 + 22.7273 + 20.6612 + 18.7829 \\
&= \mathbf{2.1713}\ \text{min} \approx \mathbf{2.17}\ 
\text{min}
\end{aligned}
$$

---

# TIR (Tasa Interna de Retorno)

La **TIR** es la tasa \($r$) que hace \($\mathrm{VAN}=0$):

$$
0=-C_0+\sum_{t=1}^{n}\frac{CF_t}{(1+r)^t}.
$$

**Ejemplo:** flujos (-60, 25 , 25 , 25):

$$
0=-60+\frac{25}{(1+r)}+\frac{25}{(1+r)^2}+\frac{25}{(1+r)^3}
\;\Rightarrow\; r\approx 0.12044\;\;(12.044\%).
$$

**Regla:** aceptar si \($\mathrm{TIR}>$\) tasa mínima requerida.

---

# Anualidades: fórmulas clave

**Anualidad ordinaria (pagos al fin de cada periodo)**

- Valor Presente:
  $$
  \mathrm{VP}=\mathrm{PMT}\cdot\frac{1-(1+i)^{-n}}{i}
  $$
- Valor Futuro:
  $$
  \mathrm{VF}=\mathrm{PMT}\cdot\frac{(1+i)^{n}-1}{i}
  $$
- Pago dado VF:
  $$
  \mathrm{PMT}=\mathrm{VF}\cdot\frac{i}{(1+i)^{n}-1}
  $$

---

# Ejemplo de anualidad (ahorro de tiempo)

Objetivo: **acumular 600 min** en \(n=12\) meses, \(i=1\%\) mensual.  
$$
\mathrm{PMT}=600\cdot\frac{0.01}{(1.01)^{12}-1}\approx \mathbf{47.3\;min/mes}
$$
> En *In Time*: “pagarte a ti mismo” minutos cada mes para “comprar” longevidad futura.

---

Entender el concepto es crucial para plasmarlo en python!🐍🐍🐍