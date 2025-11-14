---
marp: true
theme: sparta
paginate: true
title: Viernes de código
description: Redes Neuronales y LSTM aplicadas a la predicción de volatilidad
size: 16:9
math: mathjax
---

 
# Viernes de código
## 🧠 Inteligencia Artificial en Finanzas
### _Fundamentos de redes neuronales_
**Juan Camilo Pinedo Campo** · Universidad del Norte  
_24 Octubre 2025_


---

# Estructura de la clase

1. Inteligencia artificial en las finanzas modernas  
2. Fundamentos de redes neuronales  
3. Aplicaciones financieras: predicción de volatilidad  
4. Construcción del modelo en Python  
5. Evaluación e interpretación de resultados  
6. Extensiones y cierre conceptual

---

# 1. Inteligencia Artificial en Finanzas

- La IA ha transformado el análisis financiero:
  - *Modelos de predicción de riesgo y volatilidad*  
  - *Análisis algorítmico de precios y derivados*  
  - *Optimización de portafolios con machine learning*  
  - *Gestión automatizada (robo-advisors)*  

**Idea central:**  
La IA reemplaza reglas rígidas por sistemas que **aprenden patrones directamente de los datos**.

---

## De los modelos econométricos a la IA

| Enfoque | Supuestos | Limitaciones |
|----------|------------|---------------|
| **Econométrico (ej. GARCH)** | Linealidad, varianza condicional fija | Rigidez estructural |
| **Machine Learning** | Pocos supuestos, alta capacidad predictiva | Menor interpretabilidad |

**Visión moderna:** combinar ambos mundos → IA para descubrir dinámicas ocultas en los mercados.

---

# 2. Fundamentos de Redes Neuronales

Una red neuronal es un **sistema de funciones compuestas** que aprende una relación entre variables de entrada y salida.

\[
y = f(Wx + b)
\]

Donde:
- **W**: pesos o “importancias” aprendidas  
- **b**: sesgo  
- **f(·)**: función de activación no lineal  

---

# Intuición

Una neurona artificial es un **filtro no lineal de información**:

x1 ─┬─▶[ w1 ]──┐
x2 ─┼─▶[ w2 ]──┼──▶ Σ + b → f(·) → y
x3 ─┴─▶[ w3 ]──┘


Cada capa aprende un nivel distinto de abstracción:
- Capa 1: relaciones directas (retornos → volatilidad)  
- Capa 2: interacciones más complejas (momentum + shocks)  

---

# Funciones de activación

| Función | Expresión | Propósito |
|----------|------------|------------|
| ReLU | f(x) = max(0, x) | Evita saturación y acelera el entrenamiento |
| Sigmoid | f(x) = 1 / (1 + e^{-x}) | Mapear entre 0 y 1 |
| Tanh | f(x) = tanh(x) | Centra los valores entre -1 y 1 |


---

# 3. Aplicación financiera: predicción de volatilidad

La **volatilidad** mide la incertidumbre del mercado.  
En finanzas, es la base de:
- Valoración de opciones (modelo de Black–Scholes)  
- Evaluación de riesgo (VaR, Expected Shortfall)  
- Estrategias de cobertura (hedging)

---

# Motivación

Los modelos tradicionales (GARCH) describen la varianza condicional,  
pero **no captan bien las no linealidades ni los cambios de régimen**.

Las redes neuronales:
- Aprenden directamente de los datos.  
- Detectan relaciones no lineales.  
- Capturan memoria temporal mediante arquitecturas como **LSTM**.

---

# 4. Implementación práctica

**Objetivo:**  
Predecir la volatilidad anualizada del Bitcoin (BTC-USD)  
a partir de sus rendimientos y volatilidad pasada.

---


🐍 🐍 🐍 🐍
=

## Ahora vayamos a python.






