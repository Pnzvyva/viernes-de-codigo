---
marp: true
theme: sparta
paginate: true
class: lead
---




# Viernes de código
## 🧠 Inteligencia Artificial en Finanzas
### _De Random Forest a la predicción de volatilidad en derivados_
**Juan Camilo Pinedo Campo** · Universidad del Norte  
_26 septiembre 2025_

---

# 📘 Agenda

1️⃣ ¿Qué es Random Forest?  
2️⃣ Cómo aprende un modelo de IA supervisado  
3️⃣ Conexión con la predicción de volatilidad  
4️⃣ Integración con Black–Scholes  
5️⃣ Resultados en BTC–USD (Colab)  
6️⃣ Conclusiones y evolución hacia la IA moderna

---

# 🌳 ¿Qué es Random Forest?

> Un modelo de **Machine Learning supervisado** basado en **árboles de decisión**.

- Conjunto (*ensemble*) de muchos árboles entrenados con muestras distintas.
- Cada árbol aprende reglas del tipo:
  - Si `rv_21 > 0.35` → volatilidad alta  
  - Si `rv_21 <= 0.35` y `mom_21 < -0.05` → volatilidad media
- El resultado final es el **promedio** de todos los árboles.

\[
\hat{y} = \frac{1}{N}\sum_{i=1}^{N} f_i(X)
\]

**Ventaja:** aprende relaciones no lineales sin asumir fórmulas teóricas.

---

# ⚙️ Cómo aprende un Random Forest

1. **Bootstrap:** selecciona subconjuntos aleatorios del dataset.  
2. **Feature Sampling:** usa solo una fracción de las variables.  
3. **Entrena árboles de decisión independientes.**  
4. **Agrega resultados (promedio).**  

🧮 Criterio de aprendizaje: minimizar el **Error Cuadrático Medio (MSE)**  

```python
model = RandomForestRegressor(n_estimators=400, max_depth=8)
model.fit(X_train, y_train)
```

---

# 🧠 ¿Por qué es IA?

- Aprende de los **datos históricos** sin reglas explícitas.
- Captura **patrones ocultos** y no lineales.  
- Mejora la **predicción** de variables financieras.  
- Se adapta dinámicamente a cambios de mercado.  

👉 Es una forma de **inteligencia supervisada**: aprende la función que mejor mapea X → Y.

---

# 💹 Conexión con Finanzas

| Elemento | Rol |
|-----------|-----|
| Variables de entrada (`rv_5`, `rv_21`, `mom_21`) | Estado del mercado |
| Variable objetivo (`rv_fwd_21`) | Volatilidad futura |
| Modelo (Random Forest) | IA que aprende relaciones históricas |
| Resultado (`iv_hat`) | Volatilidad implícita pronosticada |
| Black–Scholes | Modelo analítico que usa esa IV |

➡️ IA proporciona **inputs inteligentes** a modelos financieros clásicos.

---

# 🔗 Integración con Black–Scholes

🔹 La IA pronostica la volatilidad `σ̂`  
🔹 Black–Scholes calcula el precio de la opción `C(S, K, T, r, σ̂)`  
🔹 Se obtienen los *Greeks* (Δ, Γ, vega, θ)  
🔹 Se ejecuta un *delta–hedging* con la IV pronosticada.

Resultado: un sistema **híbrido** IA + Finanzas Cuantitativas.

---

# 📊 Caso práctico — BTC–USD

Datos de `yfinance`: *Bitcoin (BTC–USD)*  
Periodo: 2016–2025  
Horizonte: 30 días (T = 30/252)

**Modelo:** `RandomForestRegressor`  
**Objetivo:** predecir volatilidad futura  
**Aplicación:** pricing y cobertura con θ incluido.

---

# 📈 Resultados principales

- MAE y RMSE alrededor de 0.02–0.04 (buen ajuste)
- Volatilidad pronosticada sigue bien los picos de mercado
- IV pronosticada correlaciona fuertemente con el precio de la opción
- PnL con θ incluido muestra cobertura más estable

![width:800px](https://upload.wikimedia.org/wikipedia/commons/e/e8/Random_forest_diagram_complete.png)

---

# 🧩 Random Forest como primera IA en finanzas

- Primer algoritmo **no paramétrico, interpretable y masivo** usado en bancos.  
- Adoptado entre 2005–2010 por **hedge funds cuantitativos**.  
- Mejoró predicciones de riesgo, crédito y volatilidad.  
- Aceptado por reguladores gracias a su **explicabilidad**.

📚 **Referencia:** Breiman, L. (2001) — *Random Forests*, *Machine Learning*, 45(1), 5–32.

---

# 🔮 Evolución posterior

1. **Gradient Boosting** (XGBoost, LightGBM) → árboles secuenciales.  
2. **Redes neuronales** → aprenden jerarquías más profundas.  
3. **Reinforcement Learning** → decisiones de cobertura adaptativas.  
4. **Explainable AI (XAI)** → interpretación de modelos.

---

# 🧠 Conclusión

- Random Forest fue el **puente entre estadística y aprendizaje automático**.  
- Permitió aplicar IA de forma **robusta, interpretable y práctica** en finanzas.  
- Hoy es la **base conceptual** de la IA moderna usada en derivados, riesgo y portafolios.

> “Los datos no solo informan los modelos financieros, los **enseñan**.”  


---

# 📚 Lecturas sugeridas

- **Breiman, L. (2001)** — *Random Forests*. Machine Learning.  
- **Gu, Kelly & Xiu (2020)** — *Empirical Asset Pricing via Machine Learning*. *RFS*.  
- **Hull (2022)** — *Machine Learning in Business and Finance*.  
- **Raschka (2023)** — *Machine Learning with PyTorch and Scikit-Learn*.  

---

# 🏁 Fin de la presentación

¡Gracias por tu atención!

**Herramientas:** yfinance + scikit-learn + matplotlib  



