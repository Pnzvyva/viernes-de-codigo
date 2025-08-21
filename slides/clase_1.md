---
marp: true
theme: gaia
paginate: false
class: lead
title: "Título de la charla"
author: "Juan Camilo Pinedo"
footer: "©2025 Universidad del Norte"
header: "Clase 1"
---


# Viernes de Codigo!🖥️🐍 💵⛏️


Bienvenidos!


---

## ¿Por qué aprender Python?
![w:400](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExcHZrNGRiODdhMDYybzM5OXAzemxpaW55eTRnd24zamtsaHJ3dWZ1cyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/scZPhLqaVOM1qG4lT9/giphy.gif)

---

**Porque**  
**Es versátil**
**Es práctico**
**Es duradero**
**Es sencillo de leer**

---

## ¿Que capacidades clave tiene Python en el ambito de las finanzas?

- Ingesta de datos & APIs
- Limpieza y trazabilidad de datos 
- Series de tiempo y econometría 
- Backtesting y *factor research*
- Optimización de portafolios y riesgo
- ML para alpha / crédito / fraude
- NLP

---


## Temas para esta clase.


- Conceptos basicos de python 
- Mineria de datos financieros (Uso de APIs)


---
## Conceptos de programación en Python

- `int`, `float`, `bool`, `str` → **inmutables**
- `list` → **mutable**, ordenada
- `tuple` → inmutable, ordenada
- `set` → **mutable**, sin orden, elementos únicos
- `dict` → **mutable**, pares clave→valor

```python
x = 42        # int
pi = 3.1416   # float
ok = True     # bool
s = "Hola"    # str}
```

---


## ¿Cuales son las caracteristicas de los `int`, `float`, `str` y `bool`?

- **Entero**: número sin parte decimal (… -2, -1, 0, 1, 2 …).
- **Precisión arbitraria** en Python (no “se desborda” como en otros lenguajes).
- Operaciones: `+ - * // % **` (potencia).

```python
a = 42
b = -7
c = 2 ** 50         # enteros muy grandes
d = 7 // 3          # división entera = 2
r = 7 % 3           # residuo = 1
```
---

### ¿Qué es un `float`?

- Punto flotante.

- Tiene limitaciones de precisión binaria.

- Útil para cálculos numéricos; compara con tolerancia (`math.isclose`).
---

```python
x = 0.1 + 0.2
print(x)            # 0.30000000000000004
import math
math.isclose(x, 0.3)  # True

# Para dinero, preferir Decimal:
from decimal import Decimal
Decimal("0.10") + Decimal("0.20")  # Decimal('0.30')

```

---

| Tipo  | Orden |    Duplicados | Mutable | Indexable | Uso típico                       |
| ----- | ----: | ------------: | ------: | --------: | -------------------------------- |
| list  |    Sí |            Sí |      Sí |        Sí | secuencia general                |
| tuple |    Sí |            Sí |      No |        Sí | paquete “sellado”                |
| set   |    No |            No |      Sí |        No | únicos / operaciones de conjunto |
| dict  |  Sí\* | claves únicas |      Sí | Por clave | mapeo clave→valor                |


---
