---
marp: true
theme: gaia
paginate: false
class: lead
title: "Introduccion viernes de codigo"
author: "Juan Camilo Pinedo"
footer: "©2025 Universidad del Norte"
header: "Clase 1"
---


# Viernes de Codigo!🖥️🐍 💵⛏️


Juan Camilo Pinedo Campo


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



---

| Tipo    | Descripción breve                    | Ejemplo              | Uso típico                                 |
| ------- | ------------------------------------ | -------------------- | ------------------------------------------ |
| `int`   | Número entero (positivo o negativo)  | `42`, `-7`           | Conteos, índices, aritmética entera        |
| `float` | Número con decimales (coma flotante) | `3.14`, `-0.001`     | Cálculos con decimales (valores continuos) |
| `str`   | Cadena de texto Unicode              | `"Hola"`, `"Python"` | Manejo de texto, formateo, parsing         |
| `bool`  | Valor lógico booleano                | `True`, `False`      | Condiciones, lógica booleana, *flags*      |


---

### ¿Cuales son las caracteristicas de `list`, `tuple`, `set` y `dict`?

---

| Tipo  | Orden |    Duplicados | Mutable | Indexable | Uso típico                       |
| ----- | ----: | ------------: | ------: | --------: | -------------------------------- |
| list  |    Sí |            Sí |      Sí |        Sí | secuencia general                |
| tuple |    Sí |            Sí |      No |        Sí | paquete “sellado”                |
| set   |    No |            No |      Sí |        No | únicos / operaciones de conjunto |
| dict  |  Sí\* | claves únicas |      Sí | Por clave | mapeo clave→valor                |

---

### Veamos sus diferencias con ejemplos. 
- List

```python
# LIST → mutable, ordenada, permite duplicados
frutas = ["manzana", "pera", "manzana"]
frutas.append("uva")
print(frutas)  

['manzana', 'pera', 'manzana', 'uva']
```
---

- Tuple 
```python
coordenada = (10.5, 20.3)
print(coordenada[0])  

10.5
coordenada[0] = 15 → ERROR (no se puede cambiar)
```

- Set
```python
colores = {"rojo", "verde", "rojo"}
print(colores)

{'verde', 'rojo'} #(se eliminó duplicado)
```

---

- Dict
```python
persona = {"nombre": "Ana", "edad": 25, "edad": 30}
print(persona) 

{'nombre': 'Ana', 'edad': 30} #(la última clave reemplaza a la anterior)
```

---

### La Informacion es vital

<div style="display:flex; justify-content:center; gap 400px;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Stockexchange.jpg/640px-Stockexchange.jpg" width="450"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Trading_Floor_at_the_New_York_Stock_Exchange_during_the_Zendesk_IPO.jpg/640px-Trading_Floor_at_the_New_York_Stock_Exchange_during_the_Zendesk_IPO.jpg" width="450"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/NY_stock_exchange_telephone_operator_LC-USZ62-104021.jpg/640px-NY_stock_exchange_telephone_operator_LC-USZ62-104021.jpg" width="224"/>
</div>


---

## ¿Qué es una API y por qué importa?

-  Una API (Application Programming Interface) 
    - Es un puente que permite que distintos programas se comuniquen entre sí.

- Facilita acceder a datos de manera estructurada y automática.

- En finanzas, las APIs son vitales
    - Permiten actualización en tiempo real.
    - Reducen errores frente a la descarga manual.
    - Permiten la escalabilidad frente grandes volúmenes de datos.

--- 

### La SEC

- Este es el organismo regulador del mercado de valores en EE.UU

- Crea regulaciones prohibiciones y exigencias e.g las empresas que cotizan en bolsa deben entregar reportes obligatorios (10-K, 10-Q, 8-K, etc.).
 

---

## La ventaja de la SEC y su API : El flujo de datos (Empresa → SEC → API → Analista)


<img width="1200" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iOTAwIiBoZWlnaHQ9IjI0MCIgdmlld0JveD0iMCAwIDkwMCAyNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgcm9sZT0iaW1nIiBhcmlhLWxhYmVsPSJGbHVqbyBFbXByZXNhIGEgQW5hbGlzdGEgdsOtYSBTRUMgeSBBUEkiPgogIDxkZWZzPgogICAgPHN0eWxlPgogICAgICAuYm94IHsgZmlsbDogI2Y1ZjVmNTsgc3Ryb2tlOiAjODg4OyBzdHJva2Utd2lkdGg6IDI7IHJ4OiAxMDsgfQogICAgICAudGl0bGUgeyBmb250OiA3MDAgMTZweCBzYW5zLXNlcmlmOyBmaWxsOiAjMjIyOyB9CiAgICAgIC5zdWIgeyBmb250OiA0MDAgMTJweCBzYW5zLXNlcmlmOyBmaWxsOiAjNTU1OyB9CiAgICAgIC5hcnJvdyB7IHN0cm9rZTogIzY2Njsgc3Ryb2tlLXdpZHRoOiAyOyBtYXJrZXItZW5kOiB1cmwoI2Fycik7IGZpbGw6IG5vbmU7IH0KICAgIDwvc3R5bGU+CiAgICA8bWFya2VyIGlkPSJhcnIiIG1hcmtlcldpZHRoPSIxMCIgbWFya2VySGVpZ2h0PSIxMCIgcmVmWD0iOSIgcmVmWT0iMyIgb3JpZW50PSJhdXRvIj4KICAgICAgPHBvbHlnb24gcG9pbnRzPSIwIDAsIDEwIDMsIDAgNiIgZmlsbD0iIzY2NiIvPgogICAgPC9tYXJrZXI+CiAgPC9kZWZzPgoKICA8IS0tIEVtcHJlc2EgLS0+CiAgPHJlY3QgeD0iMjAiIHk9IjQwIiB3aWR0aD0iMTgwIiBoZWlnaHQ9IjkwIiBjbGFzcz0iYm94Ii8+CiAgPHRleHQgeD0iMTEwIiB5PSI3MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgY2xhc3M9InRpdGxlIj5FbXByZXNhPC90ZXh0PgogIDx0ZXh0IHg9IjExMCIgeT0iOTIiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGNsYXNzPSJzdWIiPjEwLUsgwrAgMTAtUSDigJMgOC1LPC90ZXh0PgogIDx0ZXh0IHg9IjExMCIgeT0iMTEwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0ic3ViIj5Fc3RhZG9zICZhbXA7IG5vdGFzPC90ZXh0PgoKICA8IS0tIFNFQyAtLT4KICA8cmVjdCB4PSIyNTAiIHk9IjQwIiB3aWR0aD0iMjMwIiBoZWlnaHQ9IjkwIiBjbGFzcz0iYm94Ii8+CiAgPHRleHQgeD0iMzY1IiB5PSI3MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgY2xhc3M9InRpdGxlIj5TRUMgLyBFREdBUjwvdGV4dD4KICA8dGV4dCB4PSIzNjUiIHk9IjkyIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0ic3ViIj5SZXBvc2l0b3JpbyBwdWJsaWNvPC90ZXh0PgogIDx0ZXh0IHg9IjM2NSIgeT0iMTEwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0ic3ViIj5DSUsuIEluZMOpY2VzIMO6IEF1ZGl0b3LDrWE8L3RleHQ+CgogIDwhLS0gQVBJIC0tPgogIDxyZWN0IHg9IjUyMCIgeT0iNDAiIHdpZHRoPSIxNzAiIGhlaWdodD0iOTAiIGNsYXNzPSJib3giLz4KICA8dGV4dCB4PSI2MDUiIHk9IjcwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0idGl0bGUiPkFQSSAoU0VDKTwvdGV4dD4KICA8dGV4dCB4PSI2MDUiIHk9IjkyIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0ic3ViIj5KU09OIMO6IEVuZHBvaW50czwvdGV4dD4KICA8dGV4dCB4PSI2MDUiIHk9IjExMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgY2xhc3M9InN1YiI+UmF0ZSBsaW1pdHM8L3RleHQ+CgogIDwhLS0gQW5hbGlzdGEgLS0+CiAgPHJlY3QgeD0iNzMwIiB5PSI0MCIgd2lkdGg9IjE1MCIgaGVpZ2h0PSI5MCIgY2xhc3M9ImJveCIvPgogIDx0ZXh0IHg9IjgwNSIgeT0iNzAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGNsYXNzPSJ0aXRsZSI+QW5hbGlzdGE8L3RleHQ+CiAgPHRleHQgeD0iODA1IiB5PSI5MiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgY2xhc3M9InN1YiI+UHl0aG9uIMO6IHBhbmRhczwvdGV4dD4KICA8dGV4dCB4PSI4MDUiIHk9IjExMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgY2xhc3M9InN1YiI+TGltcGllemEgJmFtcDs7IGluc2lnaHRzPC90ZXh0PgoKICA8IS0tIEZsZWNoYXMgLS0+CiAgPGxpbmUgeDE9IjIwMCIgeTE9Ijg1IiB4Mj0iMjUwIiB5Mj0iODUiIGNsYXNzPSJhcnJvdyIvPgogIDxsaW5lIHgxPSI0ODAiIHkxPSI4NSIgeDI9IjUyMCIgeTI9Ijg1IiBjbGFzcz0iYXJyb3ciLz4KICA8bGluZSB4MT0iNjkwIiB5MT0iODUiIHgyPSI3MzAiIHkyPSI4NSIgY2xhc3M9ImFycm93Ii8+CgogIDx0ZXh0IHg9IjQ1MCIgeT0iMTgwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBjbGFzcz0ic3ViIj4KICAgIFJlcG9ydGVzIOKAkyBQdWJsaWNhY2nDs24gcmVndWxhZGEg4oCTIEFjY2VzbyBwcm9ncmFtYWJsZSDigJMgQW7DoWxpc2lzIHJlcHJvZHVjaWJsZQogIDwvdGV4dD4KPC9zdmc+" />


---

### Vamos a programar!
