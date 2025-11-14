---
marp: true
theme: einstein
paginate: true
title: Viernes de codigo
description: Trading algoritmico y trading de alta frecuencia
size: 16:9
math: mathjax
class: lead
---

 
# Viernes de código
## Trading algoritmico y alta frecuencia. ⚠️✳️
### _¿Es el mercado hoy en dia solo una ilusion?_
**Juan Camilo Pinedo Campo** · Universidad del Norte  
_7 de Noviembre del 2025_


---

# Estructura de la clase

1. Trading algoritmico

2. El crash de las 2:45 P.M 

3. Entendamos el HFT

4. ¿Es realmente este el futuro de los mercados?

5. ¿Cual es el panorama a nivel mundial?

---

# Trading Algoritmico.

<div class="multicolumn vcenter"><div>


Primero entendamos de que trata el trading algoritmico, es una forma de operar en los mercados usando algoritmos para ejecutar operaciones basadas en criterios **predefinidos**

Como funciona propiamente el "usar" algoritmos. 

- *Primero se elige un broker con API* 
  - Alpaca, IBKR, Binance, etc. 
- *Creas una cuenta y obtienes tus credenciales*   
  - *API Key:  PK1234567890ABCDEF (Ejemplo)*  
  - *Secret Key:  SKxyz987654321 (Ejemplo)*  
- *Lees documentacion del uso de la API y escribes codigo basico de conexion con el broker*
- *Dentro de la documentacion te muestran como debes enviar ordenes automaticas y te muestran como usar modos de **prueba*** 
</div><div>

![center w:320 rounded](/assets/mermaid-diagram.svg)

</div></div>

---

# Codigo ejemplo de conexion a API y ejecucion de orden.

<!-- _class: white-slide -->


<div class="multicolumn"><div>

```python
"""
Usa Python con librerías que facilitan la conexión.
pip install alpaca-trade-api   # para Alpaca
pip install ib_insync          # para Interactive Brokers
pip install python-binance  
"""

from alpaca_trade_api import REST
import os
from dotenv import load_dotenv

# Cargar claves desde archivo .env
load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # Usa "live" para cuenta real

# Conectarse al broker
api = REST(API_KEY, SECRET_KEY, BASE_URL)

# Probar conexión: ver saldo
account = api.get_account()
print(f"Saldo: ${account.cash}")
```

<figcaption> Ejemplo de conexion a una API.</figcaption>

</div><div>

```python
# Comprar 1 acción de Apple (AAPL)
api.submit_order(
    symbol="AAPL",
    qty=1,
    side="buy",
    type="market",
    time_in_force="gtc"  # Good till canceled
)

print("¡Orden enviada!")
```

<figcaption> Ejemplo de orden enviada</figcaption>

</div></div>

---



# Estrategias.

Existen muchas estrategias dentro del trading algorítmico.

<center>

<div style="height: 340px; overflow-y: auto; display: inline-block;">

| Estrategia | Descripción |
|------------|-------------|
| **Trend Following** | Sigue la tendencia con medias móviles o rupturas |
| **Momentum** | Compra lo que sube fuerte, vende lo que baja |
| **Mean Reversion** | Compra barato, vende caro: espera retorno al promedio |
| **Stat Arb** | Gana con diferencias entre activos correlacionados |
| **Market Making** | Gana del spread ofreciendo compra/venta continua |
| **VWAP** | Ejecuta al precio promedio del día (fondos) |
| **TWAP** | Divide órdenes grandes cada X minutos |
| **POV** | Sigue un % del volumen real del mercado |
| **Shortfall** | Ajusta velocidad según movimiento del precio |
| **Arbitrage** | Compra en A, vende en B al instante |
| **Index Rebal** | Aprovecha cambios masivos en índices |
| **News-Based** | Opera al leer noticias en milisegundos |
| **Sentiment** | Mide miedo/euforia en redes y prensa |
| **Machine Learning** | Predice con IA usando datos históricos |
| **Delta-Neutral** | Neutraliza riesgo de precio, gana con volatilidad |
| **Auto Hedging** | Cubre pérdidas automáticamente con derivados |

</div>

<tabcaption>Estrategias de trading algorítmico.</tabcaption>

</center>

---


# ¿Crash de las 2:45 P.M

<div class="multicolumn vcenter"><div>

Grecia estaba al borde del default pais, el deficit publico estaba disparado y necesitaban un rescate, ¿que se les ocurrio?

Para evitar el colapso economico y recibir ese rescate debia cumplir con las condiciones de organismo internacionales.

- Aumento de impuestos **(IVA, renta, combustibles)**.
- Reducción de pensiones y salarios públicos.
- Recortes en **funcionamiento** estatal. 
- **Importante:** la custodia — tú controlas la clave o un tercero la custodia.  

Las medidas generaron un **fuerte rechazo social**, protestas masivas y  eso se tradujo auna alta volatilidad en los mercados financieros.

</div><div>

![center w:320 rounded](https://i.pinimg.com/736x/7a/9d/c8/7a9dc8d570bb52255681748f887ac7fb.jpg)

</div></div>

---

## **🕛 12:00 P.M — Mercado comienza a ponerse inestable**
- Aumenta la volatilidad general.
- La incertidumbre por la crisis de Grecia entra en foco global.

---

## **🕐 1:00 P.M — NYSE activa mecanismo anti-volatilidad**
- El NYSE empieza a usar un sistema diseñado para **reducir el número de órdenes**.
- La meta era disminuir la volatilidad filtrando transacciones excesivas.

---

## **🕜 1:33 P.M – 2:25 P.M — Spoofing de Navinder Sarao**
- Navinder Sarao en Londres activa sus **bots de spoofing**.
- Coloca miles de órdenes falsas para mover el mercado a su favor.
- La manipulación crea más inestabilidad para los algoritmos.

---

## **🕝 2:30 P.M — La volatilidad se dispara +20%**
- Los indicadores de riesgo saltan abruptamente.
- Los algoritmos comienzan a reaccionar en cadena.

---

## **🕒 2:32 P.M — Waddell & Reed lanza orden de venta de $4.1B**
- Orden enorme de E-mini S&P 500 futuros.
- Usan un algoritmo que la divide en miles de micro-órdenes.

``If  
   el volumen de órdenes de venta aumenta  
then  
   vender más rápido``

- Este mecanismo crea **recursividad**: vender → más ventas → vender más.

---

## **🕞 2:37 P.M — La orden entra en modo “cascada”**
- El algoritmo acelera automáticamente el ritmo de ventas.
- Aumenta la presión bajista de forma exponencial.

---

## **🕣 2:42 P.M — HFT toma el control del mercado**
- Los creadores de mercado algorítmicos se retiran.
- Los HFT comienzan a operar entre sí en bucles veloces.
- El mercado se desancla del precio real.

---

## **🕓 2:47 P.M — El S&P 500 cae -9%**
- Se evaporan **1 trillón de dólares** en minutos.
- Entre 2:45 P.M y 2:47 P.M se transan **2 mil millones de acciones**.
- El Flash Crash queda registrado como una caída histórica.

---

# ¿Qué tiene que ver el HFT en todo esto?

- Las protestas en Grecia estaban ocurriendo en televisión en vivo.  
- El miedo al contagio financiero europeo ya estaba estresando al mercado.  
- Las instituciones financieras de EE.UU. tenían exposición a bancos de:  
  - **Francia**  
  - **Alemania**  
  - **Suiza**  
- Esos bancos estaban cargados con **deuda griega**.  
- Cuando el mercado entró en modo automático, los algoritmos amplificaron la caída:
  - Spoofing (Sarao)  
  - Venta masiva automatizada (Waddell & Reed)  
  - Retiro de liquidez (HFT)  
  - Bucles entre máquinas

El resultado: **un colapso algorítmico acelerado por el contexto internacional.**


---

# El HFT es una forma de trading algorítmico que usa:

<div class="multicolumn vcenter"><div>

- Computadoras ultrarrápidas  
- Latencias medidas en nanosegundos  
- Estrategias automatizadas  
- Operaciones masivas por segundo  

Su objetivo:  
**aprovechar micro-ineficiencias del mercado antes que cualquier otro participante.**


</div><div>

![center w:450](https://i.pinimg.com/1200x/ed/58/a9/ed58a90c4fbe9cc41b0f7191647d45d9.jpg)



</div></div>




---

# HFT en 2010 — ¿Cómo funcionaba?

<div class="multicolumn vcenter"><div>

En esa época, el HFT dependía principalmente de:

## 1. Velocidad pura (latencia)
- Comprar servidores *colocados físicamente* junto a las bolsas.
- Latencias tan bajas como 1–5 microsegundos.
- "Quien llegaba primero, ganaba".

## 2. Market Making automatizado
- Bots ofreciendo *bid* y *ask* constantemente.
- Ganancia en el spread (centavos).



</div><div>

## 3. Rebotes entre algoritmos
- Mucho del volumen real eran **algoritmos interactuando entre sí**.
- Si uno se retiraba del mercado → caída de liquidez → desorden.

## 4. Problemas:
- Spoofing (órdenes falsas)
- Quote stuffing (miles de órdenes por segundo para congestionar)
- Retiro instantáneo de liquidez
- Falta de supervisión regulatoria



</div></div>

---

# HFT Hoy: Tecnología más robusta

<div class="multicolumn vcenter"><div>


Hoy el HFT usa:

1. Colocation + fibra + microondas
- Redes de microondas Chicago–NY.
- Latencias incluso < 1 microsegundo.

2. Modelos estadísticos avanzados
- Microestructura del mercado.
- Predicción de order flow.
- Modelos que anticipan qué va a hacer otro algoritmo.

3. Mayor regulación
- Regulación anti-spoofing (Dodd-Frank).
- Circuit breakers.
- Límite al quote-to-trade ratio.




</div><div>

![center w:350](https://i.pinimg.com/1200x/4e/0c/60/4e0c60580dbc7425209a55792238b6cd.jpg)


---

# ¿Es realmente esto el futuro de los mercados?

<div class="multicolumn vcenter"><div>


La carrera por la velocidad
- Competencia mundial por reducir la latencia al mínimo.
- Uso de redes de **microondas**, **láser**, **fibra óptica de baja dispersión**.
- Centros de datos colocados estratégicamente (“**colocation**”).
- Hardware especializado: FPGA, servidores optimizados, NICs de nanosegundos.

Evolución de los algoritmos
- Pasaron de simples estrategias de arbitraje a:
  - Modelos de microestructura del mercado.
  - Predicción de ordenes (*order flow prediction*).
  - Arbitrage estadístico ultrarrápido.
- Enfoque moderno: **anticipar la intención** de otros algoritmos.
- Integración de machine learning en escalas sub-milisegundo.

</div><div align=center>

<iframe src="https://assets.pinterest.com/ext/embed.html?id=422281212082179" height="520" width="236" frameborder="0" scrolling="no" ></iframe>
<p>
<figcaption> Animacion de arbol.

</div></div>


---

<div class="multicolumn vcenter"><div>


Interconexión global de mercados
- Precios en Estados Unidos, Europa y Asia responden en microsegundos.
- El HFT ya no opera en un solo mercado: opera en **ecosistemas conectados**.
- Las bolsas compiten para ofrecer la menor latencia del planeta.
- La infraestructura global se vuelve un factor de riesgo:  
  **si un mercado se atrasa, los algoritmos lo castigan.**

</div><div>

El rol del hardware
- Servidores ultra-optimzados:  
  - CPU de baja latencia  
  - RAM ECC de respuesta inmediata  
  - Kernel Linux real-time  
- FPGAs y ASICs:  
  - Procesan órdenes sin pasar por el sistema operativo.  
  - “Algoritmos embebidos en hardware”: velocidad imposible para software tradicional.

</div></div>



---

<div class="multicolumn vcenter"><div>

Fragilidad estructural en eventos raros
- El sistema global depende de sincronización perfecta entre:
  - datos → redes → hardware → algoritmos.
- Pequeños retrasos pueden iniciar cascadas:
  - “**Liquidity Gaps**”  
  - “**Stop Logic Events**”  
  - Mini-flash crashes en activos aislados  
- Los mercados modernos son **demasiado rápidos para que un humano intervenga**.

</div><div>

El futuro técnico del HFT
- Integración de IA acelerada por hardware.
- Redes ópticas directas entre bolsas (NY ↔ Londres ↔ Tokio).
- Algoritmos autorregulados capaces de medir su propio impacto.
- Mercados donde **la toma de decisiones es 100% algorítmica**.
- Desafío principal:  
  **mantener velocidad sin sacrificar estabilidad sistémica.**

</div></div>

---

# ¿Los paises como han reacionado a este fenómeno?

¿Qué muestran los datos globales?
- El trading algorítmico domina los mercados modernos.
- El HFT representa una fracción relevante del AT, según la bolsa.
- Asia tiene restricciones más fuertes (especialmente China).
- Los porcentajes varían según:
  - tipo de activo (acciones vs. derivados)
  - horario de mercado
  - regulación de cada país

---
# 🇺🇸 Estados Unidos — NYSE & NASDAQ

## **AT: 60% – 75% del volumen total**
- La mayoría del flujo proviene de algoritmos institucionales.
- Dominan estrategias como market making, VWAP, TWAP y predicción de flujo.

## **HFT: 50% – 55% del volumen en equities**
- EE.UU. es el mercado más maduro para HFT.
- Fuerte presencia de firmas como Citadel, Virtu, Jump Trading, Hudson River Trading.

**Notas:**  
- El HFT es especialmente dominante en futuros (E-mini S&P 500).  
- Competencia basada en latencia, co-location y microestructura.

---

# 🇬🇧 Reino Unido — London Stock Exchange (LSE)

## **AT: 50% – 65% del volumen**
- Londres es un hub global de trading electrónico.
- Amplio uso de estrategias algorítmicas institucionales.

## **HFT: 35% – 45%**
- Compatible con mercados europeos.
- Alta competencia en market making de blue chips.

**Notas:**  
- Gran presencia de firmas estadounidenses operando en Londres.  
- El Brexit no redujo significativamente la actividad algorítmica.

---

# 🇩🇪🇫🇷 Europa Continental — Xetra / Euronext

## **AT: 50% – 70%**
- Mercados muy automatizados en acciones europeas.
- Xetra (Alemania) es uno de los sistemas electrónicos más avanzados.

## **HFT: 30% – 40%**
- Más restricciones que EE.UU., pero volumen relevante.
- Futuros europeos (Euro Stoxx, DAX) tienen alta actividad HFT.

**Notas:**  
- La regulación MiFID II exige mayor transparencia.  
- Limitaciones al quote-to-trade ratio afectan estrategias HFT puras.

---
# 🇯🇵 Japón — Tokyo Stock Exchange (TSE)

## **AT: 40% – 55%**
- TSE es tecnológicamente avanzado.
- Creciente adopción de ejecución algorítmica institucional.

## **HFT: 20% – 30%**
- Importante, pero menor que en EE.UU./Europa.

**Notas:**  
- Japón adoptó medidas anti-volatilidad tras eventos de 2012–2014.  
- HFT está más regulado y monitoreado.

---
# 🇭🇰 Hong Kong — HKEX

## **AT: 35% – 50%**
- Mercado mixto: institucional y minorista.

## **HFT: 15% – 25%**
- Restricciones regulatorias frenan la adopción masiva.
- Menor divulgación de datos públicos.

**Notas:**  
- HKEX limita ciertos comportamientos algorítmicos en horarios de bajo volumen.  
- El HFT está permitido, pero bajo fuerte supervisión


---
# 🇨🇳 China — Shanghai & Shenzhen (SSE / SZSE)

## **AT: 25% – 45%**
- China usa algoritmos institucionales, pero con fuerte regulación.

## **HFT: 5% – 10% (muy restringido)**
- Operaciones ultrarrápidas están limitadas por:
  - controles de cancelaciones de órdenes
  - monitoreo estricto del order-to-trade ratio
  - penalidades por actividad “excesiva”

**Notas:**  
- El gobierno chino desalienta explícitamente el HFT puro.  
- Alta presencia de minoristas reduce la cuota algorítmica.

---
# 🇮🇳 India — NSE & BSE

## **AT: 35% – 60%**
- El NSE es uno de los mercados electrónicos más activos del mundo.
- Gran crecimiento del trading algorítmico institucional.

## **HFT: 15% – 25%**
- Permitido, pero con restricciones en latencia y cancelaciones.

**Notas:**  
- India implementa controles estrictos de co-location.  
- Demanda de fibra y latencia baja está creciendo.


---

# Conclusión Global: ¿Qué muestran estos porcentajes

## Tendencia: todo apunta a más automatización.
La dirección global es clara:
**más AT, más HFT, más automatización — pero con diferentes ritmos según el país.**

Reflexion ¿Creen que esto es verdaderamente lo mejor para el mundo financiero?


---

#  Referencias y lecturas recomendadas

<div class="multicolumn vcenter"><div>

### Libros clave

- **Lewis, M.** (2014). *Flash Boys*. W. W. Norton.

- **Chan, E. P.** (2017). *Machine Trading*. Wiley. 

- **Chan, E. P.** (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.  
- **Hilpisch, Y.** (2020). *Python for Algorithmic Trading*. O’Reilly.

</div><div>

### 🌐 Artículos

- **Brogaard, Hendershott & Riordan (2014)** — “High Frequency Trading and Price Discovery.”  
- **Hendershott, Jones & Menkveld (2011)** — “Does Algorithmic Trading Improve Liquidity?”  
- **Kirilenko, Kyle, Samadi & Tuzun (2017)** — “The Flash Crash: High-Frequency Trading in an Electronic Market.”  

---

🐍 🐍 🐍 🐍
=

## Muchas gracias a todos.






